"""Tune the tabular baseline models with Optuna and patient-grouped CV.

Each trial samples hyperparameters, scores them with the shared evaluation
function (``src.models.evaluate.evaluate_model``: five patient-grouped folds
on the training set, pooled out-of-fold predictions), and Optuna's TPE
sampler proposes the next trial from the scores so far.

The objective is ROC-AUC, the main metric of the project: it does not depend
on the 0.5 threshold and accuracy is misleading with 14% declines. Scores
of the best trial are optimistic because the same folds select and score
the hyperparameters, so the held-out test set is used once, at the end, to
compare the selected models.

Usage:
    python -m src.models.tune --trials 30
    python -m src.models.tune --models random_forest --feature-sets combined --trials 50
"""

import argparse
import time
from datetime import datetime, timezone

import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.models.evaluate import evaluate_model

FEATURE_SETS = {
    "metadata-only": config.MODEL_METADATA_FEATURES,
    "combined": config.MODEL_METADATA_FEATURES + config.MRI_SCAN_FEATURES,
}


def logistic_regression(trial: optuna.Trial):
    """L1/L2-regularized logistic regression on standardized features."""
    penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=trial.suggest_float("C", 1e-3, 10.0, log=True),
        l1_ratio=1.0 if penalty == "l1" else 0.0, solver="saga",
        class_weight="balanced", max_iter=5000))


def random_forest(trial: optuna.Trial):
    """Random forest; leaf size and depth control overfitting on small data."""
    return RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 600, step=100),
        max_depth=trial.suggest_int("max_depth", 2, 12),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
        max_features=trial.suggest_float("max_features", 0.2, 1.0),
        class_weight="balanced", random_state=config.RANDOM_STATE, n_jobs=-1)


def gradient_boosting(trial: optuna.Trial):
    """Histogram gradient boosting with shrinkage and L2 regularization."""
    return HistGradientBoostingClassifier(
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_iter=trial.suggest_int("max_iter", 50, 400, step=50),
        max_depth=trial.suggest_int("max_depth", 2, 6),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 40),
        l2_regularization=trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
        class_weight="balanced", random_state=config.RANDOM_STATE)


MODEL_SPACES = {
    "logistic_regression": logistic_regression,
    "random_forest": random_forest,
    "gradient_boosting": gradient_boosting,
}


def tune(model_name: str, features: list[str], n_trials: int) -> optuna.Study:
    """Run one Optuna study and store every trial's full metrics on it."""
    def objective(trial: optuna.Trial) -> float:
        scores = evaluate_model(MODEL_SPACES[model_name](trial), features)
        for metric, value in scores.items():
            trial.set_user_attr(metric, value)
        return scores["roc_auc"]

    sampler = optuna.samplers.TPESampler(seed=config.RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return study


def main() -> None:
    """Tune every requested (model, feature set) pair and save the best trials."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--models", nargs="+", choices=list(MODEL_SPACES),
                        default=list(MODEL_SPACES))
    parser.add_argument("--feature-sets", nargs="+", choices=list(FEATURE_SETS),
                        default=list(FEATURE_SETS))
    parser.add_argument("--wandb", choices=["online", "offline", "disabled"], default="offline")
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    rows = []
    for set_name in args.feature_sets:
        for model_name in args.models:
            start = time.perf_counter()
            study = tune(model_name, FEATURE_SETS[set_name], args.trials)
            seconds = time.perf_counter() - start
            best = study.best_trial
            rows.append({"feature_set": set_name, "model": model_name,
                         **{k: round(v, 3) for k, v in best.user_attrs.items()},
                         "trials": args.trials, "seconds": round(seconds, 1),
                         "seconds_per_trial": round(seconds / args.trials, 2),
                         "best_params": best.params})
            print(f"{set_name:>14} | {model_name:<20} ROC-AUC {best.value:.3f} "
                  f"({seconds:.0f} s, {args.trials} trials)")

            if args.wandb != "disabled":
                import wandb
                run = wandb.init(entity=config.WANDB_ENTITY, project=config.WANDB_PROJECT,
                                 name=f"tuned {model_name} | {set_name}", group="tuning",
                                 config={"model": model_name, "feature_set": set_name,
                                         "trials": args.trials, **best.params},
                                 mode=args.wandb, dir=config.PROJECT_ROOT)
                for trial in study.trials:
                    run.log({"trial": trial.number, "trial_roc_auc": trial.value})
                run.summary.update(best.user_attrs)
                run.finish()

    results = pd.DataFrame(rows)
    results["created_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    results.to_csv(config.TUNING_RESULTS_CSV, index=False)
    print(f"Saved {config.TUNING_RESULTS_CSV}")


if __name__ == "__main__":
    main()

"""Train and evaluate the CNN models with patient-grouped cross-validation.

Uses the same protocol as the tabular baselines so the results are directly
comparable: only training patients are used, the five ``cv_fold`` groups
from the ML-ready dataset are the validation folds, and the metrics are
computed on the pooled out-of-fold probabilities with a 0.5 threshold.

Every fold runs a fixed number of epochs (no early stopping on the
validation fold), so the out-of-fold scores are not tuned on the data they
are measured on. Per-epoch losses and the wall-clock time of each epoch are
logged to Weights & Biases and summarized in ``reports/cnn_timing.json`` to
estimate the cost of the experiment plan.

Usage:
    python -m src.data.make_mri_volumes                       # run once first
    python -m src.models.train_cnn --model mri --epochs 10    # Model 1
    python -m src.models.train_cnn --model multimodal         # Model 3
    python -m src.models.train_cnn --model mri --wandb online # log to the team project
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch import nn
from torch.utils.data import DataLoader

from src import config
from src.data.make_mri_volumes import load_volumes
from src.models.cnn import BrainCNN, VisitPairDataset, count_parameters

FEATURE_SET_NAMES = {"mri": "mri-volume (CNN)", "multimodal": "mri-volume + metadata (CNN)"}


@dataclass
class TrainConfig:
    """Hyperparameters of one CNN experiment (logged with every run)."""

    model: str = "mri"  # "mri" (Model 1) or "multimodal" (Model 3)
    epochs: int = 10
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.3
    channels: tuple[int, ...] = (8, 16, 32, 64)
    seed: int = config.RANDOM_STATE
    # Apple MPS is not used by default: its 3D convolution and pooling kernels
    # were 2.5x slower than the CPU for this network (M2, PyTorch 2.14).
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


def classification_metrics(y: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    """Same metrics, names, and threshold as ``src.models.evaluate``."""
    preds = (probs >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y, probs),
        "pr_auc": average_precision_score(y, probs),
        "recall": recall_score(y, preds, zero_division=0),
        "precision": precision_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
        "accuracy": accuracy_score(y, preds),
    }


def load_training_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Return the training visit pairs and the MRI volume of each pair's visit t."""
    pairs = pd.read_csv(config.ML_READY_CSV)
    train = pairs[pairs["split"] == "train"].reset_index(drop=True)
    volumes = load_volumes()
    missing = set(train["mri_id"]) - set(volumes)
    if missing:
        raise ValueError(f"No MRI volume for {len(missing)} sessions, e.g. {sorted(missing)[:3]}")
    return train, np.stack([volumes[mri_id] for mri_id in train["mri_id"]])


def standardize(fit: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    """Scale features with the mean and SD of ``fit`` (the training folds only)."""
    mean, std = fit.mean(axis=0), fit.std(axis=0)
    std[std == 0] = 1.0
    return [(x - mean) / std for x in (fit, *others)]


def run_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: str,
              optimizer: torch.optim.Optimizer | None = None) -> tuple[float, np.ndarray]:
    """Run one pass over ``loader``; trains when an optimizer is given.

    Returns the mean loss and the predicted probabilities in loader order.
    """
    model.train(optimizer is not None)
    total_loss, probs = 0.0, []
    with torch.set_grad_enabled(optimizer is not None):
        for volume, tabular, label in loader:
            volume, tabular, label = volume.to(device), tabular.to(device), label.to(device)
            logits = model(volume, tabular)
            loss = loss_fn(logits, label)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(label)
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
    return total_loss / len(loader.dataset), np.concatenate(probs)


def fit_and_predict(cfg: TrainConfig, train_volumes: np.ndarray, train_tabular: np.ndarray | None,
                    train_y: np.ndarray, val_volumes: np.ndarray | None = None,
                    val_tabular: np.ndarray | None = None, val_y: np.ndarray | None = None,
                    log=None, prefix: str = "") -> tuple[nn.Module, np.ndarray | None, list]:
    """Train one model for ``cfg.epochs`` epochs and predict the validation set.

    The positive class is weighted by the stable/declined ratio of the
    training data, so both classes contribute equally to the loss (the
    same idea as ``class_weight="balanced"`` in the baselines).

    Returns:
        The trained model, the validation probabilities (None without a
        validation set), and one dict of losses and timing per epoch.
    """
    torch.manual_seed(cfg.seed)
    n_tabular = 0 if train_tabular is None else train_tabular.shape[1]
    model = BrainCNN(n_tabular, cfg.channels, dropout=cfg.dropout).to(cfg.device)
    pos_weight = torch.tensor((train_y == 0).sum() / max((train_y == 1).sum(), 1),
                              dtype=torch.float32, device=cfg.device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(VisitPairDataset(train_volumes, train_tabular, train_y, augment=True),
                              batch_size=cfg.batch_size, shuffle=True, generator=generator)
    val_loader = (DataLoader(VisitPairDataset(val_volumes, val_tabular, val_y),
                             batch_size=cfg.batch_size) if val_volumes is not None else None)

    history, val_probs = [], None
    for epoch in range(1, cfg.epochs + 1):
        start = time.perf_counter()
        train_loss, _ = run_epoch(model, train_loader, loss_fn, cfg.device, optimizer)
        record = {"epoch": epoch, "train_loss": train_loss}
        if val_loader is not None:
            val_loss, val_probs = run_epoch(model, val_loader, loss_fn, cfg.device)
            record["val_loss"] = val_loss
            record["val_roc_auc"] = (roc_auc_score(val_y, val_probs)
                                     if len(np.unique(val_y)) == 2 else float("nan"))
        record["epoch_seconds"] = time.perf_counter() - start
        history.append(record)
        if log is not None:
            log({f"{prefix}{k}": v for k, v in record.items()})
    return model, val_probs, history


def cross_validate(cfg: TrainConfig, log=None) -> dict:
    """Run patient-grouped CV on the training set and return metrics and timing."""
    train, volumes = load_training_data()
    y = train[config.TARGET_COLUMN].to_numpy()
    tabular = (train[config.MODEL_METADATA_FEATURES].to_numpy(dtype=float)
               if cfg.model == "multimodal" else None)

    oof_probs = np.zeros(len(train))
    histories, fold_seconds = [], []
    for fold in sorted(train["cv_fold"].unique()):
        is_val = (train["cv_fold"] == fold).to_numpy()
        tab_train = tab_val = None
        if tabular is not None:
            tab_train, tab_val = standardize(tabular[~is_val], tabular[is_val])

        start = time.perf_counter()
        _, probs, history = fit_and_predict(
            cfg, volumes[~is_val], tab_train, y[~is_val],
            volumes[is_val], tab_val, y[is_val], log=log, prefix=f"fold{fold}/")
        fold_seconds.append(time.perf_counter() - start)
        oof_probs[is_val] = probs
        histories.append(history)
        print(f"fold {fold}: val ROC-AUC {history[-1]['val_roc_auc']:.3f} "
              f"({fold_seconds[-1]:.0f} s)")

    epoch_seconds = [h["epoch_seconds"] for history in histories for h in history]
    predictions = train[["subject_id", "mri_id", "cv_fold", "age", "sex_male",
                         config.TARGET_COLUMN]].assign(predicted_risk=oof_probs)
    return {
        "metrics": classification_metrics(y, oof_probs),
        "predictions": predictions,
        "histories": histories,
        "timing": {
            "device": cfg.device,
            "epochs_per_fold": cfg.epochs,
            "folds": len(fold_seconds),
            "train_pairs": int(len(train)),
            "parameters": count_parameters(BrainCNN(
                0 if tabular is None else tabular.shape[1], cfg.channels)),
            "mean_epoch_seconds": float(np.mean(epoch_seconds)),
            "total_cv_seconds": float(np.sum(fold_seconds)),
        },
    }


def train_final_model(cfg: TrainConfig) -> None:
    """Fit on all training pairs and save the weights and scaling statistics.

    The held-out test set is not touched; it is reserved for the final
    evaluation of the selected models.
    """
    train, volumes = load_training_data()
    y = train[config.TARGET_COLUMN].to_numpy()
    tabular, scaling = None, None
    if cfg.model == "multimodal":
        raw = train[config.MODEL_METADATA_FEATURES].to_numpy(dtype=float)
        scaling = {"mean": raw.mean(axis=0).tolist(), "std": raw.std(axis=0).tolist()}
        (tabular,) = standardize(raw)
    model, _, _ = fit_and_predict(cfg, volumes, tabular, y)
    config.MODELS_DIR.mkdir(exist_ok=True)
    path = config.MODELS_DIR / f"cnn_{cfg.model}.pt"
    torch.save({"state_dict": model.state_dict(), "config": asdict(cfg),
                "tabular_features": config.MODEL_METADATA_FEATURES if tabular is not None else [],
                "scaling": scaling}, path)
    print(f"Saved {path}")


def save_results(cfg: TrainConfig, result: dict) -> None:
    """Append the CV metrics to the results table; write predictions and timing."""
    row = {"feature_set": FEATURE_SET_NAMES[cfg.model], "model": "cnn_3d",
           **{k: round(v, 3) for k, v in result["metrics"].items()},
           "epochs": cfg.epochs, "lr": cfg.lr, "batch_size": cfg.batch_size,
           "seconds": round(result["timing"]["total_cv_seconds"], 1),
           "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    path = config.CNN_RESULTS_CSV
    table = pd.concat([pd.read_csv(path), pd.DataFrame([row])]) if path.exists() \
        else pd.DataFrame([row])
    table.to_csv(path, index=False)
    result["predictions"].to_csv(
        config.REPORTS_DIR / config.CNN_OOF_TEMPLATE.format(model=cfg.model), index=False)

    timing = json.loads(config.CNN_TIMING_JSON.read_text()) \
        if config.CNN_TIMING_JSON.exists() else {}
    timing[cfg.model] = result["timing"]
    config.CNN_TIMING_JSON.write_text(json.dumps(timing, indent=2) + "\n")


def parse_args() -> tuple[TrainConfig, argparse.Namespace]:
    """Read the experiment settings from the command line."""
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=list(FEATURE_SET_NAMES), default=defaults.model)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--wandb", choices=["online", "offline", "disabled"], default="offline",
                        help="offline runs can be uploaded later with `wandb sync wandb/`")
    parser.add_argument("--save-model", action="store_true",
                        help="also fit on all training pairs and save to models/")
    args = parser.parse_args()
    cfg = TrainConfig(model=args.model, epochs=args.epochs, batch_size=args.batch_size,
                      lr=args.lr, weight_decay=args.weight_decay, dropout=args.dropout,
                      seed=args.seed, device=args.device)
    return cfg, args


def main() -> None:
    """Run one cross-validated CNN experiment and record its results."""
    cfg, args = parse_args()
    run = None
    if args.wandb != "disabled":
        import wandb
        run = wandb.init(entity=config.WANDB_ENTITY, project=config.WANDB_PROJECT,
                         name=f"cnn_3d | {FEATURE_SET_NAMES[cfg.model]}", group="cnn",
                         config=asdict(cfg), mode=args.wandb, dir=config.PROJECT_ROOT)

    result = cross_validate(cfg, log=run.log if run else None)
    save_results(cfg, result)
    if run is not None:
        run.summary.update({**result["metrics"], **result["timing"]})
        run.finish()
    if args.save_model:
        train_final_model(cfg)

    print(json.dumps({k: round(v, 3) for k, v in result["metrics"].items()}, indent=2))
    print(json.dumps(result["timing"], indent=2))


if __name__ == "__main__":
    main()

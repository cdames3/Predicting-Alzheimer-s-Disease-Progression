"""Compare every model's cross-validated ROC-AUC in one figure.

Reads the result tables written by the baseline notebook, the tuning script,
and the CNN training script, keeps the latest run of each (feature set,
model) pair, and draws one horizontal bar per model with chance level as
the reference line.

Output:
    reports/figures/16_model_comparison.png

Usage:
    python -m src.visualization.model_comparison
"""

import pandas as pd

from src import config
from src.visualization.report_layout import (DECLINED_COLOR, STABLE_COLOR, TEXT_SECONDARY,
                                             plt, set_style)

BASELINE_RESULTS = config.REPORTS_DIR / "baseline_results.csv"
FIGURE = config.FIGURES_DIR / "16_model_comparison.png"
TUNED_COLOR = "#123c6e"  # darker shade of the baseline blue


def load_results() -> pd.DataFrame:
    """Stack the baseline, tuned, and CNN results with a ``stage`` label."""
    frames = []
    baselines = pd.read_csv(BASELINE_RESULTS)
    frames.append(baselines[baselines["model"] != "dummy"].assign(stage="baseline"))
    if config.TUNING_RESULTS_CSV.exists():
        frames.append(pd.read_csv(config.TUNING_RESULTS_CSV).assign(stage="tuned"))
    if config.CNN_RESULTS_CSV.exists():
        cnn = pd.read_csv(config.CNN_RESULTS_CSV)
        frames.append(cnn.drop_duplicates(["feature_set", "model"], keep="last")
                      .assign(stage="CNN"))
    results = pd.concat(frames, ignore_index=True)
    results["label"] = (results["model"].str.replace("_", " ") + " | "
                        + results["feature_set"]
                        + results["stage"].map({"tuned": " (tuned)"}).fillna(""))
    return results.sort_values("roc_auc").reset_index(drop=True)


def main() -> None:
    """Draw and save the comparison figure."""
    set_style()
    results = load_results()
    dummy_auc = pd.read_csv(BASELINE_RESULTS).query("model == 'dummy'")["roc_auc"].iloc[0]
    colors = results["stage"].map({"baseline": STABLE_COLOR, "tuned": TUNED_COLOR,
                                   "CNN": DECLINED_COLOR})

    fig, ax = plt.subplots(figsize=(10, 0.32 * len(results) + 1.4))
    ax.barh(results["label"], results["roc_auc"], color=colors, height=0.65)
    for y, value in enumerate(results["roc_auc"]):
        ax.text(value + 0.004, y, f"{value:.3f}", va="center", fontsize=8, color=TEXT_SECONDARY)
    ax.axvline(0.5, color=TEXT_SECONDARY, linestyle="--", linewidth=1, zorder=3)
    ax.set_xlim(0.4, max(0.75, results["roc_auc"].max() + 0.05))
    ax.set_xlabel("ROC-AUC (pooled out-of-fold, patient-grouped 5-fold CV)")
    ax.set_title(f"ROC-AUC of every model (dashed = chance; dummy floor = {dummy_auc:.3f})",
                 loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in
               (STABLE_COLOR, TUNED_COLOR, DECLINED_COLOR)]
    ax.legend(handles, ["Baseline", "Tuned (Optuna)", "3D CNN"], loc="lower right",
              frameon=False, fontsize=8)
    fig.tight_layout()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=150, bbox_inches="tight")
    print(f"Saved {FIGURE}")


if __name__ == "__main__":
    main()

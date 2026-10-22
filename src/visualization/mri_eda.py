"""Section 4 of the EDA report: the OASIS-2 structural MRI scans.

The pages cover the scan inventory and acquisition format, example images by
dementia rating, intensity and quality statistics, and how the scan-derived
CSF proxy relates to brain volume, CDR, and the target.

The tables come from ``python -m src.data.make_mri_dataset``; the pages are
built by ``python -m src.visualization.eda_report``.
"""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src import config
from src.data.make_mri_dataset import QC_IQR_MULTIPLIER
from src.visualization.report_layout import (
    CLASS_PALETTE, DECLINED_COLOR, STABLE_COLOR, TEXT_SECONDARY, add_class_legend,
    add_notes, new_page, style_table,
)

if TYPE_CHECKING:
    from src.visualization.eda_report import ReportData

CDR_LEVELS = [0.0, 0.5, 1.0, 2.0]
VIEWS = ["sagittal", "coronal", "axial"]


def display(image: np.ndarray, view: str) -> np.ndarray:
    """Rotate a stored slice so the head is upright (face left / front on top)."""
    image = image.astype(np.float32)
    return np.rot90(image, 2) if view == "sagittal" else np.rot90(image, 1)


def page_inventory(data: "ReportData", title: str, section: str) -> plt.Figure:
    """Acquisition metadata, scans per session, sessions per patient, and CDR."""
    scans, sessions, meta = data.scans, data.sessions, data.mri_meta
    fig = new_page(title, "Every scan shares one acquisition format, so the volumes can "
                   "be stacked without resampling", section)
    ax = fig.add_axes([0.05, 0.7, 0.9, 0.16])
    ax.axis("off")
    variants = meta["header_variants"]
    counts = scans.groupby(list(variants[0])).size().rename("scans").reset_index()
    rows = [[f"{r.dim_x} x {r.dim_y} x {r.dim_z}",
             f"{r.voxel_x_mm:g} x {r.voxel_y_mm:g} x {r.voxel_z_mm:g} mm",
             "float32" if r.dtype.endswith("f4") else r.dtype, f"{r.scans:,}"]
            for r in counts.itertuples()]
    rows.append(["Format: Analyze 7.5 (.img + .hdr)", "T1-weighted MPRAGE, sagittal",
                 "big-endian", ""])
    table = ax.table(cellText=rows, colLabels=["Dimensions (x, y, z)", "Voxel size",
                                               "Data type", "Scans"],
                     loc="upper center", cellLoc="center")
    style_table(table, font_size=10, row_height=1.7)

    left, middle, right = fig.subplots(1, 3, gridspec_kw={
        "left": 0.06, "right": 0.97, "top": 0.6, "bottom": 0.1, "wspace": 0.35})
    panels = [
        (left, sessions["n_scans"].value_counts().sort_index(),
         "Repeated scans per session", "Sessions"),
        (middle, sessions.groupby("subject_id").size().value_counts().sort_index(),
         "MRI sessions per patient", "Patients"),
        (right, sessions["CDR"].value_counts().sort_index(),
         "Sessions by CDR at scan", "Sessions"),
    ]
    for ax, counts, subtitle, ylabel in panels:
        labels = [f"{k:g}" for k in counts.index]
        bars = ax.bar(labels, counts.values, color=STABLE_COLOR)
        ax.bar_label(bars, padding=2, fontsize=8)
        ax.set_title(subtitle, loc="left", fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, counts.max() * 1.15)
    return fig


def page_examples(data: "ReportData", title: str, section: str) -> plt.Figure:
    """Central slices of one typical session at each CDR level."""
    sessions = data.sessions
    fig = new_page(title, "Central slices of the first scan in one session per CDR "
                   "level (raw: not skull-stripped or registered)", section)
    slices = np.load(config.MRI_SLICES)
    axes = fig.subplots(len(VIEWS), len(CDR_LEVELS), gridspec_kw={
        "left": 0.08, "right": 0.98, "top": 0.82, "bottom": 0.05,
        "hspace": 0.08, "wspace": 0.04})
    for col, cdr in enumerate(CDR_LEVELS):
        group = sessions[(sessions["CDR"] == cdr) & (sessions["qc_flag"] == 0)]
        # Pick the session whose nWBV is closest to its CDR group's median.
        example = group.iloc[(group["nWBV"] - group["nWBV"].median()).abs().argmin()]
        for row, view in enumerate(VIEWS):
            ax = axes[row, col]
            image = display(slices[f"{example.mri_id}/{view}"], view)
            # Columns are left-right (1.25 mm) in the coronal and axial views.
            ax.imshow(image, cmap="gray", vmin=0, vmax=np.percentile(image, 99.5),
                      aspect=1 if view == "sagittal" else 0.8)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(f"CDR {cdr:g}\n{example.mri_id}, age {example.Age:.0f}, "
                             f"nWBV {example.nWBV:.3f}", fontsize=9)
            if col == 0:
                label = "Axial (orbit level)" if view == "axial" else view.capitalize()
                ax.set_ylabel(label, fontsize=10, color=TEXT_SECONDARY)
    return fig


def page_quality(data: "ReportData", title: str, section: str) -> plt.Figure:
    """Raw intensity scale, SNR, and repeat-scan agreement across sessions."""
    scans, sessions = data.scans, data.sessions
    flagged = int(sessions["qc_flag"].sum())
    fig = new_page(title, f"Raw intensity scales differ between scans; {flagged} of "
                   f"{len(sessions)} sessions are flagged as quality outliers", section)
    axes = fig.subplots(1, 3, gridspec_kw={"left": 0.06, "right": 0.97, "top": 0.8,
                                           "bottom": 0.26, "wspace": 0.3})

    ax = axes[0]
    sns.histplot(scans["foreground_p50"], bins=30, color=STABLE_COLOR, ax=ax)
    cv = scans["foreground_p50"].std() / scans["foreground_p50"].mean()
    ax.set_title("Median foreground intensity (per scan)", loc="left", fontsize=10)
    ax.set_xlabel(f"Raw intensity (coefficient of variation = {cv:.0%})")
    ax.set_ylabel("Scans")

    for ax, column, subtitle, xlabel in [
        (axes[1], "snr", "Signal-to-noise ratio", "Mean foreground / background SD"),
        (axes[2], "repeat_scan_r", "Repeat-scan agreement (motion)",
         "Pearson r, first scan vs. repeats"),
    ]:
        sns.histplot(sessions[column].dropna(), bins=30, color=STABLE_COLOR, ax=ax)
        q1, q3 = sessions[column].quantile([0.25, 0.75])
        ax.axvline(q1 - QC_IQR_MULTIPLIER * (q3 - q1), color=DECLINED_COLOR, ls="--",
                   lw=1.2)
        n_low = int(sessions[f"low_{column}"].sum())
        ax.text(0.03, 0.95, f"Dashed line: flag cutoff\n{n_low} sessions flagged",
                transform=ax.transAxes, va="top", color=DECLINED_COLOR, fontsize=8.5)
        ax.set_title(subtitle, loc="left", fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Sessions")

    add_notes(fig, [
        "Intensities are in arbitrary scanner units, so each scan must be normalized "
        "(e.g., z-score within the brain) before it is fed to a CNN.",
        f"Flag = more than {QC_IQR_MULTIPLIER:g} IQR below the first quartile. Flagged "
        "sessions stay in the dataset (mri_qc_flag) for a sensitivity analysis.",
    ], top=0.15)
    return fig


def page_vs_target(data: "ReportData", title: str, section: str) -> plt.Figure:
    """Check the CSF proxy against nWBV and CDR, then compare it with the target."""
    pairs, sessions = data.pairs, data.sessions
    auc = data.auc.set_index("feature")["auc"]
    r = sessions[["csf_fraction_proxy", "nWBV"]].corr().iloc[0, 1]
    fig = new_page(title, "The CSF proxy tracks atrophy and CDR, but, like nWBV, it "
                   "separates future decliners only weakly", section)
    axes = fig.subplots(1, 4, gridspec_kw={"left": 0.06, "right": 0.98, "top": 0.8,
                                           "bottom": 0.27, "wspace": 0.38})

    ax = axes[0]
    ax.scatter(sessions["nWBV"], sessions["csf_fraction_proxy"], s=10, alpha=0.6,
               color=STABLE_COLOR, edgecolor="none")
    ax.set_title(f"CSF proxy vs. nWBV (r = {r:.2f})", loc="left", fontsize=10)
    ax.set_xlabel("nWBV (from CSV), all sessions")
    ax.set_ylabel("CSF fraction proxy")

    ax = axes[1]
    sns.boxplot(sessions, x="CDR", y="csf_fraction_proxy", color=STABLE_COLOR,
                width=0.55, fliersize=2, ax=ax)
    ax.set_title("CSF proxy by CDR at scan", loc="left", fontsize=10)
    ax.set_xticks(range(len(CDR_LEVELS)), [f"{c:g}" for c in CDR_LEVELS])
    ax.set_ylabel("")

    for ax, feature, subtitle in [
        (axes[2], "mri_csf_proxy", "CSF proxy at visit t"),
        (axes[3], "mri_snr", "Scan SNR at visit t"),
    ]:
        sns.boxplot(pairs, x=config.TARGET_COLUMN, y=feature, hue=config.TARGET_COLUMN,
                    palette=CLASS_PALETTE, legend=False, width=0.5, fliersize=2, ax=ax)
        ax.set_xticks([0, 1], ["Stable", "Declined"])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(f"{subtitle}\ntrain AUC = {auc[feature]:.2f}", loc="left",
                     fontsize=10)

    add_notes(fig, [
        f"For comparison, nWBV from the CSV has a train AUC of {auc['nwbv']:.2f}. "
        "AUC below 0.5 means lower values go with decline.",
        "SNR should not separate the classes; an AUC near 0.5 confirms that scan "
        "quality is not a hidden proxy for the outcome.",
        "The CNN is expected to extract richer structure (e.g., ventricle and "
        "hippocampus shape) than this single global measure.",
    ], top=0.16)
    add_class_legend(fig)
    return fig

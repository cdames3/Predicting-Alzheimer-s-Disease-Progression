"""Generate the EDA report (PDF) for the ML-ready OASIS-2 dataset.

The report follows the order a reader needs:
    1. Data sources and lineage (clinical CSV and raw MRI scans)
    2. Target (stable vs. declined at the next visit)
    3. Features (clinical metadata and the MRI scan measure)
    4. MRI scans (inventory, examples, quality, relation to the target)
    5. Summary (key findings and implications for modeling)
The dataset card is appended at the end. Every chart page is also exported to
``reports/figures`` as ``<page>_<name>.png`` for the slides.

Feature-target association scores are computed on the training split only,
so the test set is not used to guide feature or model choices.

Usage:
    python -m src.data.make_mri_dataset   # MRI tables (needs the raw archives)
    python -m src.data.make_dataset       # ML-ready visit pairs
    python -m src.visualization.eda_report
"""

import json
import textwrap
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_auc_score

from src import config
from src.visualization import mri_eda
from src.visualization.report_layout import (
    CLASS_LABELS, CLASS_PALETTE, DECLINED_COLOR, DIVERGING_CMAP, PAGE_SIZE,
    SEQUENTIAL_CMAP, STABLE_COLOR, TEXT_PRIMARY, TEXT_SECONDARY, MarkdownPages,
    Report, add_class_legend, add_notes, fmt, new_page, set_style, style_table,
)

CONTINUOUS_FEATURES = [
    "age", "educ", "mmse", "etiv", "nwbv", "asf", "days_to_next",
    "days_since_baseline", "mmse_rate_prev", "nwbv_rate_prev", "mri_csf_proxy",
    "mri_snr",
]

FEATURE_LABELS = {
    "age": "Age (years)",
    "educ": "Education (years)",
    "ses": "SES (1 = highest)",
    "mmse": "MMSE (0-30)",
    "cdr": "CDR at visit t",
    "etiv": "eTIV (cm^3)",
    "nwbv": "nWBV",
    "asf": "ASF",
    "days_to_next": "Days to next visit",
    "days_since_baseline": "Days since baseline",
    "mmse_rate_prev": "MMSE change / year (prev.)",
    "nwbv_rate_prev": "nWBV change / year (prev.)",
    "mri_csf_proxy": "CSF proxy (MRI scan)",
    "mri_snr": "Scan SNR (quality only)",
}

GROUP_SOURCES = {
    "Demographics": "Clinical CSV",
    "Clinical / cognitive": "Clinical CSV",
    "MRI measures (CSV)": "Clinical CSV",
    "Visit timing": "Derived from CSV",
    "Change since previous visit": "Derived from CSV",
    "MRI scan measure": "Raw MRI scans",
}

GLOSSARY = [
    ("Visit pair", "One row: a patient's visit t (features) and next visit t+1 (target)."),
    ("CDR", "Clinical Dementia Rating: 0 none, 0.5 very mild, 1 mild, 2 moderate."),
    ("Declined", "Target = 1 when CDR at visit t+1 is higher than at visit t."),
    ("MMSE", "Mini-Mental State Examination, 0 to 30; lower = worse cognition."),
    ("nWBV", "Normalized whole-brain volume (share of intracranial volume), from the CSV."),
    ("eTIV / ASF", "Estimated total intracranial volume / atlas scaling factor (~1/eTIV)."),
    ("SES", "Socioeconomic status, 1 (highest) to 5 (lowest)."),
    ("CSF proxy", "Share of dark, fluid-like voxels in the center of the raw scan."),
    ("SNR", "Signal-to-noise ratio of a scan (image quality)."),
    ("Train AUC", "ROC-AUC of one feature alone on training patients; 0.5 = no signal."),
]


@dataclass
class ReportData:
    """Everything the pages need, loaded once."""

    pairs: pd.DataFrame     # ML-ready visit pairs (with MRI columns)
    meta: dict              # visit-pair metadata JSON
    mri_meta: dict          # MRI metadata JSON
    scans: pd.DataFrame     # one row per MRI scan
    sessions: pd.DataFrame  # one row per MRI session, with CDR, nWBV, and age
    auc: pd.DataFrame       # univariate train AUC per feature

    @property
    def train(self) -> pd.DataFrame:
        return self.pairs[self.pairs["split"] == "train"]


def univariate_auc(train: pd.DataFrame) -> pd.DataFrame:
    """Score how well each feature alone separates the classes (train only).

    AUC below 0.5 means lower values go with decline; ``strength`` folds both
    directions onto 0.5 to 1 so features can be ranked.
    """
    rows = []
    for feature in config.FEATURE_COLUMNS + ["mri_snr"]:
        auc = roc_auc_score(train[config.TARGET_COLUMN], train[feature])
        rows.append({
            "feature": feature,
            "auc": auc,
            "strength": max(auc, 1 - auc),
            "direction": "higher -> decline" if auc >= 0.5 else "lower -> decline",
        })
    return pd.DataFrame(rows).sort_values("strength", ascending=False)


def load_data() -> ReportData:
    """Load the ML-ready dataset, the MRI tables, and their metadata."""
    required = [config.ML_READY_CSV, config.MRI_SESSIONS_CSV, config.MRI_SCANS_CSV]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing}. Run `python -m src.data.make_mri_dataset` and "
            "`python -m src.data.make_dataset` first.")
    pairs = pd.read_csv(config.ML_READY_CSV)
    raw = pd.read_csv(config.RAW_CSV)[["MRI ID", "CDR", "nWBV", "Age"]]
    sessions = pd.read_csv(config.MRI_SESSIONS_CSV).merge(
        raw.rename(columns={"MRI ID": "mri_id"}), on="mri_id", how="left")
    train = pairs[pairs["split"] == "train"]
    return ReportData(
        pairs=pairs,
        meta=json.loads(config.ML_READY_METADATA.read_text()),
        mri_meta=json.loads(config.MRI_METADATA.read_text()),
        scans=pd.read_csv(config.MRI_SCANS_CSV),
        sessions=sessions,
        auc=univariate_auc(train),
    )


# ---------------------------------------------------------------------------
# Front matter
# ---------------------------------------------------------------------------

def page_cover(data: ReportData) -> plt.Figure:
    """Cover page with headline numbers and split sizes."""
    df, meta, mri = data.pairs, data.meta, data.mri_meta
    fig = plt.figure(figsize=PAGE_SIZE)
    fig.text(0.05, 0.80, "Exploratory Data Analysis", fontsize=30,
             weight="bold", color=TEXT_PRIMARY)
    fig.text(0.05, 0.74, "OASIS-2 visit pairs: clinical metadata + structural MRI  |  "
             "Predicting next-visit cognitive decline", fontsize=14,
             color=TEXT_SECONDARY)
    fig.text(0.05, 0.70, "DSCI 4350, Group 5: Christina Dames, Cinthia Ochoa Torre, "
             "Rayna Alexandre-Petrie  |  Checkpoint 2", fontsize=11,
             color=TEXT_SECONDARY)

    tiles = [
        (f"{df['subject_id'].nunique()}", "patients"),
        (f"{len(df)}", "visit pairs"),
        (f"{mri['scans']:,}", f"MRI scans ({mri['sessions']} sessions)"),
        (f"{len(config.FEATURE_COLUMNS)}", "model features"),
        (f"{df[config.TARGET_COLUMN].mean():.1%}", "declined (target = 1)"),
    ]
    for i, (value, label) in enumerate(tiles):
        x = 0.05 + i * 0.185
        fig.text(x, 0.52, value, fontsize=32, weight="bold", color=TEXT_PRIMARY)
        fig.text(x, 0.475, label, fontsize=10.5, color=TEXT_SECONDARY)

    ax = fig.add_axes([0.05, 0.14, 0.6, 0.25])
    ax.axis("off")
    cells = [[name.capitalize(), s["patients"], s["visit_pairs"], s["declined"],
              f"{s['decline_rate']:.1%}"]
             for name, s in sorted(meta["splits"].items(), reverse=True)]
    table = ax.table(cellText=cells, colLabels=["Split", "Patients", "Visit pairs",
                                                "Declined", "Decline rate"],
                     loc="upper left", cellLoc="center")
    style_table(table, font_size=11, row_height=1.8)
    fig.text(0.05, 0.08, f"Dataset file: {meta['output_file']}  |  SHA-256: "
             f"{meta['output_sha256'][:16]}...  |  random_state = {meta['random_state']}",
             fontsize=9, color=TEXT_SECONDARY)
    return fig


def page_contents(sections: list, first_page: int, appendix_page: int) -> plt.Figure:
    """Table of contents, reading guide, and glossary."""
    fig = new_page("Contents and Reading Guide",
                   "Sections follow the data from its sources to the modeling "
                   "decisions it supports")
    y = 0.82
    page = first_page
    for section, pages in sections:
        fig.text(0.05, y, section, fontsize=11.5, weight="bold", color=TEXT_PRIMARY)
        y -= 0.036
        for title, _, _ in pages:
            fig.text(0.075, y, title, fontsize=10, color=TEXT_PRIMARY)
            fig.text(0.46, y, str(page), fontsize=10, color=TEXT_SECONDARY, ha="right")
            y -= 0.03
            page += 1
        y -= 0.012
    fig.text(0.05, y, "Appendix  Dataset Card", fontsize=11.5, weight="bold",
             color=TEXT_PRIMARY)
    fig.text(0.46, y, str(appendix_page), fontsize=10, color=TEXT_SECONDARY, ha="right")

    x = 0.53
    fig.text(x, 0.82, "How to read this report", fontsize=11.5, weight="bold",
             color=TEXT_PRIMARY)
    guide = [
        "Blue = stable (0), orange = declined (1) on every chart.",
        "Train AUC scores use training patients only; the test set stays unseen.",
        "Section 3 covers all model inputs; Section 4 covers the MRI scans.",
        "Figures are saved as reports/figures/<page>_<name>.png.",
    ]
    yy = 0.785
    for line in guide:
        fig.text(x, yy, "-  " + line, fontsize=9.5, color=TEXT_PRIMARY)
        yy -= 0.03
    yy -= 0.02
    fig.text(x, yy, "Glossary", fontsize=11.5, weight="bold", color=TEXT_PRIMARY)
    yy -= 0.037
    for term, meaning in GLOSSARY:
        lines = textwrap.wrap(meaning, 62)
        fig.text(x, yy, term, fontsize=9.5, weight="bold", color=TEXT_PRIMARY, va="top")
        fig.text(x + 0.1, yy, "\n".join(lines), fontsize=9.5, color=TEXT_PRIMARY,
                 va="top", linespacing=1.35)
        yy -= 0.026 * len(lines) + 0.012
    return fig


# ---------------------------------------------------------------------------
# 1. Data sources and lineage
# ---------------------------------------------------------------------------

def draw_box(ax, x: float, y: float, w: float, h: float, head: str, body: str,
             external: bool) -> None:
    """Draw one lineage stage: blue for external sources, orange for our steps."""
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.4,rounding_size=1.5",
                                facecolor="#eef4fb" if external else "#fdf0ea",
                                edgecolor=STABLE_COLOR if external else DECLINED_COLOR,
                                linewidth=1.5))
    ax.text(x + w / 2, y + h - 6, head, ha="center", fontsize=9.5, weight="bold",
            color=TEXT_PRIMARY)
    ax.text(x + w / 2, y + h / 2 - 5, body, ha="center", va="center", fontsize=7.8,
            color=TEXT_SECONDARY, linespacing=1.5)


def arrow(ax, start: tuple, end: tuple) -> None:
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="->", color=TEXT_SECONDARY, lw=1.3))


def page_lineage(data: ReportData, title: str, section: str) -> plt.Figure:
    """Two-branch flow: clinical CSV and raw MRI scans joined into one dataset."""
    meta, mri = data.meta, data.mri_meta
    fig = new_page(title, "Two OASIS-2 sources, one ML-ready file: clinical visits and "
                   "raw MRI scans are joined on mri_id (src/data/)", section)
    ax = fig.add_axes([0.03, 0.33, 0.94, 0.53])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    w, h, step = 12.6, 30, 14.3
    col = [0.6 + i * step for i in range(7)]
    top, bottom, middle = 64, 6, 35
    lineage = meta["lineage"]
    size_gb = sum(a["size_bytes"] for a in mri["archives"]) / 1e9
    link = mri["linkage"]

    draw_box(ax, col[0], middle, w, h, "OASIS-2 study",
             "WashU Knight ADRC\nlongitudinal MRI +\nclinical visits", True)
    draw_box(ax, col[1], top, w, h, "Clinical CSV",
             f"Kaggle copy\n{lineage[0]['rows']} visits x 15\nused unmodified", True)
    draw_box(ax, col[2], top, w, h, "Clean + engineer",
             "MMSE from previous\nvisit; change\nfeatures", False)
    draw_box(ax, col[3], top, w, h, "Pair visits",
             f"visit t -> t+1\n{lineage[2]['rows']} pairs\ntarget: CDR rises", False)
    draw_box(ax, col[1], bottom, w, h, "Raw MRI archives",
             f"{len(mri['archives'])} x .tar.gz\n{size_gb:.1f} GB\n{mri['scans']:,} T1 scans",
             True)
    draw_box(ax, col[2], bottom, w, h, "Stream + QC",
             "decoded in memory\nintensity, SNR,\nCSF proxy", False)
    draw_box(ax, col[3], bottom, w, h, "Per session",
             f"{mri['sessions']} sessions\nrepeat-scan r\nQC flags", False)
    draw_box(ax, col[4], middle, w, h, "Join on mri_id",
             f"{link['pairs_with_scans_at_both_visits']} / {link['visit_pairs']} pairs\n"
             "have scans at\nboth visits", False)
    draw_box(ax, col[5], middle, w, h, "Split + impute",
             "patient-level split\n5 grouped CV folds\n"
             f"SES mode (train) = {meta['imputation']['ses_mode']:.0f}", False)
    draw_box(ax, col[6], middle, w, h, "ML-ready CSV",
             f"{lineage[-1]['rows']} rows\n{len(meta['feature_columns'])} features\n"
             "+ metadata JSON", False)

    arrow(ax, (col[0] + w + 0.4, middle + h * 0.7), (col[1] - 0.6, top + h / 2))
    arrow(ax, (col[0] + w + 0.4, middle + h * 0.3), (col[1] - 0.6, bottom + h / 2))
    for row in (top, bottom):
        for i in (1, 2):
            arrow(ax, (col[i] + w + 0.4, row + h / 2), (col[i + 1] - 0.6, row + h / 2))
    arrow(ax, (col[3] + w + 0.4, top + h / 2), (col[4] - 0.6, middle + h * 0.7))
    arrow(ax, (col[3] + w + 0.4, bottom + h / 2), (col[4] - 0.6, middle + h * 0.3))
    for i in (4, 5):
        arrow(ax, (col[i] + w + 0.4, middle + h / 2), (col[i + 1] - 0.6, middle + h / 2))

    fig.text(0.05, 0.3, "Blue = external sources    Orange = this project's "
             "transformations", fontsize=9, color=TEXT_SECONDARY)
    add_notes(fig, [
        "Top branch (src/data/make_dataset.py): MMSE filled from the patient's previous "
        "visit, backward-looking change features, each visit paired with the next.",
        "Bottom branch (src/data/make_mri_dataset.py): the 19 GB archives are streamed, "
        "not extracted (about 45 GB on disk); each scan gets quality and CSF measures.",
        "Excluded raw columns: Group (assigned from the full visit history, so it leaks "
        "the target) and Hand (constant).",
        "Imputation uses training patients only; every step is deterministic "
        "(random_state = 42) and SHA-256 hashes are stored in the metadata JSON files.",
    ], top=0.265, size=9.5, width=165)
    return fig


def page_overview(data: ReportData, title: str, section: str) -> plt.Figure:
    """Feature groups, source files, and the MRI linkage check."""
    df, meta, mri = data.pairs, data.meta, data.mri_meta
    fig = new_page(title, "One row = one visit pair; the features describe the "
                   "patient and the scan at visit t", section)

    fig.text(0.05, 0.855, "Columns by group", fontsize=11, weight="bold",
             color=TEXT_PRIMARY)
    rows = [[group, GROUP_SOURCES[group], len(columns), ", ".join(columns)]
            for group, columns in config.FEATURE_GROUPS.items()]
    rows += [
        ["MRI quality (audit)", "Raw MRI scans", len(config.MRI_QC_COLUMNS),
         ", ".join(config.MRI_QC_COLUMNS) + "  (not model inputs)"],
        ["Target", "CDR at t and t+1", 1,
         "declined  (cdr_next is kept for analysis only, never as an input)"],
        ["IDs and splits", "-", 5, "subject_id, mri_id, visit, split, cv_fold"],
    ]
    ax = fig.add_axes([0.05, 0.47, 0.9, 0.37])
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=["Group", "Source", "n", "Columns"],
                     loc="upper center", cellLoc="left",
                     colWidths=[0.18, 0.14, 0.04, 0.64])
    style_table(table, font_size=8.5, row_height=1.55)

    fig.text(0.05, 0.415, "Source and output files", fontsize=11, weight="bold",
             color=TEXT_PRIMARY)
    files = [[config.RAW_CSV.name, "Kaggle (OASIS-2)",
              f"{meta['lineage'][0]['rows']} visits",
              f"{config.RAW_CSV.stat().st_size / 1e3:.0f} KB", meta["raw_sha256"][:12]]]
    files += [[a["file"], "oasis-brains.org", f"{a['scans']} scans",
               f"{a['size_bytes'] / 1e9:.1f} GB", a["sha256"][:12]]
              for a in mri["archives"]]
    files.append([meta["output_file"], "This project", f"{len(df)} pairs",
                  f"{config.ML_READY_CSV.stat().st_size / 1e3:.0f} KB",
                  meta["output_sha256"][:12]])
    ax = fig.add_axes([0.05, 0.08, 0.56, 0.32])
    ax.axis("off")
    table = ax.table(cellText=files, colLabels=["File", "Origin", "Records", "Size",
                                                "SHA-256"],
                     loc="upper left", cellLoc="center",
                     colWidths=[0.33, 0.19, 0.16, 0.11, 0.21])
    style_table(table, font_size=8.5, row_height=1.6)

    link = mri["linkage"]
    splits = meta["splits"]
    by_split = df.groupby("split")["subject_id"].apply(set)
    overlap = len(by_split["train"] & by_split["test"])
    fig.text(0.65, 0.415, "Linkage and split checks", fontsize=11, weight="bold",
             color=TEXT_PRIMARY)
    checks = [
        ["Visits with an MRI session", f"{link['visits_with_scan']} / {link['clinical_visits']}"],
        ["Pairs with scans at t and t+1",
         f"{link['pairs_with_scans_at_both_visits']} / {link['visit_pairs']}"],
        ["Scans / sessions", f"{mri['scans']:,} / {mri['sessions']}"],
        ["Sessions flagged (QC)", f"{len(mri['qc_flagged_sessions'])}"],
        ["Train / test patients",
         f"{splits['train']['patients']} / {splits['test']['patients']}"],
        ["Patients in both splits", str(overlap)],
    ]
    ax = fig.add_axes([0.65, 0.08, 0.3, 0.32])
    ax.axis("off")
    table = ax.table(cellText=checks, colLabels=["Check", "Result"], loc="upper left",
                     cellLoc="center", colWidths=[0.68, 0.32])
    style_table(table, font_size=8.5, row_height=1.6)
    return fig


# ---------------------------------------------------------------------------
# 2. Target
# ---------------------------------------------------------------------------

def page_target(data: ReportData, title: str, section: str) -> plt.Figure:
    """Target balance by split and the CDR transition matrix."""
    df, splits = data.pairs, data.meta["splits"]
    fig = new_page(title, "Decline is rare (about 14%) and the patient-level split "
                   "keeps that rate similar in train and test", section)
    ax1 = fig.add_axes([0.07, 0.12, 0.38, 0.68])
    counts = (df.groupby(["split", config.TARGET_COLUMN]).size()
              .unstack().reindex(["train", "test"]))
    x = np.arange(len(counts))
    width = 0.38
    for offset, cls in [(-width / 2, 0), (width / 2, 1)]:
        bars = ax1.bar(x + offset, counts[cls], width - 0.02, color=CLASS_PALETTE[cls],
                       label=CLASS_LABELS[cls])
        ax1.bar_label(bars, padding=3, color=TEXT_PRIMARY, fontsize=10)
    ax1.set_xticks(x, [f"Train ({splits['train']['patients']} patients)",
                       f"Test ({splits['test']['patients']} patients)"])
    ax1.set_ylabel("Visit pairs")
    ax1.set_title("Class balance by split", loc="left")
    ax1.legend(frameon=False)

    ax2 = fig.add_axes([0.56, 0.12, 0.38, 0.68])
    transitions = pd.crosstab(df["cdr"], df["cdr_next"])
    sns.heatmap(transitions, annot=True, fmt="d", cmap=SEQUENTIAL_CMAP, cbar=False,
                linewidths=2, linecolor="white", ax=ax2, annot_kws={"fontsize": 12})
    ax2.set_xlabel("CDR at next visit (t+1)")
    ax2.set_ylabel("CDR at visit t")
    ax2.set_title("CDR transitions (above diagonal = declined)", loc="left")
    return fig


# ---------------------------------------------------------------------------
# 3. Features
# ---------------------------------------------------------------------------

def page_feature_stats(data: ReportData, title: str, section: str) -> plt.Figure:
    """Per-feature statistics by class, grouped by source, with train AUC."""
    df = data.pairs
    fig = new_page(title, f"Mean (SD) by target class on all {len(df)} pairs; "
                   "univariate ROC-AUC on the training split only", section)
    ax = fig.add_axes([0.05, 0.06, 0.9, 0.81])
    ax.axis("off")
    auc_by_feature = data.auc.set_index("feature")
    rows = []
    for group, columns in config.FEATURE_GROUPS.items():
        for i, feature in enumerate(columns):
            col = df[feature]
            stable = df.loc[df[config.TARGET_COLUMN] == 0, feature]
            declined = df.loc[df[config.TARGET_COLUMN] == 1, feature]
            rows.append([
                group if i == 0 else "",
                feature,
                f"{fmt(col.min())} to {fmt(col.max())}",
                f"{fmt(stable.mean())} ({fmt(stable.std())})",
                f"{fmt(declined.mean())} ({fmt(declined.std())})",
                f"{auc_by_feature.loc[feature, 'auc']:.2f}",
            ])
    table = ax.table(cellText=rows, colLabels=["Group", "Feature", "Range",
                                               "Stable mean (SD)", "Declined mean (SD)",
                                               "Train AUC"],
                     loc="upper center", cellLoc="center",
                     colWidths=[0.2, 0.17, 0.17, 0.17, 0.17, 0.1])
    style_table(table, font_size=8.3, row_height=1.2)
    for (row, column), cell in table.get_celld().items():
        if column == 0 and row > 0:
            cell.set_text_props(weight="bold", color=TEXT_PRIMARY, ha="left")
    return fig


def page_distributions(data: ReportData, title: str, section: str) -> plt.Figure:
    """Histogram of each continuous feature, split by target class."""
    df = data.pairs
    fig = new_page(title, "Densities are normalized within each class so the small "
                   "declined group stays visible", section)
    axes = fig.subplots(3, 4, gridspec_kw={"left": 0.05, "right": 0.98, "top": 0.84,
                                           "bottom": 0.1, "hspace": 0.55,
                                           "wspace": 0.25})
    for ax, feature in zip(axes.flat, CONTINUOUS_FEATURES):
        sns.histplot(df, x=feature, hue=config.TARGET_COLUMN, palette=CLASS_PALETTE,
                     stat="density", common_norm=False, element="step", fill=True,
                     alpha=0.25, bins=15, ax=ax, legend=False)
        ax.set_title(FEATURE_LABELS[feature], fontsize=9, loc="left")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelsize=7)
    add_class_legend(fig)
    return fig


def page_feature_vs_target(data: ReportData, title: str, section: str) -> plt.Figure:
    """Boxplots of key features and decline rates for categorical features."""
    df = data.pairs
    fig = new_page(title, "Cognitive scores and baseline CDR separate the classes more "
                   "than brain volume, the MRI scan measure, or demographics", section)
    top = fig.subplots(1, 4, gridspec_kw={"left": 0.06, "right": 0.98, "top": 0.82,
                                          "bottom": 0.54, "wspace": 0.35})
    for ax, feature in zip(top, ["mmse", "nwbv", "mri_csf_proxy", "age"]):
        sns.boxplot(df, x=config.TARGET_COLUMN, y=feature, hue=config.TARGET_COLUMN,
                    palette=CLASS_PALETTE, legend=False, width=0.5, fliersize=2, ax=ax)
        ax.set_xticks([0, 1], ["Stable", "Declined"])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(FEATURE_LABELS[feature], fontsize=10, loc="left")

    bottom = fig.subplots(1, 4, gridspec_kw={"left": 0.06, "right": 0.98, "top": 0.42,
                                             "bottom": 0.12, "wspace": 0.35})
    groups = [
        ("cdr", "Decline rate by CDR at visit t", None),
        ("sex_male", "Decline rate by sex", {0: "Female", 1: "Male"}),
        ("ses", "Decline rate by SES", None),
        ("has_prev_visit", "Decline rate by visit history",
         {0: "First visit", 1: "Has previous"}),
    ]
    overall = df[config.TARGET_COLUMN].mean()
    for ax, (feature, subtitle, names) in zip(bottom, groups):
        rates = df.groupby(feature)[config.TARGET_COLUMN].agg(["mean", "size"])
        labels = [names[k] if names else f"{k:g}" for k in rates.index]
        bars = ax.bar(labels, rates["mean"], color=DECLINED_COLOR, width=0.6)
        ax.bar_label(bars, labels=[f"n={n}" for n in rates["size"]], padding=2,
                     fontsize=7, color=TEXT_SECONDARY)
        ax.axhline(overall, color=TEXT_SECONDARY, lw=1, ls="--")
        ax.set_ylim(0, max(0.35, rates["mean"].max() * 1.25))
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_title(subtitle, fontsize=10, loc="left")
        ax.tick_params(labelsize=8)
    fig.text(0.06, 0.055, f"Dashed line = overall decline rate ({overall:.1%}). "
             "n = visit pairs in each group.", fontsize=8.5, color=TEXT_SECONDARY)
    add_class_legend(fig)
    return fig


def page_correlations(data: ReportData, title: str, section: str) -> plt.Figure:
    """Pearson correlation heatmap of the features and the target."""
    columns = config.FEATURE_COLUMNS + [config.TARGET_COLUMN]
    corr = data.pairs[columns].corr()
    csf_nwbv = corr.loc["mri_csf_proxy", "nwbv"]
    fig = new_page(title, "Pearson r, including the target. eTIV and ASF are almost "
                   f"perfectly redundant (r = {corr.loc['etiv', 'asf']:.2f}); the CSF "
                   f"proxy tracks nWBV (r = {csf_nwbv:.2f})", section)
    ax = fig.add_axes([0.2, 0.13, 0.62, 0.74])
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, cmap=DIVERGING_CMAP, vmin=-1, vmax=1, center=0,
                annot=True, fmt=".2f", annot_kws={"fontsize": 5.5}, linewidths=0.5,
                linecolor="white", cbar_kws={"shrink": 0.6, "label": "Pearson r"}, ax=ax)
    ax.tick_params(labelsize=7.5)
    return fig


def page_auc_ranking(data: ReportData, title: str, section: str) -> plt.Figure:
    """Horizontal bar chart ranking features by univariate separation."""
    fig = new_page(title, "Univariate ROC-AUC per feature on the training split "
                   "(0.5 = no signal). [scan] = computed from the raw MRI", section)
    ax = fig.add_axes([0.25, 0.12, 0.65, 0.74])
    ranked = data.auc[data.auc["feature"].isin(config.FEATURE_COLUMNS)]
    ranked = ranked.sort_values("strength")
    labels = [f"{f} [scan]" if f in config.MRI_SCAN_FEATURES else f
              for f in ranked["feature"]]
    colors = [DECLINED_COLOR if d.startswith("lower") else STABLE_COLOR
              for d in ranked["direction"]]
    bars = ax.barh(labels, ranked["strength"] - 0.5, left=0.5, color=colors, height=0.65)
    ax.bar_label(bars, labels=[f"{s:.2f}" for s in ranked["strength"]], padding=3,
                 fontsize=8, color=TEXT_PRIMARY)
    ax.set_xlim(0.5, max(0.8, ranked["strength"].max() + 0.05))
    ax.set_xlabel("max(AUC, 1 - AUC)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (DECLINED_COLOR, STABLE_COLOR)]
    ax.legend(handles, ["Lower value -> more decline", "Higher value -> more decline"],
              loc="lower right", frameon=False)
    return fig


# ---------------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------------

def bullet_page(title: str, section: str, blocks: list[tuple[str, list[str]]],
                subtitle: str = "") -> plt.Figure:
    """Text page of headed bullet lists."""
    fig = new_page(title, subtitle, section)
    y = 0.84
    for heading, points in blocks:
        fig.text(0.06, y, heading, fontsize=12.5, weight="bold", color=TEXT_PRIMARY,
                 va="top")
        y -= 0.045
        for point in points:
            lines = textwrap.wrap(point, 125)
            fig.text(0.07, y, "-", fontsize=11, color=DECLINED_COLOR, weight="bold",
                     va="top")
            fig.text(0.085, y, "\n".join(lines), fontsize=11, color=TEXT_PRIMARY,
                     va="top", linespacing=1.4)
            y -= 0.029 * len(lines) + 0.017
        y -= 0.02
    return fig


def page_findings(data: ReportData, title: str, section: str) -> plt.Figure:
    """Key findings, computed from the data."""
    df, mri, auc = data.pairs, data.mri_meta, data.auc.set_index("feature")
    target = df[config.TARGET_COLUMN]
    rate_by_cdr = df.groupby("cdr")[config.TARGET_COLUMN].mean()
    rate_by_history = df.groupby("has_prev_visit")[config.TARGET_COLUMN].mean()
    mmse = df.groupby(config.TARGET_COLUMN)["mmse"].mean()
    top = data.auc[data.auc["feature"].isin(config.FEATURE_COLUMNS)].head(3)
    corr = df[["mri_csf_proxy", "nwbv", "etiv", "asf"]].corr()
    link = mri["linkage"]
    test_pos = data.meta["splits"]["test"]["declined"]
    blocks = [
        ("Data", [
            f"{df['subject_id'].nunique()} patients, {len(df)} visit pairs, and "
            f"{mri['scans']:,} MRI scans; {link['pairs_with_scans_at_both_visits']} of "
            f"{link['visit_pairs']} pairs have scans at both visits.",
            f"Imbalanced target: {target.sum()} of {len(df)} pairs declined "
            f"({target.mean():.1%}); the test set has only {test_pos} positives.",
            "All scans share one format (256 x 256 x 128, 1 x 1 x 1.25 mm). Raw "
            "intensities vary between scans, and "
            f"{len(mri['qc_flagged_sessions'])} sessions are flagged for motion.",
        ]),
        ("Signal", [
            "Strongest single features (train AUC): " + ", ".join(
                f"{r.feature} ({r.strength:.2f})" for r in top.itertuples()) + ".",
            f"Cognition is the clearest clinical signal: mean MMSE {mmse[1]:.1f} for "
            f"decliners vs. {mmse[0]:.1f} for stable pairs; decline rate "
            f"{rate_by_cdr.get(0.5, 0):.0%} at CDR 0.5 vs. {rate_by_cdr.get(0.0, 0):.0%} "
            "at CDR 0.",
            f"Brain-volume measures are weak alone: nWBV train AUC "
            f"{auc.loc['nwbv', 'strength']:.2f}; the CSF proxy from the scans tracks nWBV "
            f"(r = {corr.loc['mri_csf_proxy', 'nwbv']:.2f}) but reaches only "
            f"{auc.loc['mri_csf_proxy', 'strength']:.2f}.",
            f"Most decline happens early in follow-up: {rate_by_history[0]:.0%} of "
            f"first-visit pairs vs. {rate_by_history[1]:.0%} of later pairs.",
            f"eTIV and ASF are redundant (r = {corr.loc['etiv', 'asf']:.2f}). Scan SNR "
            f"does not separate the classes (AUC {auc.loc['mri_snr', 'auc']:.2f}), so "
            "image quality does not leak the outcome.",
        ]),
    ]
    return bullet_page(title, section, blocks,
                       "What the clinical data and the MRI scans show, with every "
                       "number computed from the ML-ready dataset")


def page_implications(data: ReportData, title: str, section: str) -> plt.Figure:
    """What the findings mean for the three planned models."""
    blocks = [
        ("Evaluation", [
            "Report recall, F1-score, and ROC-AUC (accuracy alone rewards predicting "
            "'stable'); use class weights or resampling.",
            "Select models with the 5 patient-grouped CV folds; touch the test set once "
            "and report its metrics with their uncertainty.",
        ]),
        ("The three models", [
            f"Metadata-only: the {len(config.METADATA_FEATURES)} CSV features; keep "
            "either eTIV or ASF for linear models.",
            "MRI-only: CNN on preprocessed scans: average the 2-4 repeats, skull-strip, "
            "register to a template, z-score intensities, and resample (e.g., 128^3 or "
            "2D slices).",
            "Multimodal: combine the CNN embedding with the metadata features, using the "
            "same patient-level splits so results are comparable.",
        ]),
        ("Risks to manage", [
            f"Small sample ({len(data.pairs)} pairs): prefer transfer learning or a "
            "small CNN with augmentation and early stopping.",
            "Re-run the final models without the QC-flagged sessions as a sensitivity "
            "check.",
        ]),
    ]
    return bullet_page(title, section, blocks,
                       "How the findings shape the metadata-only, MRI-only, and "
                       "multimodal models in Sprint 3")


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

SECTIONS = [
    ("1  Data Sources and Lineage", [
        ("Data Lineage", "lineage", page_lineage),
        ("Dataset at a Glance", "dataset_overview", page_overview),
    ]),
    ("2  Target", [
        ("Target: Stable vs. Declined", "target_balance", page_target),
    ]),
    ("3  Features", [
        ("Feature Statistics", "feature_statistics", page_feature_stats),
        ("Feature Distributions by Class", "distributions", page_distributions),
        ("Features vs. Target", "feature_vs_target", page_feature_vs_target),
        ("Feature Correlations", "correlations", page_correlations),
        ("Which Features Separate the Classes?", "univariate_auc", page_auc_ranking),
    ]),
    ("4  MRI Scans", [
        ("MRI Inventory and Acquisition", "mri_inventory", mri_eda.page_inventory),
        ("Example Scans by Dementia Rating", "mri_examples", mri_eda.page_examples),
        ("MRI Intensity and Quality", "mri_quality", mri_eda.page_quality),
        ("MRI-Derived Measures vs. Target", "mri_vs_target", mri_eda.page_vs_target),
    ]),
    ("5  Summary", [
        ("Key Findings", None, page_findings),
        ("Implications for Modeling", None, page_implications),
    ]),
]


def main() -> None:
    """Build every report page, export the figures, and write the PDF."""
    data = load_data()
    set_style()

    # Start from an empty figures folder so it only holds this version's figures.
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for old in config.FIGURES_DIR.glob("*.png"):
        old.unlink()

    report = Report(config.EDA_REPORT_PDF, config.FIGURES_DIR)
    first_page = 3  # after the cover and the contents
    n_pages = sum(len(pages) for _, pages in SECTIONS)
    report.add(page_cover(data))
    report.add(page_contents(SECTIONS, first_page, first_page + n_pages))
    for section, pages in SECTIONS:
        for title, name, build in pages:
            report.add(build(data, title, section), name)
    MarkdownPages(report).render(config.DATASET_CARD.read_text())
    report.close("EDA Report and Dataset Card: OASIS-2 Visit Pairs and MRI",
                 "DSCI 4350 Group 5")

    print(f"Wrote {config.EDA_REPORT_PDF} ({report.page} pages)")
    print(data.auc.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

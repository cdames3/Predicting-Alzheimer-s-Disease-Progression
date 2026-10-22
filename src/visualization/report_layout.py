"""Shared page layout, colors, and helpers for the EDA report.

Every page gets the same header (section label, title, subtitle) and footer
(report name and page number), so the tabular and MRI pages read as one
document.
"""

import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap

PAGE_SIZE = (11, 8.5)  # US Letter, landscape
FOOTER = "DSCI 4350, Group 5  |  EDA Report: OASIS-2 Visit Pairs + MRI  |  Checkpoint 2"

# Colorblind-validated categorical slots (stable, declined) and text inks.
STABLE_COLOR = "#2a78d6"
DECLINED_COLOR = "#eb6834"
CLASS_PALETTE = {0: STABLE_COLOR, 1: DECLINED_COLOR}
CLASS_LABELS = {0: "Stable (0)", 1: "Declined (1)"}
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"
HEADER_FILL = "#f4f3f0"

# Diverging map for correlations: blue (negative), gray midpoint, orange (positive).
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "blue_gray_orange", [STABLE_COLOR, "#f0efec", DECLINED_COLOR]
)
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "blues", ["#f0f5fc", STABLE_COLOR, "#123c6e"]
)


def set_style() -> None:
    """Apply a light, recessive-grid style to every figure."""
    sns.set_theme(style="whitegrid", rc={
        "axes.edgecolor": GRID_COLOR,
        "grid.color": GRID_COLOR,
        "axes.labelcolor": TEXT_SECONDARY,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.titleweight": "bold",
        "font.size": 9,
    })


def new_page(title: str, subtitle: str = "", section: str = "") -> plt.Figure:
    """Create a landscape page with a section label, title, and subtitle."""
    fig = plt.figure(figsize=PAGE_SIZE)
    if section:
        fig.text(0.05, 0.962, section.upper(), fontsize=8.5, weight="bold",
                 color=DECLINED_COLOR)
    fig.text(0.05, 0.925, title, fontsize=18, weight="bold", color=TEXT_PRIMARY)
    if subtitle:
        fig.text(0.05, 0.893, subtitle, fontsize=10.5, color=TEXT_SECONDARY)
    return fig


def style_table(table, font_size: float = 9, row_height: float = 1.4) -> None:
    """Give a matplotlib table quiet borders and a bold header row."""
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, row_height)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor(GRID_COLOR)
        if row == 0:
            cell.set_text_props(weight="bold", color=TEXT_PRIMARY)
            cell.set_facecolor(HEADER_FILL)


def add_class_legend(fig: plt.Figure) -> None:
    """Place a shared stable/declined legend at the top right of a page."""
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_PALETTE[c]) for c in (0, 1)]
    fig.legend(handles, [CLASS_LABELS[0], CLASS_LABELS[1]], loc="upper right",
               bbox_to_anchor=(0.98, 0.95), frameon=False, ncol=2)


def add_notes(fig: plt.Figure, notes: list[str], top: float, size: float = 10,
              width: int = 140) -> None:
    """Write short bullet notes downward from ``top`` (figure coordinates)."""
    y = top
    for note in notes:
        lines = textwrap.wrap(note, width)
        fig.text(0.06, y, "-  " + "\n   ".join(lines), fontsize=size,
                 color=TEXT_PRIMARY, va="top", linespacing=1.4)
        y -= 0.026 * len(lines) + 0.014


def fmt(value: float) -> str:
    """Format a statistic without scientific notation for large values."""
    if abs(value) >= 100:
        return f"{value:,.0f}"
    return f"{value:.3g}"


class Report:
    """PDF writer that numbers pages, adds the footer, and exports PNG figures.

    Chart pages are also saved as ``<page>_<name>.png`` so the slide deck and
    the report always refer to the same figure numbers.
    """

    def __init__(self, path: Path, figures_dir: Path, footer: str = FOOTER):
        self.pdf = PdfPages(path)
        self.figures_dir = figures_dir
        self.footer = footer
        self.page = 0

    def add(self, fig: plt.Figure, name: str | None = None) -> None:
        self.page += 1
        fig.text(0.05, 0.02, self.footer, fontsize=7.5, color=TEXT_SECONDARY)
        fig.text(0.95, 0.02, str(self.page), fontsize=8, ha="right",
                 color=TEXT_SECONDARY)
        if name:
            fig.savefig(self.figures_dir / f"{self.page:02d}_{name}.png", dpi=200)
        self.pdf.savefig(fig)
        plt.close(fig)

    def close(self, title: str, author: str) -> None:
        info = self.pdf.infodict()
        info["Title"] = title
        info["Author"] = author
        self.pdf.close()


class MarkdownPages:
    """Render a simple Markdown file (headings, lists, tables) onto report pages."""

    LEFT, TOP, BOTTOM = 0.05, 0.93, 0.06
    TABLE_WIDTH_CHARS = 150  # monospace characters that fit across the page

    def __init__(self, report: Report):
        self.report = report
        self.fig = None
        self.y = self.TOP

    def _ensure_space(self, height: float) -> None:
        if self.fig is None or self.y - height < self.BOTTOM:
            if self.fig is not None:
                self.report.add(self.fig)
            self.fig = plt.figure(figsize=PAGE_SIZE)
            self.y = self.TOP

    def _write(self, text: str, size: float, weight: str = "normal",
               family: str = "sans-serif", color: str = TEXT_PRIMARY,
               indent: float = 0.0, space_after: float = 0.004) -> None:
        line_height = size * 1.45 / 72 / PAGE_SIZE[1]
        self._ensure_space(line_height)
        self.fig.text(self.LEFT + indent, self.y, text, fontsize=size, weight=weight,
                      family=family, color=color, va="top")
        self.y -= line_height + space_after

    @staticmethod
    def _clean(text: str) -> str:
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        text = re.sub(r"\*(.+?)\*", r"\1", text)
        return text.replace("`", "")

    def _table(self, rows: list[str]) -> None:
        cells = [[self._clean(c.strip()) for c in row.strip().strip("|").split("|")]
                 for row in rows if not re.match(r"^\|[\s\-|]+\|$", row.strip())]
        widths = [max(len(r[i]) for r in cells) for i in range(len(cells[0]))]
        # Wrap the last column so long descriptions stay on the page.
        fixed = sum(w + 2 for w in widths[:-1])
        widths[-1] = min(widths[-1], self.TABLE_WIDTH_CHARS - fixed)
        for n, row in enumerate(cells):
            last_lines = textwrap.wrap(row[-1], widths[-1]) or [""]
            for k, last in enumerate(last_lines):
                lead = row[:-1] if k == 0 else [""] * (len(row) - 1)
                line = "  ".join(c.ljust(w) for c, w in zip(lead + [last], widths))
                self._write(line, 7.5, weight="bold" if n == 0 else "normal",
                            family="monospace", space_after=0.001)
        self.y -= 0.01

    def render(self, markdown: str) -> None:
        lines = markdown.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("|"):
                block = []
                while i < len(lines) and lines[i].startswith("|"):
                    block.append(lines[i])
                    i += 1
                self._table(block)
                continue
            text = self._clean(line)
            if line.startswith("# "):
                self._write(text[2:], 20, weight="bold", space_after=0.015)
            elif line.startswith("## "):
                self.y -= 0.01
                self._write(text[3:], 14, weight="bold", space_after=0.008)
            elif line.startswith("### "):
                self._write(text[4:], 11, weight="bold", space_after=0.006)
            elif re.match(r"^(- |\d+\. )", line):
                marker, body = re.match(r"^(- |\d+\. )(.*)", text).groups()
                for j, wrapped in enumerate(textwrap.wrap(body, 150)):
                    prefix = marker if j == 0 else " " * len(marker)
                    self._write(prefix + wrapped, 9, indent=0.01, space_after=0.001)
                self.y -= 0.004
            elif text.strip():
                for wrapped in textwrap.wrap(text, 155):
                    self._write(wrapped, 9, space_after=0.001)
                self.y -= 0.006
            i += 1
        if self.fig is not None:
            self.report.add(self.fig)

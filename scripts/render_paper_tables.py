"""Render Tables 1-3 of rl_framework_paper_v5.md as publication table figures.

Run: python scripts/render_paper_tables.py

Tolerance curves, error attribution, and the sensitivity/ablation results are figures:
see render_tolerance_sweep_figure.py and render_result_figures.py.

Outputs are written to plots/ as 300-dpi PNG files.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PLOTS = Path(__file__).resolve().parents[1] / "plots"
NAVY = "#173B57"
BLUE = "#EAF2F8"
STRIPE = "#F6F8FA"
GREEN = "#DDF2E9"
GREEN_TEXT = "#126B50"
GRID = "#C8D2DA"
TEXT = "#17212B"


def draw_paired_value(ax, x_right, y, value, best_indexes, fontsize):
    """Draw an "a / b" cell right-aligned, with only the parts in best_indexes highlighted."""
    renderer = ax.figure.canvas.get_renderer()
    parts = value.split(" / ")
    pieces = []
    for index, part in enumerate(parts):
        if index > 0:
            pieces.append((" / ", False))
        pieces.append((part, index in best_indexes))
    for piece, is_best in reversed(pieces):
        text = ax.text(
            x_right,
            y,
            piece,
            ha="right",
            va="center",
            color=GREEN_TEXT if is_best else TEXT,
            fontsize=fontsize,
            fontweight="bold" if is_best else "normal",
        )
        extent = text.get_window_extent(renderer).transformed(ax.transAxes.inverted())
        x_right -= extent.width


def render_table(
    name,
    headers,
    rows,
    *,
    widths,
    best=(),
    best_parts=None,
    strong_rows=(),
    notes=None,
    figsize=(10, 3.2),
    fontsize=10,
):
    """Draw a clean journal-style table with optional highlighted best cells."""
    nrows, ncols = len(rows), len(headers)
    best_parts = best_parts or {}
    total_width = sum(widths)
    x_edges = [0]
    for width in widths:
        x_edges.append(x_edges[-1] + width / total_width)

    note_space = 0.34 if notes else 0
    row_height = 1 / (nrows + 1 + note_space)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Header
    y = 1 - row_height
    for col, header in enumerate(headers):
        x0, width = x_edges[col], x_edges[col + 1] - x_edges[col]
        ax.add_patch(Rectangle((x0, y), width, row_height, facecolor=NAVY, edgecolor="white", linewidth=1.2))
        ax.text(
            x0 + (0.018 if col == 0 else width - 0.018),
            y + row_height / 2,
            header,
            ha="left" if col == 0 else "right",
            va="center",
            color="white",
            fontsize=fontsize,
            fontweight="bold",
            linespacing=1.15,
        )

    # Body
    for row_index, row in enumerate(rows):
        y = 1 - (row_index + 2) * row_height
        is_strong = row_index in strong_rows
        base = BLUE if is_strong else ("white" if row_index % 2 == 0 else STRIPE)
        for col, value in enumerate(row):
            x0, width = x_edges[col], x_edges[col + 1] - x_edges[col]
            is_best = (row_index, col) in best
            fill = GREEN if is_best else (BLUE if col == 0 and not is_strong else base)
            ax.add_patch(Rectangle((x0, y), width, row_height, facecolor=fill, edgecolor=GRID, linewidth=0.65))
            if (row_index, col) in best_parts:
                draw_paired_value(
                    ax, x0 + width - 0.018, y + row_height / 2, value, best_parts[(row_index, col)], fontsize
                )
                continue
            ax.text(
                x0 + (0.018 if col == 0 else width - 0.018),
                y + row_height / 2,
                value,
                ha="left" if col == 0 else "right",
                va="center",
                color=GREEN_TEXT if is_best else TEXT,
                fontsize=fontsize,
                fontweight="bold" if is_best or is_strong else "normal",
                linespacing=1.12,
            )

    # Strong top and bottom rules give the table a publication-style frame.
    table_bottom = 1 - (nrows + 1) * row_height
    ax.plot([0, 1], [1, 1], color=NAVY, linewidth=1.8, solid_capstyle="butt")
    ax.plot([0, 1], [table_bottom, table_bottom], color=NAVY, linewidth=1.8, solid_capstyle="butt")
    if notes:
        ax.text(0, table_bottom - row_height * 0.18, notes, ha="left", va="top", fontsize=fontsize - 2, color="#4A5965")

    PLOTS.mkdir(exist_ok=True)
    fig.savefig(PLOTS / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main():
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans", "figure.facecolor": "white", "savefig.facecolor": "white",
    })

    render_table(
        "table_01_test_cohort",
        ["Source", "Recordings", "Hours", "Retained AF\nepisodes", "Mixed-rhythm\nrecordings"],
        [
            ["IRIDIA-AF", "167", "6,687", "387", "167"],
            ["SHDB-AF", "98", "2,331", "498", "77"],
            ["LTAFDB", "84", "1,954", "2,074", "56"],
            ["Total", "349", "10,972", "2,959", "300"],
        ],
        widths=[2.15, 1.45, 1.3, 1.8, 1.8], strong_rows={3}, figsize=(8.8, 2.8),
    )

    render_table(
        "table_02_method_comparison",
        ["Mean over 1–30 s", "Duration F1", "Onset F1", "Offset F1", "Boundary F1", "Ue"],
        [
            ["ECGFounder", "", "", "", "", ""],
            ["Threshold-only, θ = 0.5", "0.959", "0.156", "0.161", "0.159", "0.097"],
            ["Tuned post-processing, θ = 0.5 / g = 15 s", "0.959", "0.190", "0.196", "0.193", "0.116"],
            ["Supervised MLP", "0.961", "0.174", "0.170", "0.172", "0.105"],
            ["PPO controller", "0.961", "0.217", "0.235", "0.226", "0.141"],
            ["ResNet1D", "", "", "", "", ""],
            ["Tuned post-processing, θ = 0.5 / g = 15 s", "0.903", "0.091", "0.097", "0.094", "0.057"],
            ["Supervised MLP", "0.902", "0.085", "0.081", "0.083", "0.050"],
            ["PPO controller", "0.906", "0.108", "0.122", "0.115", "0.070"],
        ],
        widths=[3.7, 1.45, 1.15, 1.15, 1.55, 0.8],
        best={(4, c) for c in range(1, 6)} | {(8, c) for c in range(1, 6)},
        strong_rows={0, 5},
        figsize=(10.85, 4.4),
    )

    render_table(
        "table_03_episode_level",
        [
            "ECGFounder\n(2,959 ref. episodes)",
            "Pred.\nepisodes",
            "Any-ov.\nSe",
            "Any-ov.\n+P",
            "Any-ov.\nF1",
            "1–1\nSe",
            "1–1\n+P",
            "1–1\nF1",
            "Exact F1\nδ = 5 s",
            "Exact F1\nmean 1–30 s",
            "Mean count\nerror",
        ],
        [
            ["Tuned post-processing", "6,705", "0.880", "0.564", "0.687", "0.303", "0.134", "0.186", "0.129", "0.122", "18.1"],
            ["Supervised MLP", "6,917", "0.859", "0.666", "0.751", "0.312", "0.133", "0.187", "0.092", "0.095", "18.2"],
            ["PPO controller", "5,401", "0.893", "0.548", "0.679", "0.337", "0.184", "0.238", "0.154", "0.146", "14.4"],
        ],
        widths=[2.6, 1.3, 1.1, 1.1, 1.1, 0.9, 0.9, 0.9, 1.2, 1.5, 1.5],
        best={(2, 2), (1, 3), (1, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10)},
        figsize=(13.0, 2.55),
        fontsize=8,
    )

    print(f"wrote table figures as PNG + PDF to {PLOTS}")


if __name__ == "__main__":
    main()

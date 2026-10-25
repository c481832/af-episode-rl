"""Render Figure 3: the boundary-matching cases (a)-(d) of paper section 4.2.

The episode times are the ones quoted in the text; TP / FP / FN are computed with
the same greedy smallest-error matching the text describes, so the markers cannot
drift from the intervals.

Run: uv run python scripts/render_boundary_matching_figure.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

OUT_PATH = Path(__file__).resolve().parents[1] / "plots" / "figure_03_boundary_matching_examples.png"

TOLERANCE = 5
REF_Y = 1.0
PRED_Y = 0.0
BAR_HEIGHT = 0.3
MARKER_OFFSET = 0.42
LABEL_OFFSET = 0.66
REF_COLOR = "#2979d6"
PRED_COLOR = "#eb6834"
WINDOW_COLOR = "#e7e6e4"
MUTED = "#4d4d4d"

# (title, note, reference episodes, predicted episodes), times in seconds.
CASES = [
    (
        "(a) One-to-one, offset outside tolerance",
        "onset TP (+3 s) · offset FN + FP (+20 s)",
        [(100, 400)],
        [(103, 420)],
    ),
    (
        "(b) Fragmented: one reference, two predicted",
        "ends TP via different fragments · internal cut = 2 FP",
        [(100, 400)],
        [(102, 230), (260, 399)],
    ),
    (
        "(c) Merged: two references, one prediction",
        "outer boundaries TP · the gap's offset and onset = 2 FN",
        [(100, 200), (250, 400)],
        [(101, 401)],
    ),
    (
        "(d) Missed episode and false episode",
        "missed reference = 2 FN · prediction with no overlap = 2 FP",
        [(100, 200)],
        [(350, 410)],
    ),
]


def match_boundaries(ref_times, pred_times):
    """Greedy one-to-one matching within TOLERANCE, smallest error first."""
    candidates = sorted(
        (abs(pred - ref), ref, pred)
        for ref in ref_times
        for pred in pred_times
        if abs(pred - ref) <= TOLERANCE
    )
    matched_ref = set()
    matched_pred = set()
    for _, ref, pred in candidates:
        if ref in matched_ref or pred in matched_pred:
            continue
        matched_ref.add(ref)
        matched_pred.add(pred)
    return matched_ref, matched_pred


def draw_markers(ax, times, matched, row_y, direction, miss_label):
    for time in times:
        is_match = time in matched
        label = "TP" if is_match else miss_label
        marker = "o" if is_match or miss_label == "FN" else "X"
        face = "white" if label == "FN" else "black"
        ax.plot(
            time,
            row_y + direction * MARKER_OFFSET,
            marker=marker,
            markersize=8,
            markerfacecolor=face,
            markeredgecolor="black",
            markeredgewidth=1.6,
            linestyle="none",
            zorder=4,
        )
        ax.text(
            time,
            row_y + direction * LABEL_OFFSET,
            f"{label} {time}",
            ha="center",
            va="bottom" if direction > 0 else "top",
            fontsize=9,
        )


def draw_case(ax, title, note, reference, predicted):
    for episodes, row_y, color in ((reference, REF_Y, REF_COLOR), (predicted, PRED_Y, PRED_COLOR)):
        ax.broken_barh(
            [(start, end - start) for start, end in episodes],
            (row_y - BAR_HEIGHT / 2, BAR_HEIGHT),
            facecolor=color,
            edgecolor="white",
            linewidth=2,
            zorder=3,
        )

    for side in (0, 1):
        ref_times = [episode[side] for episode in reference]
        pred_times = [episode[side] for episode in predicted]
        matched_ref, matched_pred = match_boundaries(ref_times, pred_times)
        draw_markers(ax, ref_times, matched_ref, REF_Y, +1, "FN")
        draw_markers(ax, pred_times, matched_pred, PRED_Y, -1, "FP")
        for time in ref_times:
            ax.axvspan(time - TOLERANCE, time + TOLERANCE, ymin=0.08, ymax=0.9, color=WINDOW_COLOR, zorder=1)

    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.set_title(note, loc="right", fontsize=8.5, color=MUTED)
    ax.set_xlim(60, 450)
    ax.set_ylim(-0.95, 1.85)
    ax.set_yticks([REF_Y, PRED_Y], ["Reference", "Predicted"], fontsize=9.5, color=MUTED)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=9, labelcolor=MUTED)
    for name in ("left", "right", "top"):
        ax.spines[name].set_visible(False)
    ax.spines["bottom"].set_color("#cccccc")


def main():
    fig, axes = plt.subplots(len(CASES), 1, figsize=(9.5, 8.2), sharex=True)
    for ax, case in zip(axes, CASES):
        draw_case(ax, *case)
    axes[-1].set_xlabel("Time (s)", fontsize=10, color=MUTED)

    marker_style = {"color": "black", "linestyle": "none", "markersize": 8, "markeredgewidth": 1.6}
    handles = [
        Line2D([], [], marker="o", markerfacecolor="black", label=f"TP (matched within ±{TOLERANCE} s)", **marker_style),
        Line2D([], [], marker="o", markerfacecolor="white", label="FN (reference boundary unmatched)", **marker_style),
        Line2D([], [], marker="X", markerfacecolor="black", label="FP (predicted boundary unmatched)", **marker_style),
        Patch(facecolor=WINDOW_COLOR, label=f"±{TOLERANCE} s window around a reference boundary"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT_PATH, dpi=200)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

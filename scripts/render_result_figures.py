"""Render Figures 5 and 6 of rl_framework_paper_v5.md from the reported counts and scores.

Figure 5: boundary errors at delta = 5 s attributed to overlap groups (former Table 5).
Figure 6: hyperparameter sensitivity and feature ablation (former Tables 9 and 10).

The numbers are the ones the former tables carried (scripts/render_paper_tables.py history);
they come from experiment_rl_foundation.ipynb and the sweeps in runs/foundation/controller.

Run: python scripts/render_result_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

PLOTS = Path(__file__).resolve().parents[1] / "plots"

TEXT = "#17212B"
MUTED = "#5b6670"
GRID = "#dde3e8"
REFERENCE = "#8a949c"

# Same method colors and markers as Figure 4.
METHODS = {
    "Tuned post-processing": {"color": "#eb6834", "marker": "s"},
    "Supervised MLP": {"color": "#1baf7a", "marker": "^"},
    "PPO controller": {"color": "#2a78d6", "marker": "o"},
}

# Overlap-group attribution at delta = 5 s (paper section 4.6), onsets and offsets pooled.
FP_GROUPS = ["One-to-one", "Fragmented", "Merged", "Merged and fragmented", "False episode", "Total"]
FN_GROUPS = ["One-to-one", "Fragmented", "Merged", "Merged and fragmented", "Missed episode", "Total"]
FP_COUNTS = {
    "Tuned post-processing": [340, 4327, 80, 780, 5824, 11351],
    "Supervised MLP": [572, 5924, 66, 836, 4594, 11992],
    "PPO controller": [430, 2779, 82, 562, 4871, 8724],
}
FN_COUNTS = {
    "Tuned post-processing": [323, 171, 579, 2066, 709, 3848],
    "Supervised MLP": [555, 234, 479, 1967, 829, 4064],
    "PPO controller": [412, 153, 731, 1893, 630, 3819],
}
TP_COUNTS = {"Tuned post-processing": 1873, "Supervised MLP": 1657, "PPO controller": 1902}

TUNED_POST_PROCESSING_BOUNDARY_F1 = 0.193
REPORTED_BOUNDARY_F1 = 0.226

# One-at-a-time sweep around the reported configuration: (setting, (low label, F1), (high label, F1)).
SWEEP = [
    ("Policy width (hidden units)", ("16", 0.231), ("64", 0.232)),
    ("Value width", ("[4]", 0.227), ("[256, 256]", 0.238)),
    ("Rollout steps", ("4,096", 0.239), ("32,768", 0.201)),
    ("Minibatch", ("256", 0.242), ("4,096", 0.240)),
    ("Value coefficient", ("1.0", 0.227), ("3.0", 0.229)),
    ("Clip range", ("0.05", 0.211), ("0.2", 0.240)),
    ("Target KL", ("0.005", 0.226), ("0.03", 0.226)),
    ("Learning rate", ("10⁻⁴", 0.224), ("10⁻³", 0.229)),
    ("Entropy coefficient", ("0", 0.224), ("0.05", 0.225)),
    ("Epochs per rollout", ("1", 0.230), ("10", 0.230)),
    ("Passes", ("3", 0.217), ("10", 0.233)),
]

ABLATION = [
    ("Full controller", 0.226),
    ("Without heart rate", 0.226),
    ("Without forward context", 0.114),
    ("Without preceding action", 0.074),
]


def style_axis(ax, grid_axis):
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def grouped_hbars(ax, groups, counts_by_method, title):
    n_methods = len(METHODS)
    bar_height = 0.8 / n_methods
    y = np.arange(len(groups))
    xmax = max(max(values) for values in counts_by_method.values())
    for i, (method, style) in enumerate(METHODS.items()):
        offsets = y - 0.4 + bar_height * (i + 0.5)
        values = counts_by_method[method]
        ax.barh(offsets, values, height=bar_height * 0.9, color=style["color"], label=method)
        for offset, value in zip(offsets, values):
            ax.text(value + xmax * 0.01, offset, f"{value:,}", va="center", fontsize=7, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels(groups, fontsize=8.5, color=TEXT)
    ax.invert_yaxis()
    ax.set_xlim(0, xmax * 1.15)
    # Separate the total from the per-group rows.
    ax.axhline(len(groups) - 1.5, color=MUTED, linewidth=0.8)
    ax.set_title(title, fontsize=10, color=TEXT, loc="left")
    ax.set_xlabel("Boundaries at δ = 5 s", fontsize=9, color=TEXT)
    style_axis(ax, "x")


def render_attribution():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1, 1, 0.42]})
    grouped_hbars(axes[0], FP_GROUPS, FP_COUNTS, "(a) False-positive boundaries by group")
    grouped_hbars(axes[1], FN_GROUPS, FN_COUNTS, "(b) False-negative boundaries by group")
    grouped_hbars(axes[2], ["Matched"], {m: [v] for m, v in TP_COUNTS.items()}, "(c) True positives")
    # The single TP row has no total to separate; widen its y-range so its bars match (a) and (b).
    axes[2].lines[-1].remove()
    axes[2].set_ylim(2.5, -0.9)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "figure_05_boundary_error_attribution")


def reference_lines(ax):
    ax.axvline(TUNED_POST_PROCESSING_BOUNDARY_F1, color=METHODS["Tuned post-processing"]["color"],
               linestyle="--", linewidth=1.2)
    ax.axvline(REPORTED_BOUNDARY_F1, color=REFERENCE, linestyle="-", linewidth=1.0)


def render_sensitivity():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0), gridspec_kw={"width_ratios": [1.35, 1]})
    color = METHODS["PPO controller"]["color"]

    ax = axes[0]
    reference_lines(ax)
    y = np.arange(len(SWEEP))
    for row, (_, (low_label, low), (high_label, high)) in zip(y, SWEEP):
        ax.plot([low, high], [row, row], color=GRID, linewidth=2, zorder=1)
        ax.scatter([low], [row], s=42, facecolor="white", edgecolor=color, linewidth=1.6, zorder=3)
        ax.scatter([high], [row], s=42, color=color, zorder=3)
        # Put each variant's label on the outer side of its point so the pair never overlaps.
        left_is_low = low <= high
        for value, label, is_low in ((low, low_label, True), (high, high_label, False)):
            on_left = is_low == left_is_low
            ax.annotate(label, (value, row), xytext=(-7 if on_left else 7, 0), textcoords="offset points",
                        ha="right" if on_left else "left", va="center", fontsize=7, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([name for name, _, _ in SWEEP], fontsize=8.5, color=TEXT)
    ax.invert_yaxis()
    ax.set_xlim(0.185, 0.252)
    ax.set_title("(a) One-at-a-time hyperparameter sweep", fontsize=10, color=TEXT, loc="left")
    ax.set_xlabel("Mean Boundary F1 over 1–30 s", fontsize=9, color=TEXT)
    style_axis(ax, "x")

    ax = axes[1]
    reference_lines(ax)
    y = np.arange(len(ABLATION))
    for row, (name, value) in zip(y, ABLATION):
        ax.barh(row, value, height=0.55, color=color)
        ax.text(value + 0.004, row, f"{value:.3f}", va="center", fontsize=7.5, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([name for name, _ in ABLATION], fontsize=8.5, color=TEXT)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.27)
    ax.set_title("(b) Feature ablation", fontsize=10, color=TEXT, loc="left")
    ax.set_xlabel("Mean Boundary F1 over 1–30 s", fontsize=9, color=TEXT)
    style_axis(ax, "x")

    handles = [
        Line2D([], [], color=METHODS["Tuned post-processing"]["color"], linestyle="--",
               label=f"Tuned post-processing ({TUNED_POST_PROCESSING_BOUNDARY_F1:.3f})"),
        Line2D([], [], color=REFERENCE, label=f"Reported configuration ({REPORTED_BOUNDARY_F1:.3f})"),
        Line2D([], [], color=color, marker="o", markerfacecolor="white", linestyle="", label="First value in label"),
        Line2D([], [], color=color, marker="o", linestyle="", label="Second value"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "figure_06_sensitivity_and_ablation")


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS / f"{name}.{ext}", dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"wrote {PLOTS / name}.png/.pdf")


def main():
    matplotlib.rcParams.update({"font.family": "DejaVu Sans"})
    render_attribution()
    render_sensitivity()


if __name__ == "__main__":
    main()

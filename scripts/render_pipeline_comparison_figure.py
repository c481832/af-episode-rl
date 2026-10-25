"""Render Figure 1 of rl_framework_paper_v5.md: tuned post-processing versus the controller.

Two stacked panels with the same layout, one per decision stage: the frozen detector turns the
ECG into p(t), the decision stage forms episodes from it, and the episodes it forms on one real
test window (SHDB-AF record 130) are drawn against the reference.
(a) Post-processing is feed-forward with fixed constants and filters short runs afterwards.
(b) The controller is a closed loop on its preceding action, with the 30-second rule inside it.

The window was found by scanning isolated reference episodes in the test set for ones where
tuned post-processing fragments the episode or adds a false one while the controller forms it
one-to-one (6 of 254; the two agree on 201). It illustrates the mechanism behind section 7.5, not a
typical record. Labels come from the reported controller's cached rollouts, as in Figure 5.

Run: python scripts/render_pipeline_comparison_figure.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, FancyArrowPatch, FancyBboxPatch, Rectangle

from ecg_af.benchmark_eval import event_intervals
from ecg_af.data.predictions import load_fold_results
from ecg_af.util import GLOBAL_CONFIG
from render_tolerance_sweep_figure import EXPERIMENT, controller_labels, post_processing_labels

PLOTS = Path(__file__).resolve().parents[1] / "plots"
OUT_NAME = "figure_01_pipeline_comparison"

RECORD = ("shdb_af", "130")
WINDOW = (44150, 45000)  # seconds

# Same palette as the detailed controller figure (figure_02_rl_pipeline).
INK = "#111111"
MUTED = "#6b7178"
PROB = "#29a3dc"
POST = "#f07f1a"
CONTROLLER = "#1f9e4a"
MASK_FILL = "#ece6f6"
MASK_EDGE = "#6a4c9c"
BOX_LW = 1.8


def window_data():
    GLOBAL_CONFIG.load_config(EXPERIMENT)
    GLOBAL_CONFIG.name = EXPERIMENT
    probs_all, times_all = load_fold_results(
        GLOBAL_CONFIG["rl"]["base_models"]["segment"], int(GLOBAL_CONFIG["data"]["k_folds"]["n_splits"])
    )
    true_by, post_by = post_processing_labels([RECORD])
    _, controller_by = controller_labels([RECORD])

    lo, hi = WINDOW
    times = times_all[RECORD].astype(int)
    if not np.array_equal(times, np.arange(len(times))):
        raise ValueError("Expected one probability per second starting at 0")
    data = {
        "prob": probs_all[RECORD][lo:hi],
        "reference": true_by[RECORD][lo:hi],
        "post": post_by[RECORD][lo:hi],
        "controller": controller_by[RECORD][lo:hi],
    }

    # The figure's annotations describe this exact structure; fail if the data no longer has it.
    runs = {name: event_intervals(data[name], 0.5)["AF"] for name in ("reference", "post", "controller")}
    shape = {name: len(r) for name, r in runs.items()}
    if shape != {"reference": 1, "post": 3, "controller": 1}:
        raise ValueError(f"Window no longer shows the illustrated pattern: {shape}")
    return data, runs


def box(ax, x, y, w, h, text, edge=INK, fill="white", size=10.5, weight="bold", style="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.9",
                                linewidth=BOX_LW, edgecolor=edge, facecolor=fill, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            fontweight=weight, fontstyle=style, color=INK, zorder=4, linespacing=1.25)


def arrow(ax, start, end, color=INK, lw=1.6, style="-|>", connection="arc3", dashed=False):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=13, color=color,
                                 linewidth=lw, connectionstyle=connection, zorder=2,
                                 linestyle=(0, (4, 3)) if dashed else "-", shrinkA=0, shrinkB=0))


def lock(ax, x, y, s=1.0):
    ax.add_patch(Rectangle((x, y), 1.9 * s, 1.5 * s, facecolor=INK, edgecolor=INK, zorder=5))
    ax.add_patch(Arc((x + 0.95 * s, y + 1.5 * s), 1.3 * s, 1.6 * s, theta1=0, theta2=180,
                     linewidth=2.0, color=INK, zorder=5))


def mini_axes(ax, x, y, w, h):
    """An inset in data coordinates of the canvas axes."""
    fig = ax.figure
    disp = ax.transData.transform([(x, y), (x + w, y + h)])
    (x0, y0), (x1, y1) = fig.transFigure.inverted().transform(disp)
    inset = fig.add_axes((x0, y0, x1 - x0, y1 - y0))
    for side in ("top", "right"):
        inset.spines[side].set_visible(False)
    inset.tick_params(labelsize=7.5, colors=MUTED, length=2)
    return inset


def bar_row(panel, labels, y, color):
    on = labels > 0.5
    panel.fill_between(np.arange(len(labels) + 1), y - 0.22, y + 0.22,
                       where=np.append(on, on[-1]), step="post", color=color, linewidth=0)


def detector(ax, base):
    """ECG -> frozen detector, identical in both panels."""
    mid = base + 9.0
    ax.text(2.5, mid, "ECG", va="center", fontsize=10.5, fontweight="bold", color=INK)
    arrow(ax, (7.5, mid), (10.0, mid))
    box(ax, 10.0, mid - 3.5, 18.0, 7.0, "      Frozen\n      AF detector")
    lock(ax, 11.5, mid - 1.0, 0.9)
    arrow(ax, (28.0, mid), (32.5, mid))
    return mid


def detector_and_probability(ax, data, base, show_threshold):
    """ECG -> frozen detector -> p(t) over the whole window."""
    mid = detector(ax, base)

    prob = mini_axes(ax, 35.0, mid - 5.0, 17.0, 10.0)
    prob.plot(data["prob"], color=PROB, linewidth=0.8)
    prob.set_ylim(0, 1)
    prob.set_yticks([0, 1])
    prob.set_xticks([])
    prob.set_xlim(0, len(data["prob"]))
    ax.text(43.5, mid + 6.0, "$p(t)$", ha="center", fontsize=10, color=INK)
    if show_threshold:
        prob.axhline(0.5, color=POST, linewidth=1.0, linestyle="--")
        prob.text(len(data["prob"]) * 0.995, 0.55, "$\\theta$", ha="right", va="bottom", fontsize=9,
                  color=POST, fontweight="bold")
    arrow(ax, (53.0, mid), (57.0, mid))
    return mid


# Bucket edges of the multiscale forward context (paper section 3.3), in seconds after t.
FORWARD_BUCKETS = [(1, 1), (2, 3), (4, 7), (8, 15), (16, 31)]
ZOOM_BEFORE, ZOOM_AFTER = 45, 34


def decision_inputs(ax, data, runs, base):
    """What the controller sees at one decision t: its past actions and the next 31 s of p(t).

    t is the dip where post-processing splits the episode in panel (a): p(t) is below theta, but
    the controller is already in AF and the forward context shows p rising again.
    """
    mid = detector(ax, base)
    _, first_fragment, main_fragment = runs["post"]
    t = int(round((first_fragment[1] + main_fragment[0]) / 2))
    lo, hi = t - ZOOM_BEFORE, t + ZOOM_AFTER + 1
    offsets = np.arange(lo, hi) - t

    prob = mini_axes(ax, 35.0, mid - 2.0, 17.0, 7.5)
    shades = ["#d6eef9", "#b9e1f4"]
    for i, (a, b) in enumerate(FORWARD_BUCKETS):
        prob.axvspan(a - 0.5, b + 0.5, color=shades[i % 2], linewidth=0)
    prob.plot(offsets, data["prob"][lo:hi], color=PROB, linewidth=1.2)
    prob.axvline(0, color=INK, linewidth=1.0, linestyle="--")
    prob.axhline(0.5, color=MUTED, linewidth=0.7, linestyle=":")
    prob.set_ylim(0, 1)
    prob.set_yticks([0, 0.5, 1])
    prob.set_xticks([])
    prob.set_xlim(offsets[0], offsets[-1])
    prob.text(0, 1.04, "$t$", ha="center", va="bottom", fontsize=9.5, color=INK)
    prob.text(16, 0.06, "next 31 s", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.text(43.5, mid + 8.6, "$p$ around one decision $t$", ha="center", fontsize=9.5, color=INK)

    # The controller's own past actions, up to t - 1; the action at t is the one being decided.
    acts = mini_axes(ax, 35.0, mid - 5.6, 17.0, 2.4)
    acts.axis("off")
    acts.set_xlim(offsets[0], offsets[-1])
    acts.set_ylim(0, 1)
    past = data["controller"][lo:t] > 0.5
    for x, on in zip(offsets[offsets < 0], past):
        acts.add_patch(Rectangle((x - 0.5, 0.15), 1.0, 0.7, color=CONTROLLER if on else "#d5d9dd", linewidth=0))
    acts.text(1.5, 0.5, "?", ha="left", va="center", fontsize=10, fontweight="bold", color=CONTROLLER)
    ax.text(34.5, mid - 4.4, "past actions", ha="right", va="center", fontsize=8.5, color=CONTROLLER,
            fontweight="bold")
    arrow(ax, (53.0, mid), (57.0, mid))
    return mid


def episodes(ax, data, base, method_labels, method_name, color):
    """Reference and the method's episodes on the test window."""
    mid = base + 9.0
    rows = mini_axes(ax, 110.0, mid - 6.0, 28.0, 12.0)
    rows.set_xlim(0, len(data["prob"]))
    rows.set_ylim(-1.0, 2.0)
    rows.axis("off")
    bar_row(rows, data["reference"], 1.2, INK)
    bar_row(rows, method_labels, 0.0, color)
    ax.text(109.0, mid + 2.8, "Reference", ha="right", va="center", fontsize=9.5, color=INK, fontweight="bold")
    ax.text(109.0, mid - 2.0, method_name, ha="right", va="center", fontsize=9.5, color=color, fontweight="bold")
    return rows


def panel_post_processing(ax, data, runs, base):
    ax.text(2.5, base + 19.0, "(a) Tuned post-processing: fixed rules", fontsize=11.5, fontweight="bold",
            color=POST, va="center")
    mid = detector_and_probability(ax, data, base, show_threshold=True)
    box(ax, 57.0, mid - 3.5, 11.0, 7.0, "Threshold\n$\\theta$", edge=POST)
    arrow(ax, (68.0, mid), (70.5, mid), color=POST)
    box(ax, 70.5, mid - 3.5, 11.0, 7.0, "Merge gaps\n$< g$", edge=POST)
    arrow(ax, (81.5, mid), (84.0, mid), color=POST)
    box(ax, 84.0, mid - 3.5, 11.0, 7.0, "Drop runs\n< 30 s", edge=POST)
    arrow(ax, (95.0, mid), (99.0, mid), color=POST)

    rows = episodes(ax, data, base, data["post"], "Post-proc.", POST)
    false_run, first_fragment, main_fragment = runs["post"]
    rows.annotate("false episode", xy=(np.mean(false_run), -0.3), xytext=(np.mean(false_run), -0.95),
                  ha="center", va="center", fontsize=8, color=INK,
                  arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8, mutation_scale=8))
    gap_mid = (first_fragment[1] + main_fragment[0]) / 2
    rows.annotate("split (gap > $g$)", xy=(gap_mid, -0.3), xytext=(gap_mid + 60, -0.95),
                  ha="left", va="center", fontsize=8, color=INK,
                  arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8, mutation_scale=8))


def panel_controller(ax, data, runs, base):
    ax.text(2.5, base + 19.0, "(b) PPO controller: learned decision stage", fontsize=11.5,
            fontweight="bold", color=CONTROLLER, va="center")
    mid = decision_inputs(ax, data, runs, base)
    box(ax, 59.0, mid - 3.5, 12.0, 7.0, "Policy\nMLP")
    arrow(ax, (71.0, mid), (82.0, mid), color=CONTROLLER)
    box(ax, 82.0, mid - 3.5, 12.0, 7.0, "30-s\nexit mask", edge=MASK_EDGE, fill=MASK_FILL)
    arrow(ax, (94.0, mid), (99.0, mid), color=CONTROLLER)
    # The preceding action feeds the next decision: the loop post-processing lacks.
    arrow(ax, (88.0, mid - 3.5), (65.0, mid - 3.5), color=CONTROLLER, lw=2.0, connection="arc3,rad=-0.35")
    ax.text(76.5, base + 0.6, "previous action $a_{t-1}$", ha="center", fontsize=9.5,
            fontweight="bold", color=CONTROLLER)

    rows = episodes(ax, data, base, data["controller"], "Controller", CONTROLLER)
    # Two-minute scale bar instead of an absolute clock.
    end = len(data["prob"])
    rows.plot([end - 130, end - 10], [-0.85, -0.85], color=INK, linewidth=1.4)
    rows.text(end - 140, -0.85, "2 min", ha="right", va="center", fontsize=8, color=INK)


def render(data, runs):
    fig = plt.figure(figsize=(14.0, 5.4))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 141)
    ax.set_ylim(-0.5, 44.5)
    ax.axis("off")
    for y0 in (22.5, 0.0):
        ax.add_patch(FancyBboxPatch((0.6, y0), 139.8, 21.5, boxstyle="round,pad=0,rounding_size=1.2",
                                    linewidth=1.6, edgecolor=INK, facecolor="none", zorder=1))
    panel_post_processing(ax, data, runs, base=23.0)
    panel_controller(ax, data, runs, base=0.5)
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS / f"{OUT_NAME}.{ext}", dpi=300, facecolor="white")


def main():
    data, runs = window_data()
    # Importing ecg_af switches on a seaborn style; the paper figures use matplotlib's default.
    plt.style.use("default")
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})
    render(data, runs)
    print(f"wrote {PLOTS / OUT_NAME}.png/.pdf")


if __name__ == "__main__":
    main()

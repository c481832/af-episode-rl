"""Render Figure 4 of rl_framework_paper_v5.md: boundary metrics against matching tolerance.

Top row: pooled test set, Boundary F1 / Se / +P (each the mean of the onset and offset
values). Bottom row: Boundary F1 per test source. Replaces the former Tables 3, 4, and 8 of the v4 draft.

Recomputes the sweeps the notebook (experiment_rl_foundation.ipynb, cells 24 and 30-32)
draws, from the reported controller (its cached test rollouts, see CONTROLLER_ROLLOUTS), the
supervised MLP checkpoint, and tuned post-processing at the validation-selected
theta = 0.5 / g = 15 s (paper section 6.3). The per-tolerance values are also written as CSV.

Run: python scripts/render_tolerance_sweep_figure.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ecg_af.benchmark_eval import boundary_tolerance_sweep
from ecg_af.data.common import load_records_list, split_records_two_layer
from ecg_af.data.predictions import load_fold_results, load_true_second_labels
from ecg_af.postprocess import filter_short_intervals, postprocess_probabilities
from ecg_af.rl.learn import filter_trainable_records
from ecg_af.rl.supervised import load_supervised, supervised_checkpoint_path, supervised_record_labels
from ecg_af.util import GLOBAL_CONFIG

EXPERIMENT = "foundation"
PLOTS = Path(__file__).resolve().parents[1] / "plots"
OUT_NAME = "figure_04_boundary_tolerance_sweep"
CONTROLLER_ROLLOUTS = (
    Path(__file__).resolve().parents[1] / "runs" / EXPERIMENT / "controller" / "rollouts" / "main_table2" / "eed463bbf7632266"
)

TOLERANCES = [1, 2, 3, 5, 10, 15, 20, 30]
BINARY_THRESHOLD = 0.5
POST_THRESHOLD = 0.5
POST_MERGE_GAP = 15.0
MIN_DURATION = 30.0

SOURCES = {"iridia": "IRIDIA-AF", "shdb_af": "SHDB-AF", "long_term_af": "LTAFDB"}

# Same order, colors, and markers in every panel; the controller is drawn last, on top.
METHODS = {
    "Tuned post-processing": {"color": "#eb6834", "marker": "s"},
    "Supervised MLP": {"color": "#1baf7a", "marker": "^"},
    "PPO controller": {"color": "#2a78d6", "marker": "o"},
}

# Table 3 of the previous draft: the pooled Boundary F1 the figure must reproduce.
EXPECTED_TOTAL_BOUNDARY_F1 = {
    "Tuned post-processing": [0.118, 0.161, 0.180, 0.198, 0.210, 0.219, 0.225, 0.236],
    "Supervised MLP": [0.065, 0.104, 0.136, 0.171, 0.200, 0.220, 0.233, 0.247],
    "PPO controller": [0.124, 0.174, 0.203, 0.233, 0.255, 0.264, 0.271, 0.284],
}

TEXT = "#17212B"
MUTED = "#5b6670"
GRID = "#dde3e8"


def post_processing_labels(records):
    baseline_model = GLOBAL_CONFIG["rl"]["base_models"]["segment"]
    n_folds = int(GLOBAL_CONFIG["data"]["k_folds"]["n_splits"])
    probs_all, times_all = load_fold_results(baseline_model, n_folds)
    true_by, pred_by = {}, {}
    for source, record_id in records:
        key = (source, record_id)
        times = times_all[key].astype(int)
        true_by[key] = filter_short_intervals(load_true_second_labels(source, record_id))[times]
        pred_by[key] = postprocess_probabilities(probs_all[key], POST_THRESHOLD, POST_MERGE_GAP, MIN_DURATION)
    return true_by, pred_by


def controller_labels(records):
    # Read from the reported controller's rollout cache, not phase1_best.zip, which later
    # training (the gamma = 0.98 run) has overwritten. check_against_table3 confirms the run.
    true_by, pred_by = {}, {}
    for source, record_id in records:
        with np.load(CONTROLLER_ROLLOUTS / f"{source}|{record_id}.npz", allow_pickle=False) as raw:
            # Same as evaluate_record_set: only the reference is duration-filtered.
            true_by[(source, record_id)] = filter_short_intervals(np.asarray(raw["true"], dtype=np.float32))
            pred_by[(source, record_id)] = np.asarray(raw["pred"], dtype=np.float32)
    return true_by, pred_by


def supervised_labels(records):
    model, _ = load_supervised(supervised_checkpoint_path(), GLOBAL_CONFIG["rl"]["train"]["device"])
    return supervised_record_labels(model, records)


def subset(labels, keys):
    true_by, pred_by = labels
    return {k: true_by[k] for k in keys}, {k: pred_by[k] for k in keys}


def sweep_table(labels_by_method, records):
    """One row per (group, method, tolerance) with pooled boundary metrics."""
    groups = {"Total": records}
    for source, name in SOURCES.items():
        groups[name] = [key for key in records if key[0] == source]

    frames = []
    for group, keys in groups.items():
        for method, labels in labels_by_method.items():
            sweep = boundary_tolerance_sweep(*subset(labels, keys), TOLERANCES, BINARY_THRESHOLD, event="AF")
            frame = sweep[["Onset F1", "Offset F1", "Boundary F1", "Onset Se", "Offset Se", "Onset +P", "Offset +P"]].copy()
            frame["Boundary Se"] = (frame["Onset Se"] + frame["Offset Se"]) / 2
            frame["Boundary +P"] = (frame["Onset +P"] + frame["Offset +P"]) / 2
            frame.index.name = "tolerance_s"
            frame = frame.reset_index()
            frame.insert(0, "method", method)
            frame.insert(0, "group", group)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def check_against_table3(table):
    for method, expected in EXPECTED_TOTAL_BOUNDARY_F1.items():
        rows = table[(table["group"] == "Total") & (table["method"] == method)]
        got = rows["Boundary F1"].round(3).tolist()
        if got != expected:
            raise ValueError(f"{method}: recomputed Boundary F1 {got} does not match the paper's {expected}")


def draw_panel(ax, table, group, column, title):
    for method, style in METHODS.items():
        rows = table[(table["group"] == group) & (table["method"] == method)]
        ax.plot(
            rows["tolerance_s"], rows[column], color=style["color"], marker=style["marker"],
            markersize=5, linewidth=1.8, label=method,
        )
    ax.set_title(title, fontsize=10, color=TEXT, loc="left")
    ax.set_xticks([1, 5, 10, 15, 20, 30])
    ax.set_xlim(0, 31)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def render(table):
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4))
    top = [("Boundary F1", "(a) Boundary F1, all test sources"),
           ("Boundary Se", "(b) Boundary Se, all test sources"),
           ("Boundary +P", "(c) Boundary +P, all test sources")]
    for ax, (column, title) in zip(axes[0], top):
        draw_panel(ax, table, "Total", column, title)
    for ax, (letter, name) in zip(axes[1], zip("def", SOURCES.values())):
        draw_panel(ax, table, name, "Boundary F1", f"({letter}) Boundary F1, {name}")

    # Bottom row shares one scale so sources compare directly.
    bottom_max = table[table["group"] != "Total"]["Boundary F1"].max()
    for ax in axes[1]:
        ax.set_ylim(0, bottom_max * 1.08)
    for ax in axes[1]:
        ax.set_xlabel("Matching tolerance δ (s)", fontsize=9, color=TEXT)
    axes[0][0].set_ylim(0, None)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="upper center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS / f"{OUT_NAME}.{ext}", dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.1)


def main():
    GLOBAL_CONFIG.load_config(EXPERIMENT)
    GLOBAL_CONFIG.name = EXPERIMENT
    test_records = filter_trainable_records(split_records_two_layer(load_records_list())[0]["layer1"]["test"])

    labels_by_method = {
        "Tuned post-processing": post_processing_labels(test_records),
        "Supervised MLP": supervised_labels(test_records),
        "PPO controller": controller_labels(test_records),
    }
    table = sweep_table(labels_by_method, test_records)
    check_against_table3(table)
    table.round(4).to_csv(PLOTS / f"{OUT_NAME}.csv", index=False)

    # Importing ecg_af switches on a seaborn style; the paper figures use matplotlib's default.
    plt.style.use("default")
    matplotlib.rcParams.update({"font.family": "DejaVu Sans"})
    render(table)
    print(f"wrote {PLOTS / OUT_NAME}.png/.pdf/.csv")


if __name__ == "__main__":
    main()

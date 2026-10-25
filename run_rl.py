#!/usr/bin/env python3

import argparse
import fcntl
import json
from pathlib import Path

import pandas as pd
import yaml

from ecg_af.benchmark_eval import boundary_tolerance_sweep
from ecg_af.data.common import experiment_run_dir, load_records_list, split_records_two_layer
from ecg_af.rl import fit_controller
from ecg_af.rl.observations import load_controller_arrays
from ecg_af.util import GLOBAL_CONFIG

BOUNDARY_TOLERANCES = [1, 2, 3, 5, 10, 15, 20, 30]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate RL PPO controller.",
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment key in config.yaml experiments.",
    )
    parser.add_argument(
        "--sweep",
        default=None,
        help="YAML list of override sets: [{tag: name, dotted.path: value, ...}, ...].",
    )
    parser.add_argument(
        "--load-checkpoint-phase1",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip phase-1 training and reload the saved phase-1 checkpoint.",
    )
    parser.add_argument(
        "--load-checkpoint-phase2",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip phase-2 training and reload the saved final/phase-2 checkpoint.",
    )
    return parser.parse_args()


def load_sweep_specs(path):
    with open(path, "r") as f:
        specs = yaml.safe_load(f)
    if not isinstance(specs, list):
        raise ValueError(f"Sweep file {path} must be a YAML list of override dicts, got {type(specs).__name__}")
    if not specs:
        raise ValueError(f"Sweep file {path} is empty")
    tags = []
    for spec in specs:
        if not isinstance(spec, dict):
            raise ValueError(f"Sweep item must be a mapping, got {spec!r}")
        if "tag" not in spec:
            raise ValueError(f"Sweep item missing tag: {spec!r}")
        tag = spec["tag"]
        if Path(tag).name != tag or tag in {".", ".."}:
            raise ValueError(f"Invalid sweep tag {tag!r}")
        tags.append(tag)
    dupes = {tag for tag in tags if tags.count(tag) > 1}
    if dupes:
        raise ValueError(f"Duplicate sweep tags: {sorted(dupes)}")
    return specs


def mean_boundary_metrics(rollouts):
    if not rollouts:
        raise ValueError("Cannot score boundary metrics on an empty record set")
    sweep = boundary_tolerance_sweep(
        {key: pack["true"] for key, pack in rollouts.items()},
        {key: pack["pred"] for key, pack in rollouts.items()},
        BOUNDARY_TOLERANCES,
        0.5,
    )
    return {
        "boundary_f1": float(sweep["Boundary F1"].mean(skipna=False)),
        "onset_f1": float(sweep["Onset F1"].mean(skipna=False)),
        "offset_f1": float(sweep["Offset F1"].mean(skipna=False)),
        "ue": float(sweep["Ue"].mean(skipna=False)),
    }


def append_sweep_summary(overrides, pass_history_df, second_df, rollouts):
    row = {
        "tag": GLOBAL_CONFIG.run_tag,
        "overrides": json.dumps(overrides, sort_keys=True),
    }
    if not pass_history_df.empty:
        best = pass_history_df.loc[pass_history_df["selection_score"].idxmax()]
        row["selection_score"] = float(best["selection_score"])
    if second_df is not None:
        row["duration_f1"] = float(second_df.loc["f1", "total"])
    if rollouts:
        row["n_records"] = len(rollouts)
        row.update(mean_boundary_metrics(rollouts))
        for source in GLOBAL_CONFIG["data"]["override_test"]:
            source_rollouts = {key: pack for key, pack in rollouts.items() if key[0] == source}
            if not source_rollouts:
                raise ValueError(f"No test rollouts for override_test source {source!r}")
            row[f"n_{source}"] = len(source_rollouts)
            row[f"boundary_f1_{source}"] = mean_boundary_metrics(source_rollouts)["boundary_f1"]
    path = experiment_run_dir() / "controller" / "sweep_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(".csv.lock")
    with open(lock_path, "a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if path.exists():
            df = pd.concat([pd.read_csv(path), pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df.to_csv(path, index=False)
    GLOBAL_CONFIG.logger.info("Appended sweep summary to %s", path)


def log_controller_result(pass_history_df, second_df, phase_test_df):
    if phase_test_df is not None:
        GLOBAL_CONFIG.logger.info(
            "Layer-2 test (phase-1 warm-start vs phase-2 final):\n%s",
            phase_test_df.round(4),
        )

    if not pass_history_df.empty:
        rank_cols = [
            "phase",
            "pass",
            "selection_metric",
            "selection_score",
            "is_best",
            "val_f1",
            "train_f1",
            "val_reward_total",
            "train_reward_total",
        ]
        ranked = pass_history_df[rank_cols].sort_values(
            "selection_score", ascending=False
        ).round(4)
        GLOBAL_CONFIG.logger.info("Pass history:\n%s", ranked)
        best = pass_history_df.loc[pass_history_df["selection_score"].idxmax()]
        final_row = pass_history_df.iloc[-1]
        GLOBAL_CONFIG.logger.info(
            "Best selected pass : %d (%s=%.4f, val_f1=%.4f | train_f1=%.4f)",
            int(best["pass"]),
            best["selection_metric"],
            best["selection_score"],
            best["val_f1"],
            best["train_f1"],
        )
        GLOBAL_CONFIG.logger.info(
            "Last pass : %d (%s=%.4f, val_f1=%.4f)",
            int(final_row["pass"]),
            final_row["selection_metric"],
            final_row["selection_score"],
            final_row["val_f1"],
        )

    if second_df is not None:
        second_total = second_df["total"].loc[["sensitivity", "ppv", "f1"]]
        GLOBAL_CONFIG.logger.info(
            "Test evaluation (PPO controller, best-on-val checkpoint):\n%s",
            pd.Series({"duration_f1": second_df.loc["f1", "total"]})
            .round(3)
            .to_frame("PPO controller"),
        )
        GLOBAL_CONFIG.logger.info(
            "Second-level metrics (test, total):\n%s",
            second_total.round(4).to_frame("total"),
        )


def run_controller(args, overrides):
    load_controller_arrays.cache_clear()
    folds = split_records_two_layer(load_records_list())
    layer1 = folds[0]["layer1"]
    layer2 = folds[0]["layer2"]
    GLOBAL_CONFIG.logger.info(
        "layer1 train: %d val: %d test: %d | layer2 train: %d val: %d test: %d",
        len(layer1["train"]),
        len(layer1["val"]),
        len(layer1["test"]),
        len(layer2["train"]),
        len(layer2["val"]),
        len(layer2["test"]),
    )
    if GLOBAL_CONFIG.run_tag is not None:
        GLOBAL_CONFIG.logger.info(
            "Sweep tag=%s overrides=%s checkpoint_dir=%s",
            GLOBAL_CONFIG.run_tag,
            json.dumps(overrides, sort_keys=True),
            experiment_run_dir() / "controller" / GLOBAL_CONFIG.run_tag,
        )
    if not GLOBAL_CONFIG["data"]["split"]["layer2"]["train_sources"]:
        GLOBAL_CONFIG.logger.info(
            "Phase 2 disabled: controller trains/evaluates on layer-1 splits"
        )

    _, rollouts, second_df, pass_history_df, phase_test_df = fit_controller(
        layer1,
        layer2,
        load_checkpoint_phase1=args.load_checkpoint_phase1,
        load_checkpoint_phase2=args.load_checkpoint_phase2,
    )
    log_controller_result(pass_history_df, second_df, phase_test_df)
    if GLOBAL_CONFIG.run_tag is not None:
        append_sweep_summary(overrides, pass_history_df, second_df, rollouts)


def main():
    args = parse_args()
    specs = load_sweep_specs(args.sweep) if args.sweep is not None else [None]

    for spec in specs:
        GLOBAL_CONFIG.load_config(args.experiment)
        GLOBAL_CONFIG.debug_print = False
        overrides = {}
        if spec is not None:
            overrides = {k: v for k, v in spec.items() if k != "tag"}
            GLOBAL_CONFIG.apply_overrides(overrides)
            GLOBAL_CONFIG.run_tag = spec["tag"]
        run_controller(args, overrides)


if __name__ == "__main__":
    main()

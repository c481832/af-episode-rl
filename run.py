#!/usr/bin/env python3

import argparse

import numpy as np
import pandas as pd

from ecg_af.analysis import get_metrics
from ecg_af.data.common import build_folds
from ecg_af.data.predictions import load_record_fold_results, load_true_second_labels
from ecg_af.training import fit_model_k_fold
from ecg_af.util import GLOBAL_CONFIG


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ECG AF experiment.",
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Environment key in config.yaml experiments.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model key from ecg_af/config.yaml defaults.models.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        help="Mode key from ecg_af/config.yaml defaults.modes.",
    )
    parser.add_argument(
        "--reload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse saved checkpoints when available.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    GLOBAL_CONFIG.load_config(args.experiment)
    reload_checkpoint = args.reload

    folds = build_folds()

    k_fold_info = []
    for i in range(len(folds)):
        tmp = pd.concat({k: pd.DataFrame(folds[i][k]) for k in folds[i]})
        tmp.columns = ['source', 'record']
        tmp = tmp.reset_index(level=0).set_index(['source', 'record']).squeeze()
        k_fold_info.append(tmp.to_frame(f'fold_{i}'))
    k_fold_info = pd.concat(k_fold_info, axis=1)
    GLOBAL_CONFIG.logger.info("Fold distribution:\n%s", k_fold_info)

    fit_model_k_fold(
        folds,
        reload_checkpoint,
        args.mode,
        args.model,
    )

    if args.mode == "noise":
        GLOBAL_CONFIG.logger.info("Noise quality predictions exported")
        return

    split = 'test'
    n_folds = int(GLOBAL_CONFIG["data"]["k_folds"]["n_splits"])
    pred_records_probs = {}
    true_records_labels = {}
    for source, record_id in folds[0][split]:
        record_key = (source, record_id)
        probs, times = load_record_fold_results(args.model, source, record_id, n_folds)
        pred_records_probs[record_key] = probs
        true_records_labels[record_key] = load_true_second_labels(source, record_id)[
            times.astype(int)
        ].astype(np.float32)

    metrics = {
        "_".join(k): get_metrics(true_records_labels[k], pred_records_probs[k])
        for k in true_records_labels
    }
    metrics["total"] = get_metrics(
        np.concatenate(list(true_records_labels.values())),
        np.concatenate(list(pred_records_probs.values())),
    )
    metrics_df = pd.DataFrame(metrics).loc[['sensitivity', 'ppv', 'f1']]

    GLOBAL_CONFIG.logger.info("Metrics (%s):\n%s", split, metrics_df)


if __name__ == "__main__":
    main()

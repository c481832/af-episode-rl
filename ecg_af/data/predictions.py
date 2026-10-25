from functools import lru_cache
from pathlib import Path

import numpy as np

from ..util import GLOBAL_CONFIG
from .common import experiment_run_dir, load_processed_record

def fold_predictions_path(model_name, fold):
    return experiment_run_dir() / "models" / model_name / f"fold_{fold:02d}" / "predictions.npz"


def true_labels_path(source, record_id):
    return experiment_run_dir() / "true_labels" / f"{source}|{record_id}.npz"


def save_fold_results(path, record_probs, record_times):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    for (source, record_id), probs in record_probs.items():
        times = record_times[(source, record_id)]
        data[f"{source}|{record_id}|probs"] = probs.astype(np.float32)
        data[f"{source}|{record_id}|times"] = times.astype(np.float64)
    np.savez(path, **data)


@lru_cache(maxsize=None)
def load_record_fold_results(model_name, source, record_id, n_folds):
    probs_key = f"{source}|{record_id}|probs"
    times_key = f"{source}|{record_id}|times"
    probs_list = []
    times = None
    for fold in range(n_folds):
        path = fold_predictions_path(model_name, fold)
        with np.load(path, allow_pickle=False) as raw:
            probs_list.append(np.asarray(raw[probs_key], dtype=np.float32))
            if times is None:
                times = np.asarray(raw[times_key], dtype=np.float64)
    averaged = np.mean(np.stack(probs_list), axis=0)
    return averaged, times


@lru_cache(maxsize=None)
def load_fold_results(model_name, n_folds):
    all_probs = {}
    record_times = {}
    for fold in range(n_folds):
        path = fold_predictions_path(model_name, fold)
        with np.load(path, allow_pickle=False) as raw:
            seen_pairs = set()
            for key in raw.files:
                source, record_id, field = key.split("|")
                if field != "probs":
                    continue
                pair = (source, record_id)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                probs = np.asarray(raw[key], dtype=np.float32)
                if pair not in all_probs:
                    all_probs[pair] = []
                    record_times[pair] = np.asarray(
                        raw[f"{source}|{record_id}|times"], dtype=np.float64
                    )
                all_probs[pair].append(probs)
    averaged = {pair: np.mean(np.stack(probs_list), axis=0) for pair, probs_list in all_probs.items()}
    return averaged, record_times


def bin_predictions_per_second(values, times):
    if len(times) == 0:
        return np.zeros(0, dtype=np.float32)
    n_seconds = int(np.floor(times[-1])) + 1
    bin_indices = np.clip(times.astype(np.int64), 0, n_seconds - 1)
    sums = np.bincount(bin_indices, weights=values, minlength=n_seconds).astype(np.float32)
    counts = np.bincount(bin_indices, minlength=n_seconds).astype(np.float32)
    counts[counts == 0] = 1.0
    return sums / counts


@lru_cache(maxsize=None)
def load_true_second_labels(source, record_id):
    path = true_labels_path(source, record_id)
    if path.exists():
        with np.load(path, allow_pickle=False) as raw:
            return np.asarray(raw["labels"], dtype=np.int8)

    _, resampled_labels, _ = load_processed_record(source, record_id)
    target_fs = GLOBAL_CONFIG["data"]["preprocessing"]["target_fs"]
    times = np.arange(len(resampled_labels), dtype=np.float64) / target_fs
    per_second = bin_predictions_per_second(resampled_labels.astype(np.float32), times)
    labels = (per_second > 0).astype(np.int8)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, labels=labels)
    return labels

from functools import lru_cache

import numpy as np

from ..data.common import load_processed_record
from ..data.heart_rate import heart_rate_per_second, process_heart_rate
from ..data.predictions import (
    bin_predictions_per_second,
    load_fold_results,
    load_true_second_labels,
)
from ..util import GLOBAL_CONFIG


def bin_name(name, lo, hi):
    if (lo, hi) == (0, 0):
        return name
    if lo == hi:
        return f"{name}_d{lo:+d}"
    return f"{name}_d{lo:+d}_{hi:+d}"


def base_feature_spec():
    rl = GLOBAL_CONFIG["rl"]
    features = [
        (role, f"{model}_noise_flag" if role == "quality" else f"{model}_p")
        for role, model in rl["base_models"].items()
        if model is not None
    ]
    if bool(rl["observation_fields"]["heart_rate"]["enabled"]):
        features.append(("heart_rate", "heart_rate"))
    window = rl["obs_window"]
    return [(name, window.get(key, window["default"])) for key, name in features]


def observation_names():
    names = []
    for name, bins in base_feature_spec():
        names.extend(bin_name(name, lo, hi) for lo, hi in bins)
    fields = GLOBAL_CONFIG["rl"]["observation_fields"]
    if bool(fields["last_state"]["enabled"]):
        names.append("last_state")
    return names


def window_columns(column, bins):
    n = len(column)
    if n == 0:
        return [column] * len(bins)
    back = max(0, -min(lo for lo, _ in bins))
    fwd = max(0, max(hi for _, hi in bins))
    padded = np.pad(column, (back, fwd), mode="edge")
    blocks = [
        np.mean([padded[back + k:back + k + n] for k in range(lo, hi + 1)], axis=0, dtype=np.float32)
        for lo, hi in bins
    ]
    return [b if (lo, hi) == (0, 0) else b - column for b, (lo, hi) in zip(blocks, bins)]


def probs_per_second(probs, times, n_seconds):
    raw = bin_predictions_per_second(probs, times)
    out = np.zeros(n_seconds, dtype=np.float32)
    out[:min(len(raw), n_seconds)] = raw[:n_seconds]
    return out


def beat_probs_to_seconds(times, probs, n_seconds, carry_seconds):
    second_axis = np.arange(n_seconds, dtype=np.float64)
    right = np.searchsorted(times, second_axis, side="right") - 1
    aligned = np.zeros(n_seconds, dtype=np.float32)
    valid = right >= 0
    aligned[valid] = probs[right[valid]]
    elapsed = np.full(n_seconds, np.inf, dtype=np.float64)
    elapsed[valid] = second_axis[valid] - times[right[valid]]
    fresh = valid & (elapsed <= carry_seconds)
    aligned[~fresh] = 0.0
    return aligned


@lru_cache(maxsize=None)
def load_controller_arrays(source, record_id):
    rl = GLOBAL_CONFIG["rl"]
    base_models = rl["base_models"]
    roles = [role for role, model in base_models.items() if model is not None]
    n_folds = int(GLOBAL_CONFIG["data"]["k_folds"]["n_splits"])

    segment_model = base_models["segment"]
    model_names = {m for m in base_models.values() if m is not None}
    loaded = {}
    for name in model_names:
        averaged_probs, record_times = load_fold_results(name, n_folds)
        key = (source, record_id)
        loaded[name] = (averaged_probs[key], record_times[key])

    seg_probs, seg_times = loaded[segment_model]
    n_seconds = len(bin_predictions_per_second(seg_probs, seg_times))

    base_pred = (probs_per_second(seg_probs, seg_times, n_seconds) >= 0.5).astype(np.int8)

    true_second_labels = load_true_second_labels(source, record_id)[:n_seconds].astype(np.int64)

    columns = []
    for role in roles:
        probs, times = loaded[base_models[role]]
        if role == "beat":
            carry = float(rl["beat_carry_seconds"])
            column = beat_probs_to_seconds(times, probs, n_seconds, carry)
        elif role == "quality":
            threshold = float(GLOBAL_CONFIG["models"][base_models[role]]["threshold"])
            column = (probs_per_second(probs, times, n_seconds) < threshold).astype(np.float32)
        else:
            column = probs_per_second(probs, times, n_seconds)
        columns.append(column)

    if bool(rl["observation_fields"]["heart_rate"]["enabled"]):
        _, processed_labels, processed_r_peaks = load_processed_record(source, record_id)
        beat_times, peak_rr_intervals, _ = process_heart_rate(processed_labels, processed_r_peaks)
        carry = float(rl["beat_carry_seconds"])
        columns.append(heart_rate_per_second(beat_times, peak_rr_intervals, n_seconds, carry))

    windowed = []
    for (_, bins), column in zip(base_feature_spec(), columns):
        windowed.extend(window_columns(column, bins))
    base_features = np.stack(windowed, axis=1).astype(np.float32)

    true_states = true_second_labels.astype(np.int8)
    return {
        "base_features": base_features,
        "true_states": true_states,
        "base_pred": base_pred,
    }


def build_observation(base_features, last_state):
    obs = np.asarray(base_features, dtype=np.float32)
    fields = GLOBAL_CONFIG["rl"]["observation_fields"]
    if bool(fields["last_state"]["enabled"]):
        obs = np.append(obs, last_state)
    return obs.astype(np.float32)

import numpy as np

from ..postprocess import labels_to_intervals
from ..util import GLOBAL_CONFIG


def score_interval(true_interval, pred_intervals, n_labels):
    error_cap = float(GLOBAL_CONFIG["rl"]["evaluation_metric"]["error_cap_seconds"])

    start, end = true_interval
    has_onset = start > 0
    has_offset = end < n_labels

    overlapping = [p for p in pred_intervals if min(end, p[1]) > max(start, p[0])]
    envelope_start = min((p[0] for p in overlapping), default=np.nan)
    envelope_end = max((p[1] for p in overlapping), default=np.nan)
    interval_details = {
        "true_start": float(start),
        "true_end": float(end),
        "true_duration": float(end - start),
        "has_onset": bool(has_onset),
        "has_offset": bool(has_offset),
        "pred_envelope_start": float(envelope_start),
        "pred_envelope_end": float(envelope_end),
        "pred_envelope_duration": (
            float(envelope_end - envelope_start) if overlapping else np.nan
        ),
        "overlapping_pred_intervals": tuple(overlapping),
    }

    if not has_onset and not has_offset:
        return {
            **interval_details,
            "onset_error": np.nan,
            "offset_error": np.nan,
            "fragment_count": np.nan,
        }

    if not overlapping:
        return {
            **interval_details,
            "onset_error": np.nan,
            "offset_error": np.nan,
            "fragment_count": 0,
        }

    onset_error = envelope_start - start if has_onset else np.nan
    offset_error = envelope_end - end if has_offset else np.nan

    return {
        **interval_details,
        "onset_error": (
            float(onset_error) if has_onset and abs(onset_error) <= error_cap else np.nan
        ),
        "offset_error": (
            float(offset_error) if has_offset and abs(offset_error) <= error_cap else np.nan
        ),
        "fragment_count": len(overlapping),
    }


def score_labels(true_labels, pred_labels):
    true_labels = np.asarray(true_labels)
    true_intervals = labels_to_intervals(true_labels)
    pred_intervals = labels_to_intervals(pred_labels)

    if not true_intervals:
        return {
            "median_onset_error": np.nan,
            "median_offset_error": np.nan,
            "n_events": 0,
            "n_true_events": 0,
        }, []

    events = [
        score_interval(t, pred_intervals, len(true_labels)) for t in true_intervals
    ]

    onset_errors = [e["onset_error"] for e in events if not np.isnan(e["onset_error"])]
    offset_errors = [e["offset_error"] for e in events if not np.isnan(e["offset_error"])]
    scored = [e for e in events if not np.isnan(e["fragment_count"])]

    summary = {
        "median_onset_error": float(np.median(onset_errors)) if onset_errors else np.nan,
        "median_offset_error": float(np.median(offset_errors)) if offset_errors else np.nan,
        "n_events": len(scored),
        "n_true_events": len(events),
    }
    return summary, events


def truth_episode_type(true_labels):
    true_labels = np.asarray(true_labels).ravel()
    true_iv = labels_to_intervals(true_labels)
    n = len(true_labels)
    if not true_iv:
        return "non_af"
    if len(true_iv) == 1 and true_iv[0][0] == 0 and true_iv[0][1] == n:
        return "pure_af"
    return "mixed"

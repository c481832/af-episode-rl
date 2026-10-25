import numpy as np

from .util import GLOBAL_CONFIG


def keep_interval(start, end, min_duration_seconds, record_duration_seconds):
    has_unclear_onset = start == 0
    has_unclear_offset = (
        record_duration_seconds is not None and end == record_duration_seconds
    )
    return (
        has_unclear_onset
        or has_unclear_offset
        or (end - start) >= min_duration_seconds
    )


def labels_to_intervals(labels, min_duration_seconds=0.0):
    labels = np.asarray(labels, dtype=np.int64).ravel()
    padded = np.pad(labels, (1, 1))
    changes = np.diff(padded)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return [
        (float(s), float(e))
        for s, e in zip(starts, ends)
        if keep_interval(s, e, min_duration_seconds, len(labels))
    ]


def post_process_intervals(
    intervals,
    merge_gap_seconds,
    min_duration_seconds,
    record_duration_seconds=None,
):
    intervals = sorted((float(s), float(e)) for s, e in intervals if e > s)
    if not intervals:
        return []

    merged = [intervals[0]]
    for s, e in intervals[1:]:
        last_s, last_e = merged[-1]
        if s - last_e <= merge_gap_seconds:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))
    return [
        (s, e)
        for s, e in merged
        if keep_interval(
            s,
            e,
            min_duration_seconds,
            record_duration_seconds,
        )
    ]


def intervals_to_labels(n_seconds, intervals):
    labels = np.zeros(n_seconds, dtype=np.int64)
    for start_t, end_t in intervals:
        start = int(start_t)
        end = int(end_t)
        labels[start:end] = 1
    return labels


def postprocess_probabilities(probs, threshold, merge_gap_seconds, min_duration_seconds):
    """Threshold per-second probabilities, then merge and length-filter the intervals."""
    binary = (np.asarray(probs) >= threshold).astype(np.int64)
    intervals = post_process_intervals(
        labels_to_intervals(binary),
        merge_gap_seconds,
        min_duration_seconds,
        record_duration_seconds=len(binary),
    )
    return intervals_to_labels(len(binary), intervals)


def filter_short_intervals(labels):
    labels = np.asarray(labels)
    min_duration_seconds = float(
        GLOBAL_CONFIG["postprocessing"]["min_duration_seconds"]
    )
    intervals = labels_to_intervals(
        labels,
        min_duration_seconds=min_duration_seconds,
    )
    return intervals_to_labels(len(labels), intervals).astype(labels.dtype)


def post_process_labels(all_records_labels, all_records_probs):
    true_labels = {}
    pred_labels = {}
    merge_gap_seconds = float(GLOBAL_CONFIG["postprocessing"]["merge_gap_seconds"])
    min_duration_seconds = float(GLOBAL_CONFIG["postprocessing"]["min_duration_seconds"])
    threshold = GLOBAL_CONFIG["postprocessing"]["threshold"]

    for record_key in all_records_labels:
        pred_labels[record_key] = postprocess_probabilities(
            all_records_probs[record_key],
            threshold,
            merge_gap_seconds,
            min_duration_seconds,
        )
        true = np.where(all_records_labels[record_key] >= 0.5, 1, 0)
        true_labels[record_key] = filter_short_intervals(true)
    return true_labels, pred_labels

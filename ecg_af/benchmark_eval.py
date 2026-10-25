import numpy as np
import pandas as pd

from .data.predictions import load_true_second_labels
from .postprocess import (
    filter_short_intervals,
    intervals_to_labels,
    labels_to_intervals,
    post_process_intervals,
    postprocess_probabilities,
)
from .util import GLOBAL_CONFIG

BOUNDARIES = ((0, "Onset"), (1, "Offset"))
EVENTS = ("AF", "AFib", "AFL")
BOUNDARY_TOLERANCE = 3.0
EVENT_COLUMNS = (
    "TPs",
    "FN",
    "TPp",
    "FP",
    "ESe",
    "E+P",
    "DSe",
    "D+P",
    "Overlap",
    "Ref duration",
    "Test duration",
    "Onset TP",
    "Onset FN",
    "Onset FP",
    "Onset Se",
    "Onset +P",
    "Onset MAE",
    "Onset SAE",
    "Offset TP",
    "Offset FN",
    "Offset FP",
    "Offset Se",
    "Offset +P",
    "Offset MAE",
    "Offset SAE",
    "Ue",
    "TPr",
    "FNr",
    "FPr",
    "TNr",
    "Sp",
    "NPV",
    "FPR",
    "Error Rate",
    "F1",
    "Accuracy",
)


def interval_intersection(intervals_1, intervals_2):
    if not intervals_1 or not intervals_2:
        return []
    intersections = []
    i = 0
    j = 0
    while i < len(intervals_1) and j < len(intervals_2):
        lo = max(intervals_1[i][0], intervals_2[j][0])
        hi = min(intervals_1[i][1], intervals_2[j][1])
        if lo < hi:
            intersections.append([lo, hi])
        if intervals_1[i][1] < intervals_2[j][1]:
            i += 1
        else:
            j += 1
    return intersections


def interval_duration(intervals):
    return sum(end - start for start, end in intervals)


def interval_overlap(interval_1, interval_2):
    return max(interval_1[0], interval_2[0]) < min(interval_1[1], interval_2[1])


def ratio(numerator, denominator):
    if denominator == 0:
        return np.nan
    return numerator / denominator


def f1(tp, fn, fp):
    return ratio(2 * tp, 2 * tp + fn + fp)


def harmonic_f1(sensitivity, ppv):
    sensitivity = np.asarray(sensitivity, dtype=np.float64)
    ppv = np.asarray(ppv, dtype=np.float64)
    denominator = sensitivity + ppv
    with np.errstate(invalid="ignore", divide="ignore"):
        scores = np.where(denominator > 0, 2 * sensitivity * ppv / denominator, np.nan)
    return scores if scores.ndim else float(scores)


def episode_tp_fn(ref, pred):
    if not ref and not pred:
        return np.nan, np.nan
    if not pred:
        return 0, len(ref)
    if not ref:
        return 0, 0

    tp = 0
    fn = 0
    for ref_interval in ref:
        if any(interval_overlap(ref_interval, pred_interval) for pred_interval in pred):
            tp += 1
        else:
            fn += 1
    return tp, fn


def episode_tp_fp(ref, pred):
    if not ref and not pred:
        return np.nan, np.nan
    if not pred:
        return 0, 0
    if not ref:
        return 0, len(pred)

    tp = 0
    fp = 0
    for pred_interval in pred:
        if any(interval_overlap(ref_interval, pred_interval) for ref_interval in ref):
            tp += 1
        else:
            fp += 1
    return tp, fp


def boundary_times(intervals, boundary_index, record_duration):
    if boundary_index == 0:
        return [interval[0] for interval in intervals if interval[0] > 0]
    return [interval[1] for interval in intervals if interval[1] < record_duration]


def boundary_matches(ref_times, pred_times, tolerance):
    candidates = []
    for ref_index, ref_time in enumerate(ref_times):
        for pred_index, pred_time in enumerate(pred_times):
            error = pred_time - ref_time
            abs_error = abs(error)
            if abs_error <= tolerance:
                candidates.append((abs_error, ref_index, pred_index, error))

    matches = []
    used_refs = set()
    used_preds = set()
    for _, ref_index, pred_index, error in sorted(candidates):
        if ref_index in used_refs or pred_index in used_preds:
            continue
        used_refs.add(ref_index)
        used_preds.add(pred_index)
        matches.append((ref_index, pred_index, error))
    return matches


def boundary_nearest_distances(ref_times, pred_times):
    if not ref_times or not pred_times:
        return np.empty(0)
    ref = np.asarray(ref_times, dtype=np.float64)
    pred = np.asarray(pred_times, dtype=np.float64)
    return np.abs(pred[:, None] - ref[None, :]).min(axis=1)


def boundary_ue_distances(ref_intervals, pred_intervals, record_duration):
    n_obs_ref = 0
    n_obs_pred = 0
    distances = []
    for boundary_index, _ in BOUNDARIES:
        ref_times = boundary_times(ref_intervals, boundary_index, record_duration)
        pred_times = boundary_times(pred_intervals, boundary_index, record_duration)
        n_obs_ref += len(ref_times)
        n_obs_pred += len(pred_times)
        if n_obs_ref > 0:
            distances.append(boundary_nearest_distances(ref_times, pred_times))
    if n_obs_ref == 0:
        return np.empty(0), 0
    distances = np.concatenate(distances) if distances else np.empty(0)
    return distances, max(n_obs_ref, n_obs_pred)


def boundary_metrics(ref_intervals, pred_intervals, boundary_index, tolerance, record_duration):
    ref_times = boundary_times(ref_intervals, boundary_index, record_duration)
    pred_times = boundary_times(pred_intervals, boundary_index, record_duration)
    errors = [error for _, _, error in boundary_matches(ref_times, pred_times, tolerance)]
    tp = len(errors)
    fn = len(ref_times) - tp
    fp = len(pred_times) - tp
    sae = sum(abs(error) for error in errors)
    return {
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "Se": ratio(tp, tp + fn),
        "+P": ratio(tp, tp + fp),
        "SAE": sae,
        "MAE": ratio(sae, tp),
    }


def record_classification(tps, fn, tpp, fp):
    tnr = 0 if tps + fn + tpp + fp > 0 else 1
    fpr = 1 if fp >= 1 and tps + fn == 0 else 0
    fnr = 1 if fn >= 1 and tpp + fp == 0 else 0
    tpr = 1 if tnr + fpr + fnr == 0 else 0
    return tpr, tnr, fpr, fnr


def record_metrics(ref_intervals, pred_intervals, record_duration):
    tps, fn = episode_tp_fn(ref_intervals, pred_intervals)
    tpp, fp = episode_tp_fp(ref_intervals, pred_intervals)
    overlap = interval_duration(interval_intersection(ref_intervals, pred_intervals))
    ref_duration = interval_duration(ref_intervals)
    test_duration = interval_duration(pred_intervals)
    onset = boundary_metrics(ref_intervals, pred_intervals, 0, BOUNDARY_TOLERANCE, record_duration)
    offset = boundary_metrics(ref_intervals, pred_intervals, 1, BOUNDARY_TOLERANCE, record_duration)
    ue_distances, ue_n = boundary_ue_distances(ref_intervals, pred_intervals, record_duration)
    ue_hits = float((ue_distances <= BOUNDARY_TOLERANCE).sum())
    tpr, tnr, fpr, fnr = record_classification(tps, fn, tpp, fp)
    sp = ratio(tnr, tnr + fpr)
    error = ratio(fpr + fnr, tpr + tnr + fpr + fnr)
    return {
        "TPs": tps,
        "FN": fn,
        "TPp": tpp,
        "FP": fp,
        "ESe": ratio(tps, tps + fn),
        "E+P": ratio(tpp, tpp + fp),
        "DSe": ratio(overlap, ref_duration),
        "D+P": ratio(overlap, test_duration),
        "Overlap": overlap,
        "Ref duration": ref_duration,
        "Test duration": test_duration,
        "Onset TP": onset["TP"],
        "Onset FN": onset["FN"],
        "Onset FP": onset["FP"],
        "Onset Se": onset["Se"],
        "Onset +P": onset["+P"],
        "Onset MAE": onset["MAE"],
        "Onset SAE": onset["SAE"],
        "Offset TP": offset["TP"],
        "Offset FN": offset["FN"],
        "Offset FP": offset["FP"],
        "Offset Se": offset["Se"],
        "Offset +P": offset["+P"],
        "Offset MAE": offset["MAE"],
        "Offset SAE": offset["SAE"],
        "Ue hits": ue_hits,
        "Ue N": ue_n,
        "Ue": ratio(ue_hits, ue_n),
        "TPr": tpr,
        "FNr": fnr,
        "FPr": fpr,
        "TNr": tnr,
        "Sp": sp,
        "NPV": ratio(tnr, tnr + fnr),
        "FPR": 1 - sp,
        "Error Rate": error,
        "F1": f1(tpr, fnr, fpr),
        "Accuracy": 1 - error,
    }


def event_intervals(labels, threshold):
    binary = (np.asarray(labels, dtype=np.float32) >= threshold).astype(np.int8)
    afib = [[float(s), float(e)] for s, e in labels_to_intervals(filter_short_intervals(binary))]
    return {"AFib": afib, "AFL": [], "AF": afib}


def add_summary_rows(df_event):
    sum_row = {column: np.nan for column in EVENT_COLUMNS}
    for column in (
        "TPs",
        "FN",
        "TPp",
        "FP",
        "Overlap",
        "Ref duration",
        "Test duration",
        "Onset TP",
        "Onset FN",
        "Onset FP",
        "Onset SAE",
        "Offset TP",
        "Offset FN",
        "Offset FP",
        "Offset SAE",
        "Ue hits",
        "Ue N",
        "TPr",
        "FNr",
        "FPr",
        "TNr",
    ):
        sum_row[column] = df_event[column].sum()
    df_event.loc["Sum"] = pd.Series(sum_row)

    tps = float(df_event.loc["Sum", "TPs"])
    fn = float(df_event.loc["Sum", "FN"])
    tpp = float(df_event.loc["Sum", "TPp"])
    fp = float(df_event.loc["Sum", "FP"])
    onset_tp = float(df_event.loc["Sum", "Onset TP"])
    onset_fn = float(df_event.loc["Sum", "Onset FN"])
    onset_fp = float(df_event.loc["Sum", "Onset FP"])
    offset_tp = float(df_event.loc["Sum", "Offset TP"])
    offset_fn = float(df_event.loc["Sum", "Offset FN"])
    offset_fp = float(df_event.loc["Sum", "Offset FP"])
    tpr = float(df_event.loc["Sum", "TPr"])
    tnr = float(df_event.loc["Sum", "TNr"])
    fpr = float(df_event.loc["Sum", "FPr"])
    fnr = float(df_event.loc["Sum", "FNr"])
    sp = ratio(tnr, tnr + fpr)
    error = ratio(fpr + fnr, tpr + tnr + fpr + fnr)

    gross_row = {column: np.nan for column in EVENT_COLUMNS}
    gross_row.update(
        {
            "ESe": ratio(tps, tps + fn),
            "E+P": ratio(tpp, tpp + fp),
            "DSe": ratio(df_event.loc["Sum", "Overlap"], df_event.loc["Sum", "Ref duration"]),
            "D+P": ratio(df_event.loc["Sum", "Overlap"], df_event.loc["Sum", "Test duration"]),
            "Onset Se": ratio(onset_tp, onset_tp + onset_fn),
            "Onset +P": ratio(onset_tp, onset_tp + onset_fp),
            "Onset MAE": ratio(df_event.loc["Sum", "Onset SAE"], onset_tp),
            "Offset Se": ratio(offset_tp, offset_tp + offset_fn),
            "Offset +P": ratio(offset_tp, offset_tp + offset_fp),
            "Offset MAE": ratio(df_event.loc["Sum", "Offset SAE"], offset_tp),
            "Ue": ratio(float(df_event.loc["Sum", "Ue hits"]), float(df_event.loc["Sum", "Ue N"])),
            "Sp": sp,
            "NPV": ratio(tnr, tnr + fnr),
            "FPR": 1 - sp,
            "Error Rate": error,
            "F1": f1(tpr, fnr, fpr),
            "Accuracy": 1 - error,
        }
    )
    df_event.loc["Gross"] = pd.Series(gross_row)
    return df_event


def evaluation_tables(true_by_record, pred_by_record, threshold, true_threshold=0.5):
    rows_by_event = {event: {} for event in EVENTS}
    for record_key in true_by_record:
        ref_events = event_intervals(true_by_record[record_key], true_threshold)
        pred_events = event_intervals(pred_by_record[record_key], threshold)
        name = f"{record_key[0]}_{record_key[1]}" if isinstance(record_key, tuple) else str(record_key)
        record_duration = len(true_by_record[record_key])
        for event in EVENTS:
            rows_by_event[event][name] = record_metrics(
                ref_events[event],
                pred_events[event],
                record_duration,
            )
    tables = {}
    for event in EVENTS:
        df_event = pd.DataFrame.from_dict(rows_by_event[event], orient="index")
        df_event = add_summary_rows(df_event)
        tables[event] = df_event[list(EVENT_COLUMNS)]
    return tables


def gross_summary(tables, event="AF"):
    gross = tables[event].loc["Gross"].copy()
    ue = gross.pop("Ue")
    gross["Episode F1"] = harmonic_f1(gross["ESe"], gross["E+P"])
    gross["Duration F1"] = harmonic_f1(gross["DSe"], gross["D+P"])
    gross["Onset F1"] = harmonic_f1(gross["Onset Se"], gross["Onset +P"])
    gross["Offset F1"] = harmonic_f1(gross["Offset Se"], gross["Offset +P"])
    gross["Boundary F1"] = (gross["Onset F1"] + gross["Offset F1"]) / 2
    gross["Boundary MAE"] = (gross["Onset MAE"] + gross["Offset MAE"]) / 2
    gross["Ue"] = ue
    return gross


def boundary_tolerance_sweep(
    true_by_record,
    pred_by_record,
    tolerances,
    threshold,
    event="AF",
    true_threshold=0.5,
):
    max_tolerance = max(tolerances)

    matched_errors = {name: [] for _, name in BOUNDARIES}
    n_ref = {name: 0 for _, name in BOUNDARIES}
    n_pred = {name: 0 for _, name in BOUNDARIES}
    ue_distances = []
    ue_denominator = 0

    for record_key in true_by_record:
        record_duration = len(true_by_record[record_key])
        ref_intervals = event_intervals(true_by_record[record_key], true_threshold)[event]
        pred_intervals = event_intervals(pred_by_record[record_key], threshold)[event]
        distances, denom = boundary_ue_distances(ref_intervals, pred_intervals, record_duration)
        if denom > 0:
            ue_distances.append(distances)
            ue_denominator += denom
        for boundary_index, name in BOUNDARIES:
            ref_times = boundary_times(ref_intervals, boundary_index, record_duration)
            pred_times = boundary_times(pred_intervals, boundary_index, record_duration)
            n_ref[name] += len(ref_times)
            n_pred[name] += len(pred_times)
            matches = boundary_matches(ref_times, pred_times, max_tolerance)
            matched_errors[name].extend(abs(error) for _, _, error in matches)

    ue_distances = np.concatenate(ue_distances) if ue_distances else np.empty(0)

    rows = {}
    for tolerance in tolerances:
        row = {}
        for _, name in BOUNDARIES:
            errors = [error for error in matched_errors[name] if error <= tolerance]
            tp = len(errors)
            fn = n_ref[name] - tp
            fp = n_pred[name] - tp
            row[f"{name} Se"] = ratio(tp, tp + fn)
            row[f"{name} +P"] = ratio(tp, tp + fp)
            row[f"{name} MAE"] = ratio(sum(errors), tp)
            row[f"{name} F1"] = f1(tp, fn, fp)
        row["Boundary F1"] = (row["Onset F1"] + row["Offset F1"]) / 2
        row["Boundary MAE"] = (row["Onset MAE"] + row["Offset MAE"]) / 2
        row["Ue"] = ratio(float((ue_distances <= tolerance).sum()), ue_denominator)
        rows[tolerance] = row

    df_sweep = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=[
        "Onset Se", "Onset +P", "Onset MAE", "Onset F1",
        "Offset Se", "Offset +P", "Offset MAE", "Offset F1",
        "Boundary F1",
        "Boundary MAE",
        "Ue",
    ])
    df_sweep.index.name = "Tolerance (s)"
    return df_sweep


def baseline_tolerance_sweep(
    true_by_record,
    prob_by_record,
    thresholds,
    merge_gaps,
    tolerances,
    min_duration_seconds=None,
    event="AF",
    true_threshold=0.5,
):
    if min_duration_seconds is None:
        min_duration_seconds = float(GLOBAL_CONFIG["postprocessing"]["min_duration_seconds"])

    rows = {}
    for threshold in thresholds:
        for merge_gap in merge_gaps:
            pred_by_record = {
                record_key: postprocess_probabilities(
                    prob_by_record[record_key], threshold, merge_gap, min_duration_seconds
                )
                for record_key in true_by_record
            }
            rows[(float(threshold), float(merge_gap))] = boundary_tolerance_sweep(
                true_by_record,
                pred_by_record,
                tolerances,
                0.5,
                event=event,
                true_threshold=true_threshold,
            )["Boundary F1"]

    table = pd.DataFrame(rows).T
    table.index = pd.MultiIndex.from_tuples(table.index, names=["threshold", "merge_gap_seconds"])
    return table


def select_baseline(
    true_by_record,
    prob_by_record,
    thresholds,
    merge_gaps,
    tolerances,
    min_duration_seconds=None,
    event="AF",
    true_threshold=0.5,
):
    curves = baseline_tolerance_sweep(
        true_by_record,
        prob_by_record,
        thresholds,
        merge_gaps,
        tolerances,
        min_duration_seconds=min_duration_seconds,
        event=event,
        true_threshold=true_threshold,
    )
    score = curves.mean(axis=1, skipna=False)
    post_threshold, post_merge_gap = score.idxmax()

    return {
        "post_threshold": float(post_threshold),
        "post_merge_gap": float(post_merge_gap),
        "score": score,
        "curves": curves,
    }


def hysteresis_labels(probs, threshold_enter, threshold_exit):
    if threshold_exit > threshold_enter:
        raise ValueError(
            f"threshold_exit {threshold_exit} must not exceed threshold_enter {threshold_enter}"
        )
    probs = np.asarray(probs, dtype=np.float64).ravel()
    index = np.arange(len(probs))
    last_enter = np.maximum.accumulate(np.where(probs >= threshold_enter, index, -1))
    last_exit = np.maximum.accumulate(np.where(probs < threshold_exit, index, -1))
    return (last_enter > last_exit).astype(np.int64)


def postprocess_hysteresis(
    probs, threshold_enter, threshold_exit, merge_gap_seconds, min_duration_seconds
):
    binary = hysteresis_labels(probs, threshold_enter, threshold_exit)
    intervals = post_process_intervals(
        labels_to_intervals(binary),
        merge_gap_seconds,
        min_duration_seconds,
        record_duration_seconds=len(binary),
    )
    return intervals_to_labels(len(binary), intervals)


def select_hysteresis_baseline(
    true_by_record,
    prob_by_record,
    thresholds_enter,
    thresholds_exit,
    merge_gaps,
    tolerances,
    min_duration_seconds=None,
    event="AF",
    true_threshold=0.5,
):
    if min_duration_seconds is None:
        min_duration_seconds = float(GLOBAL_CONFIG["postprocessing"]["min_duration_seconds"])

    rows = {}
    for threshold_enter in thresholds_enter:
        for threshold_exit in thresholds_exit:
            if threshold_exit > threshold_enter:
                continue
            for merge_gap in merge_gaps:
                pred_by_record = {
                    record_key: postprocess_hysteresis(
                        prob_by_record[record_key],
                        threshold_enter,
                        threshold_exit,
                        merge_gap,
                        min_duration_seconds,
                    )
                    for record_key in true_by_record
                }
                rows[(float(threshold_enter), float(threshold_exit), float(merge_gap))] = (
                    boundary_tolerance_sweep(
                        true_by_record,
                        pred_by_record,
                        tolerances,
                        0.5,
                        event=event,
                        true_threshold=true_threshold,
                    )["Boundary F1"]
                )
    curves = pd.DataFrame(rows).T
    curves.index = pd.MultiIndex.from_tuples(
        curves.index,
        names=["threshold_enter", "threshold_exit", "merge_gap_seconds"],
    )

    score = curves.mean(axis=1, skipna=False)
    post_threshold_enter, post_threshold_exit, post_merge_gap = score.idxmax()
    return {
        "post_threshold_enter": float(post_threshold_enter),
        "post_threshold_exit": float(post_threshold_exit),
        "post_merge_gap": float(post_merge_gap),
        "score": score,
        "curves": curves,
    }


def test_cohort_table(test_records):
    sources = list(dict.fromkeys(s for s, _ in test_records))
    rows = []
    for source in sources:
        source_records = [(s, rid) for s, rid in test_records if s == source]
        n_seconds = 0
        n_episodes = 0
        n_mixed = 0
        for s, rid in source_records:
            labels = filter_short_intervals(load_true_second_labels(s, rid))
            n_seconds += len(labels)
            n_episodes += len(labels_to_intervals(labels, min_duration_seconds=0.0))
            n_mixed += int(0 < labels.sum() < len(labels))
        rows.append(
            {
                "Source": source,
                "Records": len(source_records),
                "Hours": int(round(n_seconds / 3600.0)),
                "Retained AF episodes": n_episodes,
                "Mixed-rhythm records": n_mixed,
            }
        )
    table = pd.DataFrame(rows)
    total = table.drop(columns="Source").sum()
    return pd.concat(
        [table, pd.DataFrame([{"Source": "Total", **total.to_dict()}])],
        ignore_index=True,
    )


def overlap_groups(ref_intervals, pred_intervals):
    items = sorted(
        [(s, e, 0, i) for i, (s, e) in enumerate(ref_intervals)]
        + [(s, e, 1, j) for j, (s, e) in enumerate(pred_intervals)]
    )
    groups = []
    group_end = None
    for start, end, kind, index in items:
        if group_end is None or start >= group_end:
            groups.append(([], []))
            group_end = end
        groups[-1][kind].append(index)
        group_end = max(group_end, end)
    return groups


def formation_category(n_ref, n_pred):
    if n_ref == 0 and n_pred == 1:
        return "false"
    if n_ref == 1 and n_pred == 0:
        return "missed"
    if n_ref == 1 and n_pred == 1:
        return "one-to-one"
    if n_ref == 1 and n_pred >= 2:
        return "fragmented"
    if n_ref >= 2 and n_pred == 1:
        return "merged"
    if n_ref >= 2 and n_pred >= 2:
        return "merged_and_fragmented"
    raise ValueError(f"Invalid overlap group sizes n_ref={n_ref}, n_pred={n_pred}")


def observable_boundaries(intervals, boundary_index, record_duration):
    if boundary_index == 0:
        return [(i, float(s)) for i, (s, _) in enumerate(intervals) if s > 0]
    return [(i, float(e)) for i, (_, e) in enumerate(intervals) if e < record_duration]


def false_boundary_causes(true_by_record, pred_by_record, tolerance, threshold=0.5, event="AF"):
    categories = [
        "one-to-one", "fragmented", "merged", "merged_and_fragmented", "missed", "false",
    ]
    counts = {c: {"TP": 0, "FN": 0, "FP": 0} for c in categories}
    for record_key in true_by_record:
        duration = len(true_by_record[record_key])
        refs = event_intervals(true_by_record[record_key], threshold)[event]
        preds = event_intervals(pred_by_record[record_key], threshold)[event]
        ref_cat, pred_cat = {}, {}
        for ref_ix, pred_ix in overlap_groups(refs, preds):
            cat = formation_category(len(ref_ix), len(pred_ix))
            ref_cat.update((i, cat) for i in ref_ix)
            pred_cat.update((j, cat) for j in pred_ix)
        for boundary_index, _ in BOUNDARIES:
            ref_b = observable_boundaries(refs, boundary_index, duration)
            pred_b = observable_boundaries(preds, boundary_index, duration)
            matches = boundary_matches([t for _, t in ref_b], [t for _, t in pred_b], tolerance)
            for ri, pj, _ in matches:
                if ref_cat[ref_b[ri][0]] != pred_cat[pred_b[pj][0]]:
                    raise ValueError(f"Matched boundary crosses groups on {record_key}")
                counts[ref_cat[ref_b[ri][0]]]["TP"] += 1
            matched_refs = {ri for ri, _, _ in matches}
            matched_preds = {pj for _, pj, _ in matches}
            for ri, (i, _) in enumerate(ref_b):
                if ri not in matched_refs:
                    counts[ref_cat[i]]["FN"] += 1
            for pj, (j, _) in enumerate(pred_b):
                if pj not in matched_preds:
                    counts[pred_cat[j]]["FP"] += 1
    return pd.DataFrame.from_dict(counts, orient="index").reindex(categories)


def boundaries_agree(ref_interval, pred_interval, boundary_index, record_duration, tolerance):
    edge = 0 if boundary_index == 0 else record_duration
    ref_time, pred_time = ref_interval[boundary_index], pred_interval[boundary_index]
    if ref_time == edge or pred_time == edge:
        return ref_time == pred_time
    return abs(pred_time - ref_time) <= tolerance


def episode_level_metrics(true_by_record, pred_by_record, tolerances, threshold=0.5, event="AF"):
    n_ref = n_pred = n_one_to_one = 0
    exact_tp = {tol: 0 for tol in tolerances}
    count_errors = []
    for record_key in true_by_record:
        duration = len(true_by_record[record_key])
        refs = event_intervals(true_by_record[record_key], threshold)[event]
        preds = event_intervals(pred_by_record[record_key], threshold)[event]
        n_ref += len(refs)
        n_pred += len(preds)
        count_errors.append(abs(len(preds) - len(refs)))
        for ref_ix, pred_ix in overlap_groups(refs, preds):
            if formation_category(len(ref_ix), len(pred_ix)) != "one-to-one":
                continue
            n_one_to_one += 1
            ref, pred = refs[ref_ix[0]], preds[pred_ix[0]]
            for tol in tolerances:
                if all(
                    boundaries_agree(ref, pred, boundary_index, duration, tol)
                    for boundary_index, _ in BOUNDARIES
                ):
                    exact_tp[tol] += 1
    exact_f1 = {
        tol: f1(exact_tp[tol], n_ref - exact_tp[tol], n_pred - exact_tp[tol])
        for tol in tolerances
    }
    return pd.Series({
        "one_to_one_episode_f1": float(
            harmonic_f1(ratio(n_one_to_one, n_ref), ratio(n_one_to_one, n_pred))
        ),
        "exact_episode_f1_delta5": float(exact_f1[5]),
        "exact_episode_f1_mean": float(np.mean([exact_f1[tol] for tol in tolerances])),
        "mean_episode_count_error": float(np.mean(count_errors)),
    })

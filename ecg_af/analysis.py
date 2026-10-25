import torch
import numpy as np

from .data.beat import load_beat_index
from .util import GLOBAL_CONFIG


@torch.no_grad()
def infer_per_beat(model, loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    record_keys = sorted({(s, r) for s, r, _ in loader.dataset.index})
    n_beats = {k: 0 for k in record_keys}
    for s, r, _ in loader.dataset.index:
        n_beats[(s, r)] += 1

    target_fs = GLOBAL_CONFIG["data"]["preprocessing"]["target_fs"]
    record_probs = {k: np.zeros(n_beats[k], dtype=np.float32) for k in record_keys}
    record_labels = {k: np.zeros(n_beats[k], dtype=np.float32) for k in record_keys}

    model.eval()
    for signals, labels, idxs in loader:
        signals = signals.to(device=device, dtype=torch.float32)
        logits = model(signals)
        probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
        labels_np = labels.detach().cpu().numpy().astype(np.float32)
        idxs_list = idxs.tolist() if hasattr(idxs, "tolist") else list(idxs)
        for i, idx in enumerate(idxs_list):
            source, record_id, beat_idx = loader.dataset.index[int(idx)]
            record_probs[(source, record_id)][int(beat_idx)] = probs[i]
            record_labels[(source, record_id)][int(beat_idx)] = labels_np[i]

    record_times = {}
    for source, record_id in record_keys:
        _, windowed_r_peaks = load_beat_index(source, record_id)
        record_times[(source, record_id)] = windowed_r_peaks.astype(np.float64) / target_fs
    return record_labels, record_probs, record_times


@torch.no_grad()
def infer_per_segment(model, loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seg_cfg = GLOBAL_CONFIG["modes"]["segment"]
    window_seconds = int(round(float(seg_cfg["window_seconds"])))
    step_seconds = int(round(float(loader.dataset.step_seconds)))

    max_segment_by_record = {}
    for source, record_id, segment_idx in loader.dataset.index:
        record_key = (source, record_id)
        prev = max_segment_by_record.get(record_key, -1)
        if int(segment_idx) > prev:
            max_segment_by_record[record_key] = int(segment_idx)

    record_prob_sum = {}
    record_label_sum = {}
    record_counts = {}
    for record_key, last_idx in max_segment_by_record.items():
        total_seconds = last_idx * step_seconds + window_seconds
        record_prob_sum[record_key] = np.zeros(total_seconds, dtype=np.float32)
        record_label_sum[record_key] = np.zeros(total_seconds, dtype=np.float32)
        record_counts[record_key] = np.zeros(total_seconds, dtype=np.float32)

    model.eval()
    for signals, labels, idxs in loader:
        signals = signals.to(device=device, dtype=torch.float32)
        logits = model(signals)
        probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
        labels_np = labels.detach().cpu().numpy().astype(np.float32)
        idxs_list = idxs.tolist() if hasattr(idxs, "tolist") else list(idxs)
        for i, idx in enumerate(idxs_list):
            source, record_id, segment_idx = loader.dataset.index[int(idx)]
            record_key = (source, record_id)
            start = int(segment_idx) * step_seconds
            end = start + window_seconds
            record_prob_sum[record_key][start:end] += probs[i]
            record_label_sum[record_key][start:end] += labels_np[i]
            record_counts[record_key][start:end] += 1.0

    record_probs = {}
    record_labels = {}
    record_times = {}
    for record_key in max_segment_by_record:
        counts = record_counts[record_key]
        valid = counts > 0
        record_probs[record_key] = (record_prob_sum[record_key][valid] / counts[valid]).astype(np.float32)
        record_labels[record_key] = (record_label_sum[record_key][valid] / counts[valid]).astype(np.float32)
        record_times[record_key] = np.flatnonzero(valid).astype(np.float64)
    return record_labels, record_probs, record_times


@torch.no_grad()
def infer_per_noise(model, loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = GLOBAL_CONFIG["modes"]["noise"]
    window_seconds = int(round(float(cfg["window_seconds"])))
    step_seconds = int(round(float(cfg["step_seconds"])))

    max_segment_by_record = {}
    for source, record_id, segment_idx in loader.dataset.index:
        record_key = (source, record_id)
        max_segment_by_record[record_key] = max(
            max_segment_by_record.get(record_key, -1), int(segment_idx)
        )

    record_prob_sum = {}
    record_counts = {}
    for record_key, last_idx in max_segment_by_record.items():
        total_seconds = last_idx * step_seconds + window_seconds
        record_prob_sum[record_key] = np.zeros(total_seconds, dtype=np.float32)
        record_counts[record_key] = np.zeros(total_seconds, dtype=np.float32)

    model.eval()
    for signals, _, idxs in loader:
        signals = signals.to(device=device, dtype=torch.float32)
        probs = torch.sigmoid(model(signals)).detach().cpu().numpy().astype(np.float32)
        idxs_list = idxs.tolist() if hasattr(idxs, "tolist") else list(idxs)
        for i, idx in enumerate(idxs_list):
            source, record_id, segment_idx = loader.dataset.index[int(idx)]
            record_key = (source, record_id)
            start = int(segment_idx) * step_seconds
            end = start + window_seconds
            record_prob_sum[record_key][start:end] += probs[i]
            record_counts[record_key][start:end] += 1.0

    record_probs = {}
    record_times = {}
    for record_key in max_segment_by_record:
        counts = record_counts[record_key]
        valid = counts > 0
        record_probs[record_key] = (
            record_prob_sum[record_key][valid] / counts[valid]
        ).astype(np.float32)
        record_times[record_key] = np.flatnonzero(valid).astype(np.float64)
    return {}, record_probs, record_times


def infer_record_probs(model, loader, mode):
    if mode == "beat":
        return infer_per_beat(model, loader)
    elif mode == "segment":
        return infer_per_segment(model, loader)
    elif mode == "noise":
        return infer_per_noise(model, loader)
    else:
        raise ValueError(f"Invalid mode: {mode}")


def k_fold_inference(k_fold_models, loader, mode):
    all_records_probs = {}
    all_records_labels = {}
    all_records_times = {}
    tmp_count = {}
    for model in k_fold_models:
        records_labels, records_probs, records_times = infer_record_probs(model, loader, mode)
        for record_key in records_labels:
            if record_key not in all_records_labels:
                all_records_labels[record_key] = records_labels[record_key]
                all_records_probs[record_key] = records_probs[record_key]
                all_records_times[record_key] = records_times[record_key]
                tmp_count[record_key] = 1
            else:
                all_records_labels[record_key] += records_labels[record_key]
                all_records_probs[record_key] += records_probs[record_key]
                tmp_count[record_key] += 1
    for record_key in all_records_labels:
        all_records_labels[record_key] /= tmp_count[record_key]
        all_records_probs[record_key] /= tmp_count[record_key]
    return all_records_labels, all_records_probs, all_records_times


def get_metrics(true_labels, pred_labels):
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    true_bin = true_labels >= 0.5
    pred_bin = pred_labels >= 0.5

    true_total = np.sum(true_bin)
    pred_total = np.sum(pred_bin)
    overlap = np.sum(true_bin & pred_bin)

    fn_dur = true_total - overlap
    fp_dur = pred_total - overlap
    tp_dur = overlap
    ppv = tp_dur / pred_total if pred_total > 0 else 0
    if true_total == 0:
        sensitivity = np.nan
        f1 = int(pred_total == 0)
    else:
        sensitivity = tp_dur / true_total
        f1 = 2 * sensitivity * ppv / (sensitivity + ppv) if (sensitivity + ppv) > 0 else 0

    return {
        'TP_dur': tp_dur,
        'FP_dur': fp_dur,
        'FN_dur': fn_dur,
        'sensitivity': sensitivity,
        'ppv': ppv,
        'f1': f1,
    }


def get_duration_metrics(true_records_labels, pred_records_probs, threshold=0.5):
    pred_records_labels = {k: (v >= threshold).astype(np.int8) for k, v in pred_records_probs.items()}
    metrics = {}
    for record_key in true_records_labels:
        true_labels = true_records_labels[record_key]
        pred_labels = pred_records_labels[record_key]
        metrics["_".join(record_key)] = get_metrics(true_labels, pred_labels)

    metrics["total"] = get_metrics(
        np.concatenate(list(true_records_labels.values())), 
        np.concatenate(list(pred_records_labels.values()))
    )
    return metrics

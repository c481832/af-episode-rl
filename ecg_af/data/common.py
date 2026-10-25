from functools import lru_cache
from pathlib import Path

import numpy as np
import wfdb

from .iridia import (
    CALIBRATION_SECONDS,
    IRIDIA_FS,
    build_iridia_labels,
    build_iridia_r_peaks,
    list_iridia_records,
    load_iridia_record,
)
from .preprocessing import (
    detect_r_peaks,
    preprocess_signal,
    resample_signal,
)
from ..util import GLOBAL_CONFIG


def source_db_dir(source):
    db_dir = Path(GLOBAL_CONFIG["data"]["sources"][source]["path"])
    if db_dir.is_absolute():
        return db_dir
    return Path(__file__).parent.parent.joinpath(db_dir).resolve()


def flatten_records(record_dict):
    all_records = []
    for source, source_records in record_dict.items():
        all_records.extend((source, record_id) for record_id in source_records)
    return all_records


def split_records(record_dict):
    split_cfg = GLOBAL_CONFIG["data"]["split"]["layer1"]
    test_size = split_cfg["test_size"]
    val_size = split_cfg["val_size"]
    n_splits = GLOBAL_CONFIG["data"]["k_folds"]["n_splits"]

    records = flatten_records(record_dict)
    n_records = len(records)
    override = GLOBAL_CONFIG["data"]["override_test"]

    if len(override) > 0:
        # Test set is fully defined by override sources. Train/val are split only
        # over the remaining pool, so adding test-only sources never disturbs them.
        test_records = [r for r in records if r[0] in override]
        pool_records = [r for r in records if r[0] not in override]
    else:
        shuffled_idx = GLOBAL_CONFIG.rng.permutation(n_records)
        shuffled_records = [records[i] for i in shuffled_idx]
        n_test = int(n_records * test_size)
        test_records = shuffled_records[:n_test]
        pool_records = shuffled_records[n_test:]
    pool_size = len(pool_records)
    n_val = int(pool_size * val_size)

    pool_perm = GLOBAL_CONFIG.rng.permutation(pool_size)
    if n_splits == 1:
        val_groups = [pool_perm[:n_val]]
    else:
        val_groups = np.array_split(pool_perm[:n_val], n_splits)

    folds = []
    for i in range(n_splits):
        val_idx = np.asarray(val_groups[i], dtype=np.int64)
        train_idx = np.setdiff1d(np.arange(pool_size), val_idx)
        val_records = [pool_records[int(j)] for j in val_idx]
        train_records = [pool_records[int(j)] for j in train_idx]
        folds.append({"train": train_records, "val": val_records, "test": test_records})
    return folds


def partition_records(records, ratios):
    n = len(records)
    perm = GLOBAL_CONFIG.rng.permutation(n)
    shuffled = [records[int(i)] for i in perm]
    chunks = []
    start = 0
    for ratio in ratios[:-1]:
        end = start + int(n * ratio)
        chunks.append(shuffled[start:end])
        start = end
    chunks.append(shuffled[start:])
    return chunks


def split_records_two_layer(record_dict):
    layer2_cfg = GLOBAL_CONFIG["data"]["split"]["layer2"]
    train_sources = layer2_cfg["train_sources"]
    train_size = 1.0 - layer2_cfg["test_size"] - layer2_cfg["val_size"]
    ratios = (
        train_size,
        layer2_cfg["val_size"],
        layer2_cfg["test_size"],
    )
    folds = split_records(record_dict)
    two_layer_folds = []
    for fold in folds:
        test_records = list(fold["test"])
        if train_sources:
            pool = [r for r in test_records if r[0] in train_sources]
            held_out = [r for r in test_records if r[0] not in train_sources]
            l2_train, l2_val, l2_test = partition_records(pool, ratios)
            l2_test = l2_test + held_out
        else:
            l2_train, l2_val, l2_test = [], [], test_records
        two_layer_folds.append(
            {
                "layer1": {
                    "train": fold["train"],
                    "val": fold["val"],
                    "test": fold["test"],
                },
                "layer2": {"train": l2_train, "val": l2_val, "test": l2_test},
            }
        )
    return two_layer_folds


def build_folds():
    folds = []
    for fold in split_records_two_layer(load_records_list()):
        layer1 = fold["layer1"]
        folds.append(
            {
                "train": layer1["train"],
                "val": layer1["val"],
                "test": layer1["test"],
            }
        )
    return folds


def load_records_list():
    records = {}
    for source, source_cfg in GLOBAL_CONFIG["data"]["sources"].items():
        db_dir = source_db_dir(source)
        excluded = set(source_cfg["excluded_records"])

        if source == "iridia":
            ids = list_iridia_records(db_dir)
        elif source_cfg["list_hea_files"]:
            ids = sorted(p.stem for p in db_dir.glob("*.hea"))
        else:
            with open(db_dir.joinpath("RECORDS")) as f:
                ids = [line.strip() for line in f if line.strip()]

        records[source] = [r for r in ids if r not in excluded]
    return records


def select_signal_channel(record):
    preferred = GLOBAL_CONFIG["data"]["preferred_leads"]
    sig_names = list(record.sig_name)
    candidates = [(i, n) for i, n in enumerate(sig_names) if record.p_signal[:, i].any()]
    for lead in preferred:
        for i, n in candidates:
            if n == lead:
                return i


def clean_aux_note(note, rhythms_to_remove):
    cleaned_note = note.replace("\x00", "").strip()
    if rhythms_to_remove is not None:
        for rhythm in rhythms_to_remove:
            cleaned_note = cleaned_note.replace(rhythm, "")
    return cleaned_note


def build_labels_token(ann, raw_len, target_rhythms, strip_rhythms):
    sample_labels = np.zeros(raw_len, dtype=np.int8)
    rhythm_events = [
        (int(s), clean_aux_note(n, rhythms_to_remove=strip_rhythms))
        for s, n in zip(ann.sample, ann.aux_note)
        if clean_aux_note(n, rhythms_to_remove=strip_rhythms).startswith("(")
    ]
    for i, (idx, rhythm) in enumerate(rhythm_events):
        end_idx = rhythm_events[i + 1][0] if i + 1 < len(rhythm_events) else raw_len
        if rhythm in target_rhythms:
            sample_labels[idx:end_idx] = 1
    return sample_labels


def build_labels_start_end(ann, raw_len, target_rhythms):
    target_set = set(target_rhythms)
    sample_labels = np.zeros(raw_len, dtype=np.int8)
    active = {}  # rhythm name -> start sample
    for sample, aux in zip(ann.sample, ann.aux_note):
        if not aux:
            continue
        for token in aux.replace("\x00", "").strip().lstrip("(").split("|"):
            token = token.lstrip("(")
            if token.endswith("_start"):
                name = token[: -len("_start")]
                if name in target_set:
                    active[name] = int(sample)
            elif token.endswith("_end"):
                name = token[: -len("_end")]
                start = active.pop(name, None)
                if start is not None:
                    sample_labels[start : int(sample)] = 1
    for start in active.values():
        sample_labels[start:] = 1
    return sample_labels


def build_labels(ann, raw_len, schema, target_rhythms, strip_rhythms):
    if schema == "start_end":
        return build_labels_start_end(ann, raw_len, target_rhythms)
    return build_labels_token(ann, raw_len, target_rhythms, strip_rhythms)


def build_noise_filter_mask(raw_signal):
    noise_filter_threshold = GLOBAL_CONFIG["data"]["preprocessing"]["noise_filter_threshold"]
    if noise_filter_threshold is None:
        return np.ones_like(raw_signal, dtype=bool)
    zero_mask = raw_signal == 0
    noise_filter_mask = np.zeros_like(zero_mask, dtype=bool)

    padded_zero_mask = np.concatenate(([False], zero_mask, [False]))
    zero_changes = np.diff(padded_zero_mask.astype(np.int8))
    zero_starts = np.flatnonzero(zero_changes == 1)
    zero_ends = np.flatnonzero(zero_changes == -1)

    for start, end in zip(zero_starts, zero_ends):
        if end - start > noise_filter_threshold:
            noise_filter_mask[start:end] = True

    return ~noise_filter_mask


def build_beat_filter_mask(rr_intervals_seconds, n_beats):
    max_rr_seconds = GLOBAL_CONFIG["data"]["r_peaks"]["max_rr_seconds"]
    beat_filter_mask = np.ones(n_beats, dtype=bool)
    invalid_rr = rr_intervals_seconds > max_rr_seconds
    invalid_idx = np.flatnonzero(invalid_rr)
    beat_filter_mask[invalid_idx] = False
    beat_filter_mask[invalid_idx + 1] = False
    return beat_filter_mask


def load_record(source, record_id):
    db_dir = source_db_dir(source)
    return wfdb.rdrecord(str(db_dir.joinpath(record_id)))


def load_annotation(source, record_id, ann_ext="atr"):
    db_dir = source_db_dir(source)
    return wfdb.rdann(str(db_dir.joinpath(record_id)), ann_ext)


def annotation_file_exists(source, record_id, ann_ext):
    return source_db_dir(source).joinpath(f"{record_id}.{ann_ext}").exists()


def experiment_run_dir():
    raw = GLOBAL_CONFIG["train"]["checkpoint_dir"]
    return (Path(__file__).parent.parent / raw / GLOBAL_CONFIG.name).resolve()


def processed_record_path(source, record_id):
    return experiment_run_dir() / "processed_records" / f"{source}|{record_id}.npz"


@lru_cache(maxsize=1)
def load_processed_record(source, record_id):
    if GLOBAL_CONFIG.debug_print:
        print(f"Loading processed record {source} {record_id}", end="\r")

    path = processed_record_path(source, record_id)
    if path.exists():
        with np.load(path, allow_pickle=False) as raw:
            processed_signal = np.asarray(raw["processed_signal"], dtype=np.float32)
            resampled_labels = np.asarray(raw["resampled_labels"], dtype=np.int8)
            processed_r_peaks = np.asarray(raw["processed_r_peaks"], dtype=np.int64)
            return processed_signal, resampled_labels, processed_r_peaks

    target_fs = GLOBAL_CONFIG["data"]["preprocessing"]["target_fs"]

    if source == "iridia":
        rec_dir = source_db_dir(source).joinpath(record_id)
        record = load_iridia_record(rec_dir, record_id)
        fs = int(round(record.fs))
        raw_signal = record.p_signal[:, select_signal_channel(record)]
        raw_labels = build_iridia_labels(rec_dir, record_id, len(raw_signal))
        start_idx = int(CALIBRATION_SECONDS * IRIDIA_FS)
        raw_r_peaks = build_iridia_r_peaks(rec_dir, record_id, len(raw_signal))
    else:
        schema = GLOBAL_CONFIG["data"]["sources"][source]["rhythm_schema"]
        rhythms_cfg = GLOBAL_CONFIG["data"]["rhythms"]
        target_rhythms = (
            rhythms_cfg["target_start_end"] if schema == "start_end" else rhythms_cfg["target"]
        )
        strip_rhythms = rhythms_cfg["strip_rhythms"]

        record = load_record(source, record_id)
        ann = load_annotation(source, record_id)
        fs = int(round(record.fs))
        raw_signal = record.p_signal[:, select_signal_channel(record)]
        raw_labels = build_labels(ann, len(raw_signal), schema, target_rhythms, strip_rhythms)
        start_idx = int(ann.sample[0])

    resampled_signal = resample_signal(raw_signal[start_idx:], orig_fs=fs, target_fs=target_fs).astype(np.float32)
    resampled_labels = (resample_signal(raw_labels[start_idx:], orig_fs=fs, target_fs=target_fs) > 0).astype(np.int8)

    noise_filter_mask = build_noise_filter_mask(resampled_signal)
    resampled_to_processed = np.cumsum(noise_filter_mask) - 1
    resampled_signal = resampled_signal[noise_filter_mask]
    resampled_labels = resampled_labels[noise_filter_mask]
    processed_signal = preprocess_signal(resampled_signal)

    if source == "iridia":
        resampled_r_peaks = np.rint((raw_r_peaks - start_idx) * target_fs / fs).astype(np.int64)
        valid_peak_mask = (resampled_r_peaks >= 0) & (resampled_r_peaks < len(noise_filter_mask))
        valid_peak_mask[valid_peak_mask] = noise_filter_mask[resampled_r_peaks[valid_peak_mask]]
        processed_r_peaks = resampled_to_processed[resampled_r_peaks[valid_peak_mask]]
        processed_r_peaks = np.unique(processed_r_peaks)
        if len(processed_r_peaks) < 2:
            min_rr_seconds = GLOBAL_CONFIG["data"]["r_peaks"]["min_rr_seconds"]
            wavelet = GLOBAL_CONFIG["data"]["r_peaks"]["wavelet"]
            level = GLOBAL_CONFIG["data"]["r_peaks"]["level"]
            processed_r_peaks = detect_r_peaks(
                processed_signal, target_fs, min_rr_seconds, wavelet, level
            )
    elif annotation_file_exists(source, record_id, "qrs"):
        qrs = load_annotation(source, record_id, ann_ext="qrs")
        resampled_r_peaks = np.rint((qrs.sample - start_idx) * target_fs / fs).astype(np.int64)
        valid_peak_mask = (resampled_r_peaks >= 0) & (resampled_r_peaks < len(noise_filter_mask))
        valid_peak_mask[valid_peak_mask] = noise_filter_mask[resampled_r_peaks[valid_peak_mask]]
        processed_r_peaks = resampled_to_processed[resampled_r_peaks[valid_peak_mask]]
        processed_r_peaks = np.unique(processed_r_peaks)
    else:
        min_rr_seconds = GLOBAL_CONFIG["data"]["r_peaks"]["min_rr_seconds"]
        wavelet = GLOBAL_CONFIG["data"]["r_peaks"]["wavelet"]
        level = GLOBAL_CONFIG["data"]["r_peaks"]["level"]
        processed_r_peaks = detect_r_peaks(processed_signal, target_fs, min_rr_seconds, wavelet, level)

    rr_intervals_seconds = np.diff(processed_r_peaks) / target_fs
    beat_filter_mask = build_beat_filter_mask(rr_intervals_seconds, len(processed_r_peaks))
    processed_r_peaks = processed_r_peaks[beat_filter_mask].astype(np.int64)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        processed_signal=processed_signal,
        resampled_labels=resampled_labels,
        processed_r_peaks=processed_r_peaks,
    )
    return processed_signal, resampled_labels, processed_r_peaks


def get_record_info(records):
    target_fs = GLOBAL_CONFIG["data"]["preprocessing"]["target_fs"]
    per_record = []
    for source, record_id in records:
        _, resampled_labels, _ = load_processed_record(source, record_id)

        n_samples = int(len(resampled_labels))
        af_samples = int(resampled_labels.sum())
        af_ratio_samples = float(af_samples / n_samples) if n_samples else 0.0
        duration_seconds = float(n_samples / target_fs) if target_fs else 0.0

        per_record.append(
            {
                "source": source,
                "record_id": record_id,
                "n_samples": n_samples,
                "af_samples": af_samples,
                "af_ratio_samples": af_ratio_samples,
                "duration_seconds": duration_seconds,
            }
        )

    return per_record


def log_binary_labels(tag, y):
    y = np.asarray(y).ravel()
    if y.size == 0:
        GLOBAL_CONFIG.logger.info("labels %s: empty", tag)
    else:
        GLOBAL_CONFIG.logger.info("labels %s: n=%d pos_frac=%.4f", tag, y.size, float(y.mean()))



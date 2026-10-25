from pathlib import Path

import h5py
import numpy as np
import pandas as pd

CALIBRATION_SECONDS = 30
IRIDIA_FS = 200
SIG_NAMES = ["I", "II"]


class IridiaRecord:
    def __init__(self, p_signal, fs, sig_name):
        self.p_signal = p_signal
        self.fs = fs
        self.sig_name = sig_name


def list_iridia_records(db_dir):
    return sorted(
        p.name
        for p in Path(db_dir).iterdir()
        if p.is_dir() and p.name.startswith("record_")
    )


def concat_h5(h5_paths, dataset_name):
    parts = []
    for path in h5_paths:
        with h5py.File(path, "r") as f:
            parts.append(np.asarray(f[dataset_name][:]))
    return np.concatenate(parts, axis=0)


def file_offsets(h5_paths, dataset_name):
    offsets = [0]
    for path in h5_paths[:-1]:
        with h5py.File(path, "r") as f:
            offsets.append(offsets[-1] + f[dataset_name].shape[0])
    return offsets


def load_iridia_record(rec_dir, record_id):
    rec_dir = Path(rec_dir)
    ecg_paths = sorted(rec_dir.glob(f"{record_id}_ecg_*.h5"))
    if not ecg_paths:
        raise FileNotFoundError(f"No ECG HDF5 files for {record_id} in {rec_dir}")
    p_signal = concat_h5(ecg_paths, "ecg")
    return IridiaRecord(p_signal, IRIDIA_FS, SIG_NAMES)


def build_iridia_labels(rec_dir, record_id, raw_len):
    rec_dir = Path(rec_dir)
    ecg_paths = sorted(rec_dir.glob(f"{record_id}_ecg_*.h5"))
    offsets = file_offsets(ecg_paths, "ecg")
    labels_path = rec_dir.joinpath(f"{record_id}_ecg_labels.csv")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing ECG labels for {record_id}: {labels_path}")

    sample_labels = np.zeros(raw_len, dtype=np.int8)
    episodes = pd.read_csv(labels_path)
    for _, row in episodes.iterrows():
        start = offsets[int(row["start_file_index"])] + int(row["start_qrs_index"])
        end = offsets[int(row["end_file_index"])] + int(row["end_qrs_index"])
        sample_labels[start:end] = 1
    return sample_labels


def build_iridia_r_peaks(rec_dir, record_id, n_samples):
    rec_dir = Path(rec_dir)
    rr_paths = sorted(rec_dir.glob(f"{record_id}_rr_*.h5"))
    if not rr_paths:
        raise FileNotFoundError(f"No RR HDF5 files for {record_id} in {rec_dir}")

    rr_ms = concat_h5(rr_paths, "rr")
    calibration_samples = CALIBRATION_SECONDS * IRIDIA_FS
    peak_samples = np.rint(np.cumsum(rr_ms) / 1000.0 * IRIDIA_FS).astype(np.int64)
    peak_samples = peak_samples[peak_samples >= calibration_samples]
    peak_samples = peak_samples[peak_samples < n_samples]
    return np.unique(peak_samples)

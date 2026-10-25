import numpy as np

from ..util import GLOBAL_CONFIG


def process_heart_rate(processed_labels, processed_r_peaks):
    target_fs = GLOBAL_CONFIG["data"]["preprocessing"]["target_fs"]
    peak_labels = processed_labels[processed_r_peaks][1:]
    peak_rr_intervals = np.diff(processed_r_peaks) / target_fs
    beat_times = processed_r_peaks[1:].astype(np.float64) / target_fs
    return beat_times, peak_rr_intervals, peak_labels


def heart_rate_per_second(beat_times, peak_rr_intervals, n_seconds, carry_seconds):
    hr_cfg = GLOBAL_CONFIG["rl"]["observation_fields"]["heart_rate"]
    hr_lo = float(hr_cfg["hr_lo"])
    hr_hi = float(hr_cfg["hr_hi"])
    hr = np.clip(60.0 / peak_rr_intervals, hr_lo, hr_hi).astype(np.float32)
    hr = (hr - hr_lo) / (hr_hi - hr_lo)
    second_axis = np.arange(n_seconds, dtype=np.float64)
    right = np.searchsorted(beat_times, second_axis, side="right") - 1
    aligned = np.zeros(n_seconds, dtype=np.float32)
    valid = right >= 0
    aligned[valid] = hr[right[valid]]
    elapsed = np.full(n_seconds, np.inf, dtype=np.float64)
    elapsed[valid] = second_axis[valid] - beat_times[right[valid]]
    fresh = valid & (elapsed <= carry_seconds)
    aligned[~fresh] = 0.0
    return aligned

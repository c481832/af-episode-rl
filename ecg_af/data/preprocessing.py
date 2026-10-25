from math import gcd

import numpy as np
import pywt
from scipy.signal import (
    butter,
    filtfilt,
    find_peaks,
    iirnotch,
    medfilt,
    resample_poly,
    sosfiltfilt,
)

from ..util import GLOBAL_CONFIG


def preprocess_signal(signal):
    cfg = GLOBAL_CONFIG["data"]["preprocessing"]
    target_fs = cfg["target_fs"]

    if cfg["baseline_median_seconds"] is not None:
        signal = baseline_median_subtract(signal, fs=target_fs, kernel_seconds=cfg["baseline_median_seconds"])
    if cfg["notch"] is not None:
        signal = notch_filter(signal, fs=target_fs, **cfg["notch"])
    if cfg["bandpass"] is not None:
        signal = bandpass_filter(signal, fs=target_fs, **cfg["bandpass"])
    if cfg["dwt_denoise"] is not None:
        signal = dwt_denoise_signal(signal, **cfg["dwt_denoise"])
    if cfg["zscore"]:
        signal = normalize_signal(signal)

    return signal


def resample_signal(signal, orig_fs, target_fs):
    if target_fs == orig_fs:
        return signal
    scale = gcd(orig_fs, target_fs)
    return resample_poly(signal, up=target_fs // scale, down=orig_fs // scale)


def bandpass_filter(signal, lowcut, highcut, fs, order):
    sos = butter(order, [lowcut, highcut], btype="bandpass", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, signal, axis=-1)
    return filtered


def notch_filter(signal, freq, q, fs):
    b, a = iirnotch(freq, q, fs)
    return filtfilt(b, a, signal, axis=-1)


def baseline_median_subtract(signal, fs, kernel_seconds):
    kernel_size = int(round(kernel_seconds * fs))
    if kernel_size % 2 == 0:
        kernel_size += 1
    baseline = medfilt(signal, kernel_size=kernel_size)
    return signal - baseline


def dwt_denoise_signal(signal, wavelet, level):
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)

    d1 = coeffs[-1]
    sigma = np.median(np.abs(d1)) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))

    coeffs_clean = [coeffs[0]]
    for detail in coeffs[1:]:
        coeffs_clean.append(pywt.threshold(detail, threshold, mode="soft"))

    return pywt.waverec(coeffs_clean, wavelet=wavelet)[:len(signal)]


def detect_r_peaks(signal, fs, min_rr_seconds, wavelet, level):
    coeffs_detect = pywt.wavedec(signal, wavelet=wavelet, level=level)

    coeffs_d4 = [np.zeros_like(c) for c in coeffs_detect]
    coeffs_d4[1] = coeffs_detect[1]
    d4 = pywt.waverec(coeffs_d4, wavelet=wavelet)[:len(signal)]

    coeffs_d3 = [np.zeros_like(c) for c in coeffs_detect]
    coeffs_d3[2] = coeffs_detect[2]
    d3 = pywt.waverec(coeffs_d3, wavelet=wavelet)[:len(signal)]

    qrs_enhanced = d3 + d4
    qrs_energy = qrs_enhanced ** 2

    min_distance = int(min_rr_seconds * fs)
    prominence = np.std(qrs_energy)
    r_peaks, _ = find_peaks(
        qrs_energy,
        distance=min_distance,
        prominence=prominence,
    )

    return r_peaks


def normalize_signal(signal, eps=1e-8):
    """Z-score normalise *signal* along the last axis."""
    mean = signal.mean(axis=-1, keepdims=True)
    std = signal.std(axis=-1, keepdims=True)
    return (signal - mean) / np.maximum(std, eps)

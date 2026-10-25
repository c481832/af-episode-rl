import numpy as np
import matplotlib.pyplot as plt

from .util import GLOBAL_CONFIG
from .data.common import load_processed_record

plt.style.use("seaborn-v0_8")


def plot_record_predictions(
    source,
    record_id,
    true_records_labels,
    pred_records_probs,
    pred_records_times,
    mode,
    start_seconds,
    end_seconds,
):
    true = np.asarray(true_records_labels[(source, record_id)], dtype=np.float64)
    pred = np.asarray(pred_records_probs[(source, record_id)], dtype=np.float64)

    times = np.asarray(pred_records_times[(source, record_id)], dtype=np.float64)
    duration = float(times[-1]) if len(times) > 0 else 0.0
    start = float(start_seconds)
    end = duration if end_seconds is None else float(end_seconds)

    mask = (times >= start) & (times <= end)
    times_slice = times[mask]
    pred_slice = pred[mask]
    true_slice = true[mask]

    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.plot(times_slice, pred_slice, linewidth=1.2, color="#1f77b4", label="Predicted prob")
    ax.step(times_slice, true_slice, where="post", linewidth=1.8, color="#2ca02c", label="True label")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="Threshold 0.5")
    ax.set_title(f"{source}/{record_id} | predictions ({mode}) | {start:.0f}s - {end:.0f}s")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Score")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(start, end)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    return fig


def plot_record(source, record_id, start_seconds, end_seconds):
    target_fs = GLOBAL_CONFIG["data"]["preprocessing"]["target_fs"]
    processed_signal, sample_labels, processed_r_peaks = load_processed_record(source, record_id)

    duration = len(processed_signal) / target_fs
    start = float(start_seconds)
    end = duration if end_seconds is None else float(end_seconds)

    start_sample = int(round(start * target_fs))
    end_sample = min(int(round(end * target_fs)), len(processed_signal))
    signal_t = np.arange(start_sample, end_sample) / target_fs
    signal_slice = processed_signal[start_sample:end_sample]
    label_slice = sample_labels[start_sample:end_sample]

    beat_t = processed_r_peaks / target_fs
    beat_mask = (beat_t >= start) & (beat_t <= end)
    selected_beat_t = beat_t[beat_mask]
    selected_peaks = processed_r_peaks[beat_mask]
    selected_peak_y = processed_signal[selected_peaks]
    beat_labels = sample_labels[selected_peaks]

    rr_seconds = np.diff(processed_r_peaks) / target_fs
    valid_rr = rr_seconds > 0
    hr = 60 / rr_seconds[valid_rr]
    hr_t = beat_t[1:][valid_rr]
    hr_mask = (hr_t >= start) & (hr_t <= end)

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

    axes[0].plot(signal_t, signal_slice, linewidth=0.8, color="#1f77b4", label="signal")
    axes[0].scatter(selected_beat_t, selected_peak_y, s=16, color="red", marker="o", label="R peaks", zorder=3)
    axes[0].set_title(f"{source}/{record_id} | {start:.0f}s - {end:.0f}s")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.25)

    axes[1].scatter(hr_t[hr_mask], hr[hr_mask], s=16, label="heart rate", color="#1f77b4")
    axes[1].fill_between(
        selected_beat_t,
        0,
        1,
        where=beat_labels > 0,
        color="red",
        alpha=0.08,
        transform=axes[1].get_xaxis_transform(),
    )
    axes[1].set_ylabel("HR (bpm)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.25)

    axes[2].step(signal_t, label_slice, where="post", linewidth=1.8, color="#2ca02c")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Label")
    axes[2].set_yticks([0, 1], ["Non-target", "Target"])
    axes[2].set_xlim(start, end)
    axes[2].grid(True, alpha=0.25)

    plt.tight_layout()
    return fig





def plot_flat_profile_response(response, detector_label):
    probabilities = response["probabilities"]
    curves = response["curves"]
    enter, remain = response["enter"], response["remain"]

    blue, orange = "#0072B2", "#D55E00"
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    if enter is not None and remain is not None:
        left, right = sorted((enter, remain))
        ax.axvspan(left, right, color="#EAF2F8", alpha=0.9, zorder=0)
        ax.text(
            (left + right) / 2,
            0.08,
            "State-dependent\ndecision region",
            ha="center",
            va="center",
            fontsize=8,
            color="#36566E",
        )

    ax.plot(probabilities, curves[0], color=blue, linewidth=2.5, label="Preceding action: non-AF (0)")
    ax.plot(probabilities, curves[1], color=orange, linewidth=2.5, label="Preceding action: AF (1)")
    ax.axhline(0.5, color="#555555", linestyle="--", linewidth=1.1)
    ax.text(0.012, 0.515, "Action threshold", color="#555555", fontsize=8, va="bottom")

    for crossing, color, label in ((enter, blue, "Enter AF"), (remain, orange, "Remain in AF")):
        if crossing is None:
            continue
        ax.vlines(crossing, 0, 0.5, color=color, linestyle=":", linewidth=1.4)
        ax.scatter([crossing], [0.5], s=42, color=color, edgecolor="white", linewidth=0.8, zorder=5)
        ax.annotate(
            f"{label}: p ≈ {crossing:.3f}",
            xy=(crossing, 0.5),
            xytext=(8, 14 if label == "Enter AF" else -22),
            textcoords="offset points",
            color=color,
            fontsize=8.5,
            ha="left",
            va="center",
        )

    ax.set_title(
        f"Policy response to a flat {detector_label} probability profile", loc="left", fontweight="bold"
    )
    ax.set_xlabel(f"{detector_label} probability (current; forward contrasts = 0)")
    ax.set_ylabel("Policy probability of AF action")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 11))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(axis="both", color="#E1E5E8", linewidth=0.7)
    ax.legend(frameon=False, loc="upper left")
    fig.text(
        0.11,
        0.015,
        f"Heart rate fixed at the training median: {response['median_bpm']:.1f} bpm "
        f"(normalized {response['median_hr']:.3f}). Forward contrasts = 0. "
        "Raw policy output; 30-s action mask not applied.",
        fontsize=8,
        color="#4A5965",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    plt.show()

from functools import lru_cache

import numpy as np
import torch
from torch.utils.data import Dataset

from ..util import GLOBAL_CONFIG
from .common import load_processed_record
from .preprocessing import resample_signal


@lru_cache(maxsize=None)
def load_noise_index(source, record_id):
    cfg = GLOBAL_CONFIG["modes"]["noise"]
    input_fs = int(GLOBAL_CONFIG["data"]["preprocessing"]["target_fs"])
    window_samples = int(round(float(cfg["window_seconds"]) * input_fs))
    step_samples = max(1, int(round(float(cfg["step_seconds"]) * input_fs)))

    signal, _, _ = load_processed_record(source, record_id)
    starts = np.arange(
        0, len(signal) - window_samples + 1, step_samples, dtype=np.int64
    )
    return starts


class NoiseDataset(Dataset):
    def __init__(self, records, split):
        self.records = records
        self.split = split
        self.step_seconds = float(GLOBAL_CONFIG["modes"]["noise"]["step_seconds"])
        self.input_fs = int(GLOBAL_CONFIG["data"]["preprocessing"]["target_fs"])
        self.target_fs = int(GLOBAL_CONFIG["models"]["noise_detector"]["target_fs"])
        self.window_samples = int(
            round(
                float(GLOBAL_CONFIG["modes"]["noise"]["window_seconds"])
                * self.input_fs
            )
        )
        self.index = []
        self.starts = {}

        for source, record_id in records:
            key = (source, record_id)
            starts = load_noise_index(source, record_id)
            self.starts[key] = starts
            for segment_idx in range(len(starts)):
                self.index.append((source, record_id, segment_idx))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        source, record_id, segment_idx = self.index[idx]
        signal = load_processed_record(source, record_id)[0]
        start = int(self.starts[(source, record_id)][segment_idx])
        segment = signal[start:start + self.window_samples]
        segment = resample_signal(
            segment, orig_fs=self.input_fs, target_fs=self.target_fs
        ).astype(np.float32, copy=False)
        label = torch.tensor(0.0, dtype=torch.float32)
        return torch.from_numpy(segment).unsqueeze(0), label, idx

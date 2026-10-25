from functools import lru_cache

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..util import GLOBAL_CONFIG
from .common import load_processed_record, log_binary_labels


def beat_window_bounds():
    pre_beat = GLOBAL_CONFIG["modes"]["beat"]["pre_beat_window"]
    post_beat = GLOBAL_CONFIG["modes"]["beat"]["post_beat_window"]
    return pre_beat, post_beat


@lru_cache(maxsize=None)
def load_beat_index(source, record_id):
    pre_beat, post_beat = beat_window_bounds()

    processed_signal, processed_labels, processed_r_peaks = load_processed_record(source, record_id)
    n_samples = len(processed_signal)
    valid_window_mask = (processed_r_peaks >= pre_beat) & (processed_r_peaks + post_beat <= n_samples)
    windowed_r_peaks = processed_r_peaks[valid_window_mask].astype(np.int64)
    beat_labels = processed_labels[windowed_r_peaks].astype(np.int8)

    return beat_labels, windowed_r_peaks


class AFBeatDataset(Dataset):
    def __init__(self, records, split):
        self.records = records
        self.split = split
        self.pre_beat, self.post_beat = beat_window_bounds()
        # test is a single pass over every record; slice its signals lazily so only
        # one is resident. train/val are revisited each epoch, so keep them materialized.
        self.cache_signal = split != "test"
        self.index = []
        self.labels = {}
        self.peaks = {}
        self.signals = {}

        for source, record_id in records:
            key = (source, record_id)
            beat_labels, windowed_r_peaks = load_beat_index(source, record_id)
            self.labels[key] = beat_labels
            self.peaks[key] = windowed_r_peaks
            if self.cache_signal:
                self.signals[key] = load_processed_record(source, record_id)[0]
            for beat_idx in range(len(windowed_r_peaks)):
                self.index.append((source, record_id, beat_idx))

    def __len__(self):
        return len(self.index)

    def get_signal(self, key):
        if self.cache_signal:
            return self.signals[key]
        return load_processed_record(key[0], key[1])[0]

    def __getitem__(self, idx):
        source, record_id, beat_idx = self.index[idx]
        key = (source, record_id)
        center = int(self.peaks[key][beat_idx])
        beat = self.get_signal(key)[center - self.pre_beat:center + self.post_beat].astype(np.float32, copy=True)
        signal = torch.from_numpy(beat).unsqueeze(0)
        label = torch.tensor(float(self.labels[key][beat_idx]), dtype=torch.float32)
        return signal, label, idx


def prepare_beat_data(folds, k):
    batch_size = GLOBAL_CONFIG["train"]["batch_size"]
    pin_memory = torch.cuda.is_available()

    train_dataset = AFBeatDataset(folds[k]["train"], split="train")
    val_dataset = AFBeatDataset(folds[k]["val"], split="val")
    test_dataset = AFBeatDataset(folds[k]["test"], split="test")

    for split, ds in (("train", train_dataset), ("val", val_dataset), ("test", test_dataset)):
        y = np.concatenate(list(ds.labels.values())) if ds.labels else np.array([])
        log_binary_labels(f"beats fold{k} {split}", y)

    return {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory) if len(train_dataset) else None,
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory) if len(val_dataset) else None,
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory) if len(test_dataset) else None,
    }

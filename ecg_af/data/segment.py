from functools import lru_cache

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..util import GLOBAL_CONFIG
from .common import load_processed_record, log_binary_labels


def segment_step_seconds(split):
    seg_cfg = GLOBAL_CONFIG["modes"]["segment"]
    if split in ("train", "val"):
        return float(seg_cfg["train_step_seconds"])
    if split == "test":
        return float(seg_cfg["test_step_seconds"])
    raise ValueError(f"Invalid split: {split}")


def segment_window_samples():
    window_seconds = GLOBAL_CONFIG["modes"]["segment"]["window_seconds"]
    target_fs = GLOBAL_CONFIG["data"]["preprocessing"]["target_fs"]
    return int(round(float(window_seconds) * target_fs))


@lru_cache(maxsize=None)
def load_segment_index(source, record_id, step_seconds):
    threshold = GLOBAL_CONFIG["modes"]["segment"]["af_ratio_threshold"]
    target_fs = GLOBAL_CONFIG["data"]["preprocessing"]["target_fs"]

    window_samples = segment_window_samples()
    step_samples = max(1, int(round(float(step_seconds) * target_fs)))

    processed_signal, processed_labels, _ = load_processed_record(source, record_id)
    n_samples = len(processed_signal)
    segment_starts = np.arange(0, n_samples - window_samples + 1, step_samples, dtype=np.int64)

    label_cumsum = np.concatenate(([0], np.cumsum(processed_labels, dtype=np.int64)))
    window_af_counts = label_cumsum[segment_starts + window_samples] - label_cumsum[segment_starts]
    segment_labels = (window_af_counts > threshold * window_samples).astype(np.int8)

    return segment_labels, segment_starts


def segment_export_step_seconds():
    return float(GLOBAL_CONFIG["modes"]["segment"]["test_step_seconds"])


class AFSegmentDataset(Dataset):
    def __init__(self, records, split, step_seconds=None):
        self.records = records
        self.split = split
        self.step_seconds = (
            float(step_seconds) if step_seconds is not None else segment_step_seconds(split)
        )
        self.window_samples = segment_window_samples()
        # train/val epoch loops cache signals; single-pass fine-step export and test
        # slice lazily so only one record is resident at a time.
        self.cache_signal = split in ("train", "val") and step_seconds is None
        self.index = []
        self.labels = {}
        self.starts = {}
        self.signals = {}

        for source, record_id in records:
            key = (source, record_id)
            segment_labels, segment_starts = load_segment_index(source, record_id, self.step_seconds)
            self.labels[key] = segment_labels
            self.starts[key] = segment_starts
            if self.cache_signal:
                self.signals[key] = load_processed_record(source, record_id)[0]
            for seg_idx in range(len(segment_starts)):
                self.index.append((source, record_id, seg_idx))

    def __len__(self):
        return len(self.index)

    def get_signal(self, key):
        if self.cache_signal:
            return self.signals[key]
        return load_processed_record(key[0], key[1])[0]

    def __getitem__(self, idx):
        source, record_id, seg_idx = self.index[idx]
        key = (source, record_id)
        start = int(self.starts[key][seg_idx])
        segment = self.get_signal(key)[start:start + self.window_samples].astype(np.float32, copy=True)
        signal = torch.from_numpy(segment).unsqueeze(0)
        label = torch.tensor(float(self.labels[key][seg_idx]), dtype=torch.float32)
        return signal, label, idx


def prepare_segment_data(folds, k):
    batch_size = GLOBAL_CONFIG["train"]["batch_size"]
    pin_memory = torch.cuda.is_available()

    train_dataset = AFSegmentDataset(folds[k]["train"], split="train")
    val_dataset = AFSegmentDataset(folds[k]["val"], split="val")
    test_dataset = AFSegmentDataset(folds[k]["test"], split="test")

    for split, ds in (("train", train_dataset), ("val", val_dataset), ("test", test_dataset)):
        y = np.concatenate(list(ds.labels.values())) if ds.labels else np.array([])
        log_binary_labels(f"segments fold{k} {split}", y)

    return {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory) if len(train_dataset) else None,
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory) if len(val_dataset) else None,
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory) if len(test_dataset) else None,
    }

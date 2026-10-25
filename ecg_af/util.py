import copy
import yaml
import os
from pathlib import Path
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch


import logging
import sys

def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.propagate = False  # prevent duplicate logs

    # Avoid adding handlers multiple times (important in notebooks / reloads)
    if logger.handlers:
        return logger

    # Formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    os.makedirs(Path(__file__).parent / "logs", exist_ok=True)
    file_handler = logging.FileHandler(Path(__file__).parent / "logs" / f"{GLOBAL_CONFIG.name}.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Console handler (print)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Add both
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class ConfigStore:
    def __init__(self):
        self.name = None
        self.run_tag = None
        self._config = None
        self.rng = None
        self.debug_print = False

    def set(self, name, config):
        self.name = name
        self.run_tag = None
        self._config = config
        self.rng = np.random.default_rng(config["data"]["seed"])
        self.logger = setup_logger()

    def load_config(self, experiment_name):
        with open(Path(__file__).parent / "config.yaml", "r") as f:
            config = yaml.safe_load(f)
        experiment = config["experiments"][experiment_name]
        out = resolve_config(config["defaults"], experiment)
        set_reproducibility(out["data"]["seed"])

        self.set(experiment_name, out)

    def apply_overrides(self, overrides):
        for path, value in overrides.items():
            keys = path.split(".")
            node = self._config
            for key in keys[:-1]:
                if key not in node:
                    raise KeyError(f"Unknown config path {path!r}: missing {key!r}")
                node = node[key]
                if not isinstance(node, dict):
                    raise KeyError(f"Unknown config path {path!r}: {key!r} is not a mapping")
            leaf = keys[-1]
            if leaf not in node:
                raise KeyError(f"Unknown config path {path!r}: missing {leaf!r}")
            node[leaf] = value

    def __getitem__(self, key):
        return self._config[key]

    def __iter__(self):
        return iter(self._config)

    def __len__(self):
        return len(self._config)


def deep_merge(base, override):
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def resolve_config(defaults, experiment):
    defaults = copy.deepcopy(defaults)
    out = deep_merge(defaults, experiment)

    return out


def set_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


GLOBAL_CONFIG = ConfigStore()

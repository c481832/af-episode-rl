import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..analysis import get_metrics
from ..benchmark_eval import boundary_tolerance_sweep
from ..postprocess import filter_short_intervals
from ..util import GLOBAL_CONFIG
from .evaluations import controller_checkpoint_dir, reward_metrics_for_records
from .observations import load_controller_arrays, observation_names


def make_supervised_model(n_obs, net_arch):
    layers = []
    n_in = n_obs
    for n_hidden in net_arch:
        layers.append(nn.Linear(n_in, int(n_hidden)))
        layers.append(nn.Tanh())
        n_in = int(n_hidden)
    layers.append(nn.Linear(n_in, 1))
    return nn.Sequential(*layers)


def supervised_checkpoint_path():
    return controller_checkpoint_dir() / "supervised_best.pt"


def stack_controller_seconds(record_ids):
    keep = [
        i for i, name in enumerate(observation_names())
        if name != "last_state"
    ]
    features = []
    labels = []
    for source, record_id in record_ids:
        arrays = load_controller_arrays(source, record_id)
        features.append(arrays["base_features"][:, keep])
        labels.append(filter_short_intervals(arrays["true_states"]))
    if len(features) == 0:
        raise ValueError("No records to stack for the supervised comparator")
    return (
        np.concatenate(features, axis=0).astype(np.float32),
        np.concatenate(labels, axis=0).astype(np.float32),
    )


@torch.no_grad()
def rollout_supervised(model, record_ids):
    if len(record_ids) == 0:
        return {}
    model.eval()
    device = next(model.parameters()).device
    keep = [
        i for i, name in enumerate(observation_names())
        if name != "last_state"
    ]
    out = {}
    for source, record_id in record_ids:
        arrays = load_controller_arrays(source, record_id)
        features = arrays["base_features"][:, keep]
        logits = model(torch.as_tensor(features, device=device)).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
        pred = (probs > 0.5).astype(np.float32)
        true = arrays["true_states"].astype(np.float32)
        out[(source, record_id)] = (true, pred, probs)
    return out


def supervised_record_labels(model, record_ids):
    rollouts = rollout_supervised(model, record_ids)
    true_by = {}
    pred_by = {}
    for key in record_ids:
        true_pp, pred_pp, _probs = rollouts[key]
        true_by[key] = filter_short_intervals(true_pp)
        pred_by[key] = filter_short_intervals(pred_pp)
    return true_by, pred_by


def load_supervised(path, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = make_supervised_model(int(payload["n_obs"]), list(payload["net_arch"]))
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload["history"]


def fit_supervised(train_record_ids, val_record_ids, load_checkpoint):
    train_cfg = GLOBAL_CONFIG["rl"]["train"]
    supervised_cfg = train_cfg["supervised"]
    device = train_cfg["device"]
    net_arch = list(train_cfg["net_arch"])
    batch_size = int(train_cfg["batch_size"])
    epochs = int(supervised_cfg["n_epochs"])
    lr = float(supervised_cfg["learning_rate"])
    if epochs <= 0:
        raise ValueError(f"Supervised n_epochs must be > 0, got {epochs}")
    seed = int(GLOBAL_CONFIG["data"]["seed"])
    coverage_lambda = float(GLOBAL_CONFIG["rl"]["ppo_reward"]["coverage"]["lambda"])
    checkpoint_path = supervised_checkpoint_path()

    if load_checkpoint:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No supervised checkpoint at {checkpoint_path}")
        GLOBAL_CONFIG.logger.info("Reloading supervised comparator from %s", checkpoint_path)
        model, history = load_supervised(checkpoint_path, device)
        return model, pd.DataFrame(history), checkpoint_path

    torch.manual_seed(seed)
    x_train, y_train = stack_controller_seconds(train_record_ids)
    n_pos = float((y_train >= 0.5).sum())
    n_neg = float((y_train < 0.5).sum())
    if n_pos == 0.0:
        raise ValueError("Supervised train set has no AF seconds after duration filtering")
    pos_weight = torch.tensor(n_neg / n_pos, device=device, dtype=torch.float32)

    model = make_supervised_model(x_train.shape[1], net_arch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    GLOBAL_CONFIG.logger.info(
        "Supervised comparator: epochs=%d lr=%g batch=%d pos_weight=%.4f "
        "n_obs=%d net_arch=%s train_seconds=%d train=%d val=%d device=%s",
        epochs, lr, batch_size, float(pos_weight), x_train.shape[1], net_arch,
        len(x_train), len(train_record_ids), len(val_record_ids), device,
    )

    history = []
    best_val_boundary_f1 = None
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_seen = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb).squeeze(-1)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, yb, pos_weight=pos_weight
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(yb)
            n_seen += len(yb)
        train_loss = total_loss / n_seen

        true_by, pred_by = supervised_record_labels(model, val_record_ids)
        val_f1 = float(
            get_metrics(
                np.concatenate([true_by[key] for key in val_record_ids]),
                np.concatenate([pred_by[key] for key in val_record_ids]),
            )["f1"]
        )
        val_boundary_f1 = float(
            boundary_tolerance_sweep(
                true_by,
                pred_by,
                [1, 2, 3, 5, 10, 15, 20, 30],
                0.5,
            )["Boundary F1"].mean(skipna=False)
        )
        if np.isnan(val_boundary_f1):
            raise ValueError("Supervised val Boundary F1 is nan")
        _, val_reward = reward_metrics_for_records(true_by, pred_by, 0.5, coverage_lambda)
        is_best = best_val_boundary_f1 is None or val_boundary_f1 > best_val_boundary_f1
        if is_best:
            best_val_boundary_f1 = val_boundary_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_f1": val_f1,
            "val_boundary_f1": val_boundary_f1,
            "val_reward_total": val_reward["reward_total"],
            "val_n_switches": val_reward["n_switches"],
            "is_best": is_best,
        }
        history.append(row)
        GLOBAL_CONFIG.logger.info(
            "Supervised epoch %d/%d | train_loss=%.4f val_boundary_f1=%.4f val_f1=%.4f "
            "val_reward_total=%.4f val_nsw=%.2f%s",
            epoch, epochs, train_loss, val_boundary_f1, val_f1,
            val_reward["reward_total"], val_reward["n_switches"],
            " [best]" if is_best else "",
        )

    if best_state is None:
        raise RuntimeError("Supervised comparator did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_obs": int(x_train.shape[1]),
            "net_arch": net_arch,
            "history": history,
        },
        checkpoint_path,
    )
    GLOBAL_CONFIG.logger.info(
        "Supervised best val_boundary_f1=%.4f; saved %s", best_val_boundary_f1, checkpoint_path
    )
    return model, pd.DataFrame(history), checkpoint_path

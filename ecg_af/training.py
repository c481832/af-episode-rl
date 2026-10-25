import copy
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from .analysis import infer_record_probs
from .util import GLOBAL_CONFIG
from .models import get_model_registry
from .data.beat import AFBeatDataset, prepare_beat_data
from .data.noise import NoiseDataset
from .data.predictions import fold_predictions_path, save_fold_results
from .data.segment import AFSegmentDataset, prepare_segment_data, segment_export_step_seconds
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")


def prepare_data(folds, k, mode):
    if mode == "segment":
        return prepare_segment_data(folds, k)
    elif mode == "beat":
        return prepare_beat_data(folds, k)
    else:
        raise ValueError(f"Invalid mode: {mode}")


def export_loader(folds, k, mode, split):
    batch_size = GLOBAL_CONFIG["train"]["batch_size"]
    pin_memory = torch.cuda.is_available()
    records = folds[k][split]
    if mode == "beat":
        dataset = AFBeatDataset(records, split=split)
    elif mode == "segment":
        dataset = AFSegmentDataset(
            records, split=split, step_seconds=segment_export_step_seconds()
        )
    elif mode == "noise":
        dataset = NoiseDataset(records, split=split)
    else:
        raise ValueError(f"Invalid mode for export: {mode}")
    if len(dataset) == 0:
        return None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )


def save_training_predictions(model, mode, folds, k, checkpoint_dir):
    merged_probs = {}
    merged_times = {}
    for split in ("train", "val", "test"):
        loader = export_loader(folds, k, mode, split)
        if loader is not None:
            _, record_probs, record_times = infer_record_probs(model, loader, mode)
            for record_key, probs in record_probs.items():
                if record_key in merged_probs:
                    raise ValueError(
                        f"Duplicate record {record_key} across splits for fold {k:02d}"
                    )
                merged_probs[record_key] = probs
                merged_times[record_key] = record_times[record_key]
        for source, record_id in folds[k][split]:
            record_key = (source, record_id)
            if record_key not in merged_probs:
                GLOBAL_CONFIG.logger.warning(
                    "Record %s/%s produced no predictions (too short?), saving empty arrays",
                    source, record_id,
                )
                merged_probs[record_key] = np.zeros(0, dtype=np.float32)
                merged_times[record_key] = np.zeros(0, dtype=np.float64)
    out_path = checkpoint_dir / "predictions.npz"
    save_fold_results(out_path, merged_probs, merged_times)
    GLOBAL_CONFIG.logger.info(
        "Saved %d record predictions to %s",
        len(merged_probs),
        out_path,
    )


def plot_training_history(history):
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    train_f1 = [row["train_f1"] for row in history]
    val_f1 = [row["val_f1"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_loss, marker="o", label="train_loss")
    axes[0].plot(epochs, val_loss, marker="o", label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Focal loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_f1, marker="o", label="train_f1")
    axes[1].plot(epochs, val_f1, marker="o", label="val_f1")
    axes[1].set_title("F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    plt.show()


def compute_batch_metrics(logits, targets):
    preds = (torch.sigmoid(logits) >= 0.5).float()
    labels = (targets >= 0.5).float()

    tp = float(((preds == 1) & (labels == 1)).sum().item())
    fp = float(((preds == 1) & (labels == 0)).sum().item())
    fn = float(((preds == 0) & (labels == 1)).sum().item())
    tn = float(((preds == 0) & (labels == 0)).sum().item())

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn
    }


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        return ((1 - p_t) ** self.gamma * bce).mean()


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train(train)
    total_loss = 0.0
    total_tp = 0.0
    total_fp = 0.0
    total_tn = 0.0
    total_fn = 0.0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for signals, labels, _ in loader:
            signals = signals.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.float32)
            logits = model(signals)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            batch_size = signals.size(0)
            total_loss += loss.item() * batch_size
            metrics = compute_batch_metrics(logits.detach(), labels)
            total_tp += metrics["tp"]
            total_fp += metrics["fp"]
            total_fn += metrics["fn"]
            total_tn += metrics["tn"]

    eps = 1e-8
    precision = total_tp / (total_tp + total_fp + eps)
    recall = total_tp / (total_tp + total_fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)

    return {
        "loss": total_loss / len(loader.dataset),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def init_model(model, info_print):
    device = GLOBAL_CONFIG["train"]["device"]
    model = get_model_registry()[model]()
    model = model.to(device)
    if info_print:
        GLOBAL_CONFIG.logger.info(model)
    return model


def init_optimizer(model):
    train_cfg = GLOBAL_CONFIG["train"]
    trainable = [p for p in model.parameters() if p.requires_grad]
    if train_cfg["scheduler"]:
        optimizer = torch.optim.AdamW(
            trainable,
            lr=GLOBAL_CONFIG["train"]["lr"],
            betas=(0.9, 0.999),
            weight_decay=GLOBAL_CONFIG["train"]["weight_decay"],
        )
    else:
        optimizer = torch.optim.Adam(
            trainable,
            lr=GLOBAL_CONFIG["train"]["lr"],
            betas=(0.9, 0.999),
            weight_decay=GLOBAL_CONFIG["train"]["weight_decay"],
        )
    return optimizer


def reload_model(model, optimizer, checkpoint_dir):
    checkpoint_path = checkpoint_dir / "best.pt"
    if checkpoint_path.exists():
        GLOBAL_CONFIG.logger.info("Reloading checkpoint from %s...", checkpoint_path)
        history = load_checkpoint(checkpoint_path, model, optimizer)
        best_val_loss = float("inf")
        best_epoch = 0
        for epoch in range(1, GLOBAL_CONFIG["train"]["epochs"] + 1):
            i = epoch - 1
            if history[i]["val_loss"] < best_val_loss:
                best_epoch = epoch
                best_val_loss = history[i]["val_loss"]
            GLOBAL_CONFIG.logger.info(
                "Epoch %03d | train_loss=%.4f train_f1=%.4f | val_loss=%.4f val_f1=%.4f | learning_rate=%.6f",
                epoch,
                history[i]["train_loss"],
                history[i]["train_f1"],
                history[i]["val_loss"],
                history[i]["val_f1"],
                optimizer.param_groups[0]["lr"],
            )
        GLOBAL_CONFIG.logger.info("Best epoch: %s, best val loss: %s", best_epoch, best_val_loss)
        return True, model, history
    else:
        return False, model, None
                

def fit_model_single(data_dict, model, optimizer, checkpoint_dir, model_name):
    device = GLOBAL_CONFIG["train"]["device"]
    criterion = BinaryFocalLoss(gamma=GLOBAL_CONFIG["train"]["gamma"])
    train_cfg = GLOBAL_CONFIG["train"]

    if train_cfg["scheduler"]:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=train_cfg["scheduler"]["mode"],
            factor=train_cfg["scheduler"]["factor"],
            patience=train_cfg["scheduler"]["patience"],
            min_lr=train_cfg["scheduler"]["min_lr"],
        )

    GLOBAL_CONFIG.logger.info(
        "Training model %s on device=%s for %s epochs...",
        model_name,
        device,
        GLOBAL_CONFIG["train"]["epochs"],
    )
    
    best_val_loss = float("inf")
    best_epoch = 0
    best_state_dict = None
    best_optimizer_state_dict = None
    history = []
    for epoch in range(1, GLOBAL_CONFIG["train"]["epochs"] + 1):
        train_metrics = run_epoch(
            model=model,
            loader=data_dict["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train=True,
        )
        val_metrics = run_epoch(
            model=model,
            loader=data_dict["val"],
            criterion=criterion,
            optimizer=None,
            device=device,
            train=False,
        )

        epoch_row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_f1": train_metrics["f1"],
            "val_loss": val_metrics["loss"],
            "val_f1": val_metrics["f1"],
        }
        history.append(epoch_row)
        if train_cfg["scheduler"]:
            scheduler.step(val_metrics["loss"])

        if val_metrics["loss"] < best_val_loss:
            best_epoch = epoch
            best_val_loss = val_metrics["loss"]
            best_state_dict = copy.deepcopy(model.state_dict())
            best_optimizer_state_dict = copy.deepcopy(optimizer.state_dict())

        checkpoint_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
        save_checkpoint(checkpoint_path, model.state_dict(), optimizer.state_dict(), history)

        GLOBAL_CONFIG.logger.info(
            "Epoch %03d | train_loss=%.4f train_f1=%.4f | val_loss=%.4f val_f1=%.4f | learning_rate=%.6f",
            epoch,
            train_metrics["loss"],
            train_metrics["f1"],
            val_metrics["loss"],
            val_metrics["f1"],
            optimizer.param_groups[0]["lr"],
        )

    if best_state_dict is not None:
        checkpoint_path = checkpoint_dir / "best.pt"
        save_checkpoint(checkpoint_path, best_state_dict, best_optimizer_state_dict, history)
        model.load_state_dict(best_state_dict)
    GLOBAL_CONFIG.logger.info("Best epoch: %s, best val loss: %s", best_epoch, best_val_loss)
    return model, history


def fit_model_k_fold(folds, reload_checkpoint, mode, model):
    k_fold_models = []
    k_fold_histories = []
    data_loaders = []

    for k in range(GLOBAL_CONFIG["data"]["k_folds"]["n_splits"]):
        if k == 0:
            info_print = True
        else:
            info_print = False
        GLOBAL_CONFIG.logger.info("Training model %s for fold %02d...", model, k)
        nn_model = init_model(model, info_print)

        checkpoint_dir = fold_predictions_path(model, k).parent
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        inference_only = bool(
            GLOBAL_CONFIG["models"].get(model, {}).get("inference_only", False)
        )
        if inference_only:
            if not reload_checkpoint:
                raise ValueError(f"Model {model!r} is inference-only; pass --reload")
            data_dict = None
            history = None
            loaded = True
        else:
            data_dict = prepare_data(folds, k, mode)
            optimizer = init_optimizer(nn_model)
            loaded, nn_model, history = (
                reload_model(nn_model, optimizer, checkpoint_dir)
                if reload_checkpoint
                else (False, nn_model, None)
            )
            if not loaded:
                nn_model, history = fit_model_single(data_dict, nn_model, optimizer, checkpoint_dir, model)
        pred_path = checkpoint_dir / "predictions.npz"
        if not loaded or not pred_path.exists() or mode == "segment":
            save_training_predictions(nn_model, mode, folds, k, checkpoint_dir)
        data_loaders.append(data_dict)
        k_fold_models.append(nn_model)
        k_fold_histories.append(history)
    return k_fold_models, k_fold_histories, data_loaders


def save_checkpoint(path, model_state_dict, optimizer_state_dict, history):
    torch.save({
        "model": model_state_dict,
        "optimizer": optimizer_state_dict,
        "history": history,    
    }, path)


def load_checkpoint(path, model, optimizer):
    device = GLOBAL_CONFIG["train"]["device"]
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["history"]

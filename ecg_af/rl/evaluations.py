import hashlib

import numpy as np

from ..analysis import get_metrics as get_second_metrics
from ..data.common import experiment_run_dir
from ..postprocess import filter_short_intervals
from ..util import GLOBAL_CONFIG
from .observations import load_controller_arrays
from .policy import rollout_policy
from .reward import episode_reward_breakdown, switch_count


def controller_checkpoint_dir():
    path = experiment_run_dir() / "controller"
    if GLOBAL_CONFIG.run_tag is not None:
        path = path / GLOBAL_CONFIG.run_tag
    path.mkdir(parents=True, exist_ok=True)
    return path


def evaluate_record_set(policy, record_ids, policy_key):
    digest = hashlib.sha256()
    for name, tensor in sorted(policy.policy.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    policy_key = f"{policy_key}/{digest.hexdigest()[:16]}"
    cache_dir = controller_checkpoint_dir() / "rollouts" / policy_key

    preds = {}
    miss_ids = []
    for source, record_id in record_ids:
        path = (cache_dir / f"{source}|{record_id}.npz").resolve()
        if path.exists():
            with np.load(path, allow_pickle=False) as raw:
                preds[(source, record_id)] = (
                    np.asarray(raw["true"], dtype=np.float32),
                    np.asarray(raw["pred"], dtype=np.float32),
                    np.asarray(raw["pred_probs"], dtype=np.float32),
                )
        else:
            miss_ids.append((source, record_id))

    for (source, record_id), (true_pp, pred_pp, pred_probs) in rollout_policy(
        policy, miss_ids
    ).items():
        path = (cache_dir / f"{source}|{record_id}.npz").resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, true=true_pp, pred=pred_pp, pred_probs=pred_probs)
        preds[(source, record_id)] = (true_pp, pred_pp, pred_probs)

    rollouts = {}
    all_true = {}
    all_pred = {}
    second_metrics = {}
    for source, record_id in record_ids:
        true_pp, pred_pp, pred_probs = preds[(source, record_id)]
        # Only the reference is filtered: the controller's output already obeys the
        # min-duration rule, because the exit action is masked until a run confirms
        # (see AFControllerEnv.action_masks).
        true_pp = filter_short_intervals(true_pp)
        second = get_second_metrics(true_pp, pred_pp)
        rollouts[(source, record_id)] = {
            "true": true_pp,
            "pred": pred_pp,
            "pred_probs": pred_probs,
            "second": second,
        }
        record_key = f"{source}_{record_id}"
        second_metrics[record_key] = second
        all_true[record_key] = true_pp
        all_pred[record_key] = pred_pp
    second_metrics["total"] = get_second_metrics(
        np.concatenate(list(all_true.values())),
        np.concatenate(list(all_pred.values())),
    )
    return rollouts, second_metrics


def baseline_reference(record_ids):
    all_true, all_pred, switch_counts = [], [], []
    for source, record_id in record_ids:
        arrays = load_controller_arrays(source, record_id)
        true_labels = filter_short_intervals(arrays["true_states"])
        pred_labels = filter_short_intervals(arrays["base_pred"])
        all_true.append(true_labels)
        all_pred.append(pred_labels)
        switch_counts.append(switch_count(pred_labels))
    f1 = float(get_second_metrics(np.concatenate(all_true), np.concatenate(all_pred))["f1"])
    return {"f1": f1, "mean_switches": float(np.mean(switch_counts))}


def reward_metrics_from_rollouts(rollouts, coverage_lambda):
    true_by_record = {record_key: pack["true"] for record_key, pack in rollouts.items()}
    pred_by_record = {record_key: pack["pred"] for record_key, pack in rollouts.items()}
    return reward_metrics_for_records(
        true_by_record, pred_by_record, 0.5, coverage_lambda
    )


def reward_metrics_for_records(true_by_record, pred_by_record, threshold, coverage_lambda):
    per_record = {}
    totals, edges, coverages, switch_rewards, n_switches, n_outside = [], [], [], [], [], []
    for record_key in true_by_record:
        true_labels = (np.asarray(true_by_record[record_key]) >= 0.5).astype(np.int8)
        pred_labels = (np.asarray(pred_by_record[record_key]) >= threshold).astype(np.int8)
        rb = episode_reward_breakdown(true_labels, pred_labels, coverage_lambda)
        per_record[f"{record_key[0]}_{record_key[1]}" if isinstance(record_key, tuple) else record_key] = rb
        totals.append(rb["total"])
        edges.append(rb["edge"])
        coverages.append(rb["coverage"])
        switch_rewards.append(rb["switch_reward"])
        n_switches.append(rb["n_switches"])
        n_outside.append(rb["switches_outside_edge"])
    breakdowns = list(per_record.values())
    total = {
        f"reward_{name}": float(np.mean([b[name] for b in breakdowns])) if breakdowns else np.nan
        for name in ("structural", "onset", "offset", "fragment")
    }
    total |= {
        "reward_total": float(np.mean(totals)) if totals else np.nan,
        "reward_edge": float(np.mean(edges)) if edges else np.nan,
        "reward_coverage": float(np.mean(coverages)) if coverages else np.nan,
        "reward_switch": float(np.mean(switch_rewards)) if switch_rewards else np.nan,
        "n_switches": float(np.mean(n_switches)) if n_switches else np.nan,
        "n_switches_outside_edge": float(np.mean(n_outside)) if n_outside else np.nan,
    }
    return per_record, total

import numpy as np

from ..postprocess import labels_to_intervals
from ..util import GLOBAL_CONFIG


def side_kernel(dist, tau, reach):
    d = np.maximum(np.asarray(dist, dtype=np.float64), 0.0)
    return np.where(d <= reach, np.exp(-d / tau), 0.0)


def edge_weights(true_states):
    true_states = np.asarray(true_states).astype(np.int8)
    n = len(true_states)
    edge_cfg = GLOBAL_CONFIG["rl"]["ppo_reward"]["edge"]
    tau = float(edge_cfg["tau_seconds"])
    reach = float(edge_cfg["reach_seconds"])
    transitions = np.diff(true_states)
    onsets = np.where(transitions == 1)[0] + 1
    offsets = np.where(transitions == -1)[0] + 1
    edges = np.concatenate([onsets, offsets])
    idx = np.arange(n)
    edge_w = np.zeros(n, dtype=np.float64)
    for edge_pos in edges:
        edge_w += side_kernel(np.abs(idx - edge_pos), tau, reach)
    return edge_w


def event_channels(interval, n_labels, env_start, env_end, n_fragments):
    cfg = GLOBAL_CONFIG["rl"]["ppo_reward"]["structural"]
    start, end = interval
    has_onset = start > 0
    has_offset = end < n_labels
    if n_fragments == 0 or not (has_onset or has_offset):
        return np.zeros(3)

    tau = float(cfg["tau_seconds"])
    return np.array([
        float(cfg["onset_weight"]) * np.exp(-abs(env_start - start) / tau) if has_onset else 0.0,
        float(cfg["offset_weight"]) * np.exp(-abs(env_end - end) / tau) if has_offset else 0.0,
        -float(cfg["fragment_weight"]) * (n_fragments - 1),
    ])


class StructuralScorer:
    def __init__(self, true_states):
        self.n_labels = len(true_states)
        self.intervals = labels_to_intervals(true_states)
        self.event_index = np.full(self.n_labels, -1, dtype=np.int64)
        for i, (start, end) in enumerate(self.intervals):
            self.event_index[int(start):int(end)] = i
        n_events = len(self.intervals)
        self.env_start = np.zeros(n_events)
        self.env_end = np.zeros(n_events)
        self.n_fragments = np.zeros(n_events, dtype=np.int64)
        self.paid = np.zeros((n_events, 3))
        self.interval_start = 0
        self.overlapped = []

    def step(self, t, pred_state, prev_pred_state):
        if pred_state == 0:
            self.overlapped = []
            return 0.0
        if prev_pred_state == 0:
            self.interval_start = t
            self.overlapped = []

        event = int(self.event_index[t])
        if event >= 0 and event not in self.overlapped:
            self.overlapped.append(event)
            self.n_fragments[event] += 1
            if self.n_fragments[event] == 1:
                self.env_start[event] = self.interval_start
        self.env_end[self.overlapped] = t + 1

        reward = 0.0
        for event in self.overlapped:
            channels = event_channels(
                self.intervals[event],
                self.n_labels,
                self.env_start[event],
                self.env_end[event],
                self.n_fragments[event],
            )
            reward += float(channels.sum() - self.paid[event].sum())
            self.paid[event] = channels
        return reward

    def totals(self):
        return dict(zip(("onset", "offset", "fragment"), self.paid.sum(axis=0)))


def structural_breakdown(true_labels, pred_labels):
    cfg = GLOBAL_CONFIG["rl"]["ppo_reward"]["structural"]
    n_labels = len(true_labels)
    true_iv = np.asarray(labels_to_intervals(true_labels), dtype=np.float64).reshape(-1, 2)
    pred_iv = np.asarray(labels_to_intervals(pred_labels), dtype=np.float64).reshape(-1, 2)
    if len(true_iv) == 0 or len(pred_iv) == 0:
        return {"onset": 0.0, "offset": 0.0, "fragment": 0.0}

    start, end = true_iv[:, 0], true_iv[:, 1]
    overlap = np.minimum(end[:, None], pred_iv[:, 1]) > np.maximum(start[:, None], pred_iv[:, 0])
    n_fragments = overlap.sum(axis=1)
    env_start = np.where(overlap, pred_iv[:, 0], np.inf).min(axis=1)
    env_end = np.where(overlap, pred_iv[:, 1], -np.inf).max(axis=1)

    tau = float(cfg["tau_seconds"])
    has_onset, has_offset = start > 0, end < n_labels
    scored = (n_fragments > 0) & (has_onset | has_offset)
    onset = np.where(
        scored & has_onset,
        float(cfg["onset_weight"]) * np.exp(-np.abs(env_start - start) / tau),
        0.0,
    )
    offset = np.where(
        scored & has_offset,
        float(cfg["offset_weight"]) * np.exp(-np.abs(env_end - end) / tau),
        0.0,
    )
    fragment = np.where(scored, -float(cfg["fragment_weight"]) * (n_fragments - 1), 0.0)
    return {
        "onset": float(onset.sum()),
        "offset": float(offset.sum()),
        "fragment": float(fragment.sum()),
    }


def step_reward_channels(true_state, pred_state, prev_pred_state, edge_w_t):
    edge_weight = float(GLOBAL_CONFIG["rl"]["ppo_reward"]["edge"]["weight"])
    switch_weight = float(GLOBAL_CONFIG["rl"]["ppo_reward"]["switch"]["weight"])
    correct = true_state == pred_state
    r_edge = edge_weight * float(edge_w_t) * (1.0 if correct else -1.0)
    r_coverage = 0.0 if correct else -1.0
    switched = pred_state != prev_pred_state
    r_switch = -switch_weight if (switched and float(edge_w_t) == 0.0) else 0.0
    return r_edge, r_coverage, r_switch


def switch_count(pred_labels):
    pred = np.asarray(pred_labels).astype(np.int8)
    if len(pred) == 0:
        return 0
    prev = np.empty(len(pred), dtype=np.int8)
    prev[0] = 0
    prev[1:] = pred[:-1]
    return int(np.sum(pred != prev))


def episode_reward_breakdown(true_labels, pred_labels, coverage_lambda):
    edge_cfg = GLOBAL_CONFIG["rl"]["ppo_reward"]["edge"]
    switch_weight = float(GLOBAL_CONFIG["rl"]["ppo_reward"]["switch"]["weight"])
    true_states = np.asarray(true_labels).astype(np.int8)
    pred = np.asarray(pred_labels).astype(np.int8)
    n_seconds = len(true_states)
    if n_seconds == 0:
        return {"total": 0.0, "edge": 0.0, "coverage": 0.0, "switch_reward": 0.0,
                "structural": 0.0, "onset": 0.0, "offset": 0.0, "fragment": 0.0,
                "tp": 0, "fp": 0, "fn": 0, "n_switches": 0,
                "n_true_events": 0, "switches_outside_edge": 0}

    correct = pred == true_states
    tp = int(np.sum((pred == 1) & (true_states == 1)))
    fp = int(np.sum((pred == 1) & (true_states == 0)))
    fn = int(np.sum((pred == 0) & (true_states == 1)))

    edge_w = edge_weights(true_states)
    edge = float(float(edge_cfg["weight"]) * np.sum(np.where(correct, edge_w, -edge_w)))
    coverage = float(coverage_lambda * -np.sum(~correct))

    structural = {"onset": 0.0, "offset": 0.0, "fragment": 0.0}
    if GLOBAL_CONFIG["rl"]["ppo_reward"]["structural"]["enabled"]:
        structural = structural_breakdown(true_states, pred)
    structural_total = sum(structural.values())

    prev = np.empty(n_seconds, dtype=np.int8)
    prev[0] = 0
    prev[1:] = pred[:-1]
    flip_mask = pred != prev
    n_switch = int(np.sum(flip_mask))
    switches_outside_edge = int(np.sum(flip_mask & (edge_w == 0.0)))
    switch_reward = float(-switch_weight * switches_outside_edge)

    n_true_events = len(labels_to_intervals(true_states))
    total = edge + coverage + switch_reward + structural_total
    return {
        "total": total,
        "edge": edge,
        "coverage": coverage,
        "switch_reward": switch_reward,
        "structural": structural_total,
        **structural,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_switches": n_switch,
        "n_true_events": n_true_events,
        "switches_outside_edge": switches_outside_edge,
    }

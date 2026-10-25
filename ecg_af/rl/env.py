import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..postprocess import filter_short_intervals
from ..util import GLOBAL_CONFIG
from .metrics import truth_episode_type
from .observations import build_observation, load_controller_arrays
from .reward import StructuralScorer, edge_weights, step_reward_channels


class RecordSampler:
    def __init__(self, record_ids, seed):
        self.record_ids = list(record_ids)
        self.rng = np.random.default_rng(int(seed))
        self.queue = []
        self.coverage_lambda = float(GLOBAL_CONFIG["rl"]["ppo_reward"]["coverage"]["lambda"])

    def draw(self):
        if not self.queue:
            order = self.rng.permutation(len(self.record_ids))
            self.queue = [int(i) for i in order[::-1]]
        return self.record_ids[self.queue.pop()]


class AFControllerEnv(gym.Env):
    def __init__(self, record_ids, sampler):
        super().__init__()
        self.record_ids = list(record_ids)
        self.sampler = sampler
        self.min_duration = int(GLOBAL_CONFIG["postprocessing"]["min_duration_seconds"])

        sample = load_controller_arrays(*self.record_ids[0])["base_features"][0]
        n_features = len(build_observation(sample, 0.0))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        source, record_id = self.sampler.draw()
        self.source = source
        self.record_id = record_id
        arrays = load_controller_arrays(source, record_id)
        self.features = arrays["base_features"]
        self.true_states = filter_short_intervals(arrays["true_states"])
        self.episode_type = truth_episode_type(self.true_states)
        self.edge_w = edge_weights(self.true_states)
        self.structural = StructuralScorer(self.true_states) if GLOBAL_CONFIG["rl"]["ppo_reward"]["structural"]["enabled"] else None
        self.pred_states = np.zeros(len(self.features), dtype=np.float32)
        self.t = 0
        self.pred_state = 0
        self.n_switches = 0
        self.af_run_start = -1
        self.af_run_length = 0
        return build_observation(self.features[0], 0.0), {}

    def action_masks(self):
        locked = (
            self.pred_state == 1
            and self.af_run_start > 0
            and self.af_run_length < self.min_duration
        )
        return np.array([not locked, True])

    def step(self, action):
        step_t = self.t
        pred_state = int(action)
        true_state = int(self.true_states[self.t])
        prev_pred_state = self.pred_state

        r_edge, r_flat, r_switch = step_reward_channels(
            true_state, pred_state, prev_pred_state, self.edge_w[self.t]
        )
        r_coverage = float(self.sampler.coverage_lambda) * r_flat
        r_structural = 0.0
        if self.structural is not None:
            r_structural = self.structural.step(step_t, pred_state, prev_pred_state)
        reward = r_edge + r_coverage + r_switch + r_structural
        if pred_state != prev_pred_state:
            self.n_switches += 1

        self.pred_states[self.t] = float(pred_state)
        if pred_state == 1:
            self.af_run_length = self.af_run_length + 1 if prev_pred_state == 1 else 1
            self.af_run_start = self.af_run_start if prev_pred_state == 1 else step_t
        else:
            self.af_run_start, self.af_run_length = -1, 0
        self.pred_state = pred_state
        self.t += 1
        terminated = self.t >= len(self.features)

        info = {
            "source": self.source,
            "record_id": self.record_id,
            "record_length": len(self.features),
            "episode_type": self.episode_type,
            "pred_state": pred_state,
            "true_state": true_state,
            "r_edge": r_edge,
            "r_coverage": r_coverage,
            "r_switch": r_switch,
            "r_structural": r_structural,
            "record_key": f"{self.source}|{self.record_id}",
            "t": step_t,
        }
        if terminated:
            true_labels = self.true_states.astype(np.int8)
            pred_labels = self.pred_states.astype(np.int8)
            info.update({
                "tp": int(np.sum((pred_labels == 1) & (true_labels == 1))),
                "fp": int(np.sum((pred_labels == 1) & (true_labels == 0))),
                "fn": int(np.sum((pred_labels == 0) & (true_labels == 1))),
                "n_switches": self.n_switches,
            })
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = build_observation(
                self.features[self.t],
                float(self.pred_state),
            )

        return obs, reward, terminated, False, info

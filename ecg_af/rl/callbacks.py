import time

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from ..util import GLOBAL_CONFIG
from .evaluations import (
    baseline_reference,
    controller_checkpoint_dir,
    evaluate_record_set,
    reward_metrics_from_rollouts,
)


class PassMetricsCallback(BaseCallback):
    ROLLOUT_DIAGNOSTIC_KEYS = (
        "buf_mean_abs_value",
        "buf_adv_on_edge",
        "buf_adv_on_flat",
        "buf_frac_reward_nonzero",
    )

    def __init__(self, phase_name, train_records, val_records, sampler):
        super().__init__()
        self.rollout_diagnostics = {k: np.nan for k in self.ROLLOUT_DIAGNOSTIC_KEYS}
        self.phase_name = phase_name
        self.train_records = list(train_records)
        self.val_records = list(val_records)
        self.sampler = sampler
        self.pass_history = []
        self.pass_count = 0
        self.episode_count = 0
        self.baseline = None
        self.best = {"score": -np.inf, "pass": None}
        self.pass_start_time = None

    def _on_training_start(self):
        self.pass_start_time = time.perf_counter()

    def ensure_baseline(self):
        if self.baseline is not None:
            return
        val_baseline = (
            baseline_reference(self.val_records)
            if self.val_records
            else {"f1": np.nan, "mean_switches": np.nan}
        )
        self.baseline = {
            "train": baseline_reference(self.train_records),
            "val": val_baseline,
        }
        GLOBAL_CONFIG.logger.info(
            "Baseline reference | train f1=%.4f sw=%.1f | val f1=%.4f sw=%.1f",
            self.baseline["train"]["f1"],
            self.baseline["train"]["mean_switches"],
            self.baseline["val"]["f1"],
            self.baseline["val"]["mean_switches"],
        )

    def _on_rollout_end(self):
        buffer = self.model.rollout_buffer
        advantages = np.abs(buffer.advantages.ravel())
        rewards = buffer.rewards.ravel()
        edge = rewards != 0.0
        self.rollout_diagnostics = {
            "buf_mean_abs_value": float(np.abs(buffer.values).mean()),
            "buf_adv_on_edge": float(advantages[edge].mean()) if edge.any() else np.nan,
            "buf_adv_on_flat": float(advantages[~edge].mean()) if (~edge).any() else np.nan,
            "buf_frac_reward_nonzero": float(edge.mean()),
        }

    def _on_step(self):
        train_cfg = GLOBAL_CONFIG["rl"]["train"]
        for done in self.locals["dones"]:
            if not done:
                continue
            self.episode_count += 1
            if self.episode_count % len(self.train_records) != 0:
                continue
            self.pass_count += 1
            pass_seconds = time.perf_counter() - self.pass_start_time
            gamma0_passes = int(train_cfg[self.phase_name]["gamma0_passes"])
            is_final = self.pass_count >= gamma0_passes + int(train_cfg[self.phase_name]["n_passes"])
            eval_every = int(train_cfg["eval_every_n_passes"])
            if is_final or self.pass_count % eval_every == 0:
                self.record_pass(pass_seconds)
            if self.pass_count == gamma0_passes:
                self.model.gamma = self.model.rollout_buffer.gamma = float(train_cfg["gamma"])
                GLOBAL_CONFIG.logger.info(
                    "Phase %s: gamma 0 -> %g after pass %d",
                    self.phase_name, self.model.gamma, self.pass_count,
                )
            self.pass_start_time = time.perf_counter()
            if is_final:
                return False
        return True

    def update_multipliers(self, train_f1):
        reward_cfg = GLOBAL_CONFIG["rl"]["ppo_reward"]
        tol = float(reward_cfg["coverage"]["low"])

        cov = reward_cfg["coverage"]["constraint"]
        f1_floor = self.baseline["train"]["f1"] - tol
        lam_cov = self.sampler.coverage_lambda + float(cov["lr"]) * (f1_floor - train_f1)
        self.sampler.coverage_lambda = float(np.clip(lam_cov, float(cov["min"]), float(cov["max"])))

    @staticmethod
    def selection_score(metric, metrics):
        if metric not in metrics:
            choices = ", ".join(sorted(metrics))
            raise ValueError(
                f"Unknown RL selection_metric={metric!r}; expected one of: {choices}"
            )
        score = float(metrics[metric])
        return score if np.isfinite(score) else -np.inf

    def training_diagnostics(self):
        keys = {
            "ppo_approx_kl": "train/approx_kl",
            "ppo_clip_fraction": "train/clip_fraction",
            "ppo_entropy_loss": "train/entropy_loss",
            "ppo_policy_gradient_loss": "train/policy_gradient_loss",
            "ppo_value_loss": "train/value_loss",
            "ppo_explained_variance": "train/explained_variance",
        }
        logger = getattr(self.model, "_logger", None)
        values = {} if logger is None else logger.name_to_value
        return {
            name: float(values.get(logger_key, np.nan))
            for name, logger_key in keys.items()
        }

    def record_pass(self, pass_seconds):
        self.ensure_baseline()
        train_cfg = GLOBAL_CONFIG["rl"]["train"]
        policy_key = f"{self.phase_name}/pass_{self.pass_count:03d}"
        if self.val_records:
            val_rollouts, val_second = evaluate_record_set(
                self.model, self.val_records, policy_key
            )
        else:
            val_rollouts = {}
            val_second = {"total": {"f1": np.nan}}
        train_rollouts, train_second = evaluate_record_set(
            self.model, self.train_records, policy_key
        )
        coverage_lambda = float(self.sampler.coverage_lambda)
        _, val_reward = reward_metrics_from_rollouts(val_rollouts, coverage_lambda)
        _, train_reward = reward_metrics_from_rollouts(train_rollouts, coverage_lambda)
        f1 = float(val_second["total"]["f1"])
        train_f1 = float(train_second["total"]["f1"])
        train_switches = float(train_reward["n_switches"])

        self.update_multipliers(train_f1)
        selection_metric = str(train_cfg["selection_metric"])
        selection_values = {
            "train_reward_total": train_reward["reward_total"],
            "val_reward_total": val_reward["reward_total"],
            "train_f1": train_f1,
            "val_f1": f1,
        }
        score = self.selection_score(selection_metric, selection_values)
        ignore_pass0 = bool(GLOBAL_CONFIG["rl"]["train"]["ignore_pass0"])
        # Gamma-0 passes are never selected; the switch pass is the gamma stage's pass 0.
        first_selectable = int(train_cfg[self.phase_name]["gamma0_passes"]) + int(ignore_pass0)
        selectable = self.pass_count >= first_selectable
        if self.val_records:
            is_best = selectable and score > self.best["score"]
        else:
            is_best = selectable
        checkpoint_path = controller_checkpoint_dir() / f"{self.phase_name}_best.zip"
        if is_best:
            self.best = {"score": score, "pass": self.pass_count}
            self.model.save(checkpoint_path)

        diagnostics = self.training_diagnostics()
        self.pass_history.append({
            "pass": self.pass_count,
            "gamma": float(self.model.gamma),
            "selection_metric": selection_metric,
            "selection_score": score,
            "is_best": is_best,
            "val_f1": f1,
            "train_f1": train_f1,
            "val_reward_total": val_reward["reward_total"],
            "val_reward_edge": val_reward["reward_edge"],
            "val_reward_coverage": val_reward["reward_coverage"],
            "val_reward_switch": val_reward["reward_switch"],
            "val_reward_structural": val_reward["reward_structural"],
            "val_reward_onset": val_reward["reward_onset"],
            "val_reward_offset": val_reward["reward_offset"],
            "val_reward_fragment": val_reward["reward_fragment"],
            "val_n_switches": val_reward["n_switches"],
            "val_switches_outside_edge": val_reward["n_switches_outside_edge"],
            "train_reward_total": train_reward["reward_total"],
            "train_reward_edge": train_reward["reward_edge"],
            "train_reward_coverage": train_reward["reward_coverage"],
            "train_reward_switch": train_reward["reward_switch"],
            "train_reward_structural": train_reward["reward_structural"],
            "train_reward_onset": train_reward["reward_onset"],
            "train_reward_offset": train_reward["reward_offset"],
            "train_reward_fragment": train_reward["reward_fragment"],
            "train_n_switches": train_switches,
            "train_switches_outside_edge": train_reward["n_switches_outside_edge"],
            "baseline_train_f1": self.baseline["train"]["f1"],
            "baseline_train_switches": self.baseline["train"]["mean_switches"],
            **diagnostics,
            **self.rollout_diagnostics,
        })
        phase_cfg = train_cfg[self.phase_name]
        n_passes = int(phase_cfg["gamma0_passes"]) + int(phase_cfg["n_passes"])
        lam_cov = float(self.sampler.coverage_lambda)
        GLOBAL_CONFIG.logger.info(
            "Completed pass %d/%d (%d episodes, %d timesteps, %.1fs) | "
            "val f1=%.4f %s=%.4f%s λcov=%.2f | "
            "val reward=%.1f edge=%.1f cov=%.1f sw=%.1f nsw=%.1f osw=%.1f "
            "str=%.2f(on %.2f off %.2f frg %.2f) | "
            "train f1=%.4f reward=%.1f edge=%.1f cov=%.1f sw=%.1f nsw=%.1f osw=%.1f "
            "str=%.2f(on %.2f off %.2f frg %.2f) | "
            "|V|=%.3f adv edge/flat=%.3f/%.3f ev=%.2f | saved %s",
            self.pass_count,
            n_passes,
            self.episode_count,
            self.num_timesteps,
            pass_seconds,
            f1,
            selection_metric,
            score,
            " [best]" if is_best else "",
            lam_cov,
            val_reward["reward_total"],
            val_reward["reward_edge"],
            val_reward["reward_coverage"],
            val_reward["reward_switch"],
            val_reward["n_switches"],
            val_reward["n_switches_outside_edge"],
            val_reward["reward_structural"],
            val_reward["reward_onset"],
            val_reward["reward_offset"],
            val_reward["reward_fragment"],
            train_f1,
            train_reward["reward_total"],
            train_reward["reward_edge"],
            train_reward["reward_coverage"],
            train_reward["reward_switch"],
            train_reward["n_switches"],
            train_reward["n_switches_outside_edge"],
            train_reward["reward_structural"],
            train_reward["reward_onset"],
            train_reward["reward_offset"],
            train_reward["reward_fragment"],
            self.rollout_diagnostics["buf_mean_abs_value"],
            self.rollout_diagnostics["buf_adv_on_edge"],
            self.rollout_diagnostics["buf_adv_on_flat"],
            diagnostics["ppo_explained_variance"],
            checkpoint_path if is_best else "-",
        )

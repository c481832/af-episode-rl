import shutil

import numpy as np
import pandas as pd
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import FloatSchedule
from stable_baselines3.common.vec_env import DummyVecEnv

from ..util import GLOBAL_CONFIG
from .callbacks import PassMetricsCallback
from .evaluations import controller_checkpoint_dir
from .observations import load_controller_arrays, observation_names
from .env import AFControllerEnv, RecordSampler


def make_controller_env(record_ids, sampler):
    """The env with the exit action masked until an AF run reaches min_duration."""
    env = AFControllerEnv(record_ids, sampler)
    return Monitor(ActionMasker(env, lambda e: e.unwrapped.action_masks()))


def load_ppo(path, *, env, device):
    return MaskablePPO.load(
        path,
        env=env,
        device=device,
        custom_objects={"learning_rate": 0.0, "lr_schedule": FloatSchedule(0.0)},
    )


def apply_threshold_warm_start(policy, warm_cfg, phase_name):
    """Initialize the policy as a single-feature threshold on the segment model's prob."""
    a = float(warm_cfg["a"])
    k = float(warm_cfg["k"])
    net = policy.policy
    linears = [m for m in net.mlp_extractor.policy_net if isinstance(m, torch.nn.Linear)]
    segment_model = GLOBAL_CONFIG["rl"]["base_models"]["segment"]
    seg = observation_names().index(f"{segment_model}_p")
    with torch.no_grad():
        lin1 = linears[0]
        action_net = net.action_net
        lin1.weight[0].zero_()
        lin1.weight[0, seg] = a
        lin1.bias[0] = -a * 0.5
        action_net.weight.zero_()
        action_net.bias.zero_()
        action_net.weight[1, 0] = k / 2.0
        action_net.weight[0, 0] = -k / 2.0
    GLOBAL_CONFIG.logger.info(
        "Phase %s: threshold warm-start on %s_p (a=%g, k=%g)",
        phase_name, segment_model, a, k,
    )


def train_controller_phase(phase_name, train_record_ids, val_record_ids, init_from):
    seed = int(GLOBAL_CONFIG["data"]["seed"])
    base_cfg = GLOBAL_CONFIG["rl"]["train"]
    phase_cfg = base_cfg[phase_name]
    device = base_cfg["device"]
    ckpt_dir = controller_checkpoint_dir()
    checkpoint_path = ckpt_dir / f"{phase_name}_best.zip"

    phase_dir = ckpt_dir / phase_name
    phase_dir.mkdir(parents=True, exist_ok=True)
    for cache_name in (phase_name, f"{phase_name}_best"):
        cache_dir = (ckpt_dir / "rollouts" / cache_name).resolve()
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            GLOBAL_CONFIG.logger.info("Cleared rollout cache %s", cache_dir)

    steps_per_pass = sum(
        len(load_controller_arrays(src, rid)["base_features"])
        for src, rid in train_record_ids
    )
    n_passes = int(phase_cfg["n_passes"])
    # The first gamma0_passes run at gamma 0; PassMetricsCallback then switches to `gamma`.
    gamma0_passes = int(phase_cfg["gamma0_passes"])
    total_timesteps = steps_per_pass * (gamma0_passes + n_passes + 1)
    learning_rate = float(phase_cfg["learning_rate"])
    ent_coef = float(phase_cfg["ent_coef"])
    n_epochs = int(phase_cfg["n_epochs"])
    target_kl = float(base_cfg["target_kl"])
    normalize_advantage = bool(base_cfg["normalize_advantage"])
    vf_coef = float(base_cfg["vf_coef"])

    warm_cfg = base_cfg["warm_start"]
    use_warm_start = init_from is False and bool(warm_cfg["enabled"])
    lr_decay = base_cfg["lr_decay"]
    if lr_decay == "linear":
        lr_arg = lambda progress_remaining: float(progress_remaining) * learning_rate
    elif lr_decay == "none":
        lr_arg = learning_rate
    else:
        raise ValueError(f"Unknown lr_decay={lr_decay!r}; expected 'linear' or 'none'")

    n_envs = int(base_cfg["n_envs"])
    sampler = RecordSampler(train_record_ids, seed)
    gamma = 0.0 if gamma0_passes > 0 else float(base_cfg["gamma"])
    gae_lambda = float(base_cfg["gae_lambda"])
    if n_envs == 1:
        env = make_controller_env(train_record_ids, sampler)
    else:
        GLOBAL_CONFIG.logger.info("Using %d in-process environments", n_envs)
        env = DummyVecEnv(
            [lambda: make_controller_env(train_record_ids, sampler) for _ in range(n_envs)]
        )

    if init_from is not False:
        GLOBAL_CONFIG.logger.info("Phase %s: warm-starting from %s", phase_name, init_from)
        policy = load_ppo(init_from, env=env, device=device)
        policy.learning_rate = lr_arg
        policy.lr_schedule = FloatSchedule(lr_arg)
        policy.ent_coef = ent_coef
        policy.n_epochs = n_epochs
        policy.target_kl = target_kl
        policy.normalize_advantage = normalize_advantage
        policy.vf_coef = vf_coef
        # A checkpoint keeps the discount it was trained with; this phase's gamma wins.
        policy.gamma = policy.rollout_buffer.gamma = gamma
    else:
        policy = MaskablePPO(
            base_cfg["policy"],
            env,
            policy_kwargs=dict(net_arch=dict(
                pi=list(base_cfg["net_arch"]), vf=list(base_cfg["vf_arch"])
            )),
            learning_rate=lr_arg,
            n_steps=int(base_cfg["n_steps"]),
            batch_size=int(base_cfg["batch_size"]),
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=float(base_cfg["clip_range"]),
            target_kl=target_kl,
            ent_coef=ent_coef,
            normalize_advantage=normalize_advantage,
            vf_coef=vf_coef,
            seed=seed,
            device=device,
            verbose=GLOBAL_CONFIG.debug_print,
        )
        if use_warm_start:
            apply_threshold_warm_start(policy, warm_cfg, phase_name)

    GLOBAL_CONFIG.logger.info(
        "Phase %s: %d passes, %d steps/pass, %d total timesteps, "
        "lr=%g (%s) ent=%g n_epochs=%d target_kl=%s norm_adv=%s vf_coef=%g "
        "gamma=%g gae_lambda=%g vf_arch=%s, train=%d val=%d, device=%s",
        phase_name, gamma0_passes + n_passes, steps_per_pass, total_timesteps,
        learning_rate, lr_decay, ent_coef, n_epochs, target_kl, normalize_advantage, vf_coef,
        gamma, gae_lambda, list(base_cfg["vf_arch"]),
        len(train_record_ids), len(val_record_ids), device,
    )
    callback = PassMetricsCallback(phase_name, train_record_ids, val_record_ids, sampler)
    callback.model = policy
    callback.record_pass(0.0)
    policy.learn(total_timesteps=total_timesteps, callback=callback)
    history_path = phase_dir / "pass_history.csv"
    pd.DataFrame(callback.pass_history).to_csv(history_path, index=False)
    GLOBAL_CONFIG.logger.info("Saved pass history to %s", history_path)

    if not checkpoint_path.exists():
        raise RuntimeError(f"Phase {phase_name} did not produce a checkpoint")
    GLOBAL_CONFIG.logger.info(
        "Phase %s best pass: %s (%s=%.4f); loading %s",
        phase_name,
        callback.best["pass"],
        base_cfg["selection_metric"],
        callback.best["score"],
        checkpoint_path,
    )
    policy = load_ppo(checkpoint_path, env=env, device=device)
    return policy, callback.pass_history, checkpoint_path

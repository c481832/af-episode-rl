import numpy as np
import torch

from ..data.common import processed_record_path
from ..util import GLOBAL_CONFIG
from .observations import build_observation, load_controller_arrays, observation_names


@torch.no_grad()
def policy_af_probs(policy, batch):
    obs_t = torch.as_tensor(batch, device=policy.device)
    features = policy.policy.extract_features(obs_t, policy.policy.pi_features_extractor)
    latent_pi = policy.policy.mlp_extractor.forward_actor(features)
    logits = policy.policy.action_net(latent_pi)
    return torch.softmax(logits, dim=1)[:, 1].cpu().numpy()


def rollout_policy(policy, record_ids):
    if not record_ids:
        return {}

    policy.policy.set_training_mode(False)

    records = []
    for source, record_id in record_ids:
        arrays = load_controller_arrays(source, record_id)
        features = arrays["base_features"]
        n = len(features)
        records.append(
            {
                "key": (source, record_id),
                "features": features,
                "true": arrays["true_states"].astype(np.float32),
                "n": n,
                "pred": np.zeros(n, dtype=np.float32),
                "probs": np.empty(n, dtype=np.float32),
            }
        )

    n_records = len(records)
    n_obs = len(build_observation(records[0]["features"][0], 0.0))
    lengths = np.array([r["n"] for r in records])
    max_n = int(lengths.max())

    prev_state = np.zeros(n_records, dtype=np.float32)
    min_duration = int(GLOBAL_CONFIG["postprocessing"]["min_duration_seconds"])
    af_run_start = np.full(n_records, -1, dtype=np.int64)
    af_run_length = np.zeros(n_records, dtype=np.int64)

    for t in range(max_n):
        active = np.nonzero(lengths > t)[0]
        batch = np.empty((len(active), n_obs), dtype=np.float32)
        for j, ri in enumerate(active):
            batch[j] = build_observation(
                records[ri]["features"][t],
                prev_state[ri],
            )

        probs_af = policy_af_probs(policy, batch)
        actions = (probs_af > 0.5).astype(np.float32)
        locked = (
            (prev_state[active] == 1)
            & (af_run_start[active] > 0)
            & (af_run_length[active] < min_duration)
        )
        actions = np.where(locked, 1.0, actions)

        for j, ri in enumerate(active):
            records[ri]["probs"][t] = probs_af[j]
            records[ri]["pred"][t] = actions[j]

        in_af = actions == 1
        continuing = in_af & (prev_state[active] == 1)
        af_run_length[active] = np.where(in_af, np.where(continuing, af_run_length[active] + 1, 1), 0)
        af_run_start[active] = np.where(in_af, np.where(continuing, af_run_start[active], t), -1)
        prev_state[active] = actions

    out = {}
    for r in records:
        out[r["key"]] = (r["true"], r["pred"], r["probs"])
    return out


def training_median_heart_rate_bpm(records):
    fs = float(GLOBAL_CONFIG["data"]["preprocessing"]["target_fs"])
    hr_cfg = GLOBAL_CONFIG["rl"]["observation_fields"]["heart_rate"]
    lo, hi = float(hr_cfg["hr_lo"]), float(hr_cfg["hr_hi"])
    heart_rates = []
    for source, record_id in records:
        path = processed_record_path(source, record_id)
        if not path.exists():
            raise FileNotFoundError(f"Missing processed training record: {path}")
        with np.load(path, allow_pickle=False) as data:
            peaks = np.asarray(data["processed_r_peaks"], dtype=np.int64)
        rr_seconds = np.diff(peaks) / fs
        if len(rr_seconds):
            heart_rates.append(np.clip(60.0 / rr_seconds, lo, hi))
    if not heart_rates:
        raise RuntimeError("No training RR intervals were available")
    return float(np.median(np.concatenate(heart_rates)))


def flat_profile_observations(probabilities, heart_rate, last_state):
    names = observation_names()
    probability_columns = [i for i, name in enumerate(names) if name.endswith("_p")]
    if len(probability_columns) != 1:
        raise RuntimeError(f"Expected one detector probability in {names}")
    batch = np.zeros((len(probabilities), len(names)), dtype=np.float32)
    for column, name in enumerate(names):
        if column in probability_columns:
            batch[:, column] = probabilities
        elif name == "heart_rate":
            batch[:, column] = heart_rate
        elif name == "last_state":
            batch[:, column] = last_state
        elif "_p_d+" not in name:
            raise RuntimeError(f"No flat-profile value for observation {name!r}")
    return batch


def decision_crossing(x, y, threshold=0.5):
    indices = np.flatnonzero((y[:-1] < threshold) & (y[1:] >= threshold))
    if not len(indices):
        return None
    i = int(indices[0])
    return float(x[i] + (threshold - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))


def flat_profile_response(policy, train_records):
    median_bpm = training_median_heart_rate_bpm(train_records)
    hr_cfg = GLOBAL_CONFIG["rl"]["observation_fields"]["heart_rate"]
    hr_lo, hr_hi = float(hr_cfg["hr_lo"]), float(hr_cfg["hr_hi"])
    median_hr = (median_bpm - hr_lo) / (hr_hi - hr_lo)

    policy.policy.set_training_mode(False)
    probabilities = np.linspace(0.0, 1.0, 1001, dtype=np.float32)
    curves = {
        last_state: policy_af_probs(
            policy, flat_profile_observations(probabilities, median_hr, last_state)
        )
        for last_state in (0, 1)
    }
    return {
        "probabilities": probabilities,
        "curves": curves,
        "enter": decision_crossing(probabilities, curves[0]),
        "remain": decision_crossing(probabilities, curves[1]),
        "median_bpm": median_bpm,
        "median_hr": median_hr,
    }

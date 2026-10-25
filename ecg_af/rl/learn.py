import pandas as pd

from ..util import GLOBAL_CONFIG
from .evaluations import controller_checkpoint_dir, evaluate_record_set
from .observations import load_controller_arrays
from .ppo import load_ppo, train_controller_phase


def filter_trainable_records(record_ids):
    return [
        (source, record_id)
        for source, record_id in record_ids
        if len(load_controller_arrays(source, record_id)["base_features"]) > 0
    ]


def controller_phase1_splits(layer1, layer2, run_phase2):
    train_ids = list(layer1["train"])
    val_ids = list(layer1["val"])
    if run_phase2:
        test_ids = list(layer2["test"])
    else:
        test_ids = list(layer1["test"])
    return train_ids, val_ids, test_ids


def resolve_phase_best(phase_name):
    if phase_name not in {"phase1", "phase2"}:
        raise ValueError(f"Unknown controller phase: {phase_name!r}")
    path = controller_checkpoint_dir() / f"{phase_name}_best.zip"
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint for {phase_name}: {path}")
    return path


def fit_controller(layer1, layer2, load_checkpoint_phase1, load_checkpoint_phase2):
    checkpoint_dir = controller_checkpoint_dir()
    device = GLOBAL_CONFIG["rl"]["train"]["device"]
    run_phase2 = bool(GLOBAL_CONFIG["data"]["split"]["layer2"]["train_sources"])

    layer1 = {split: filter_trainable_records(records) for split, records in layer1.items()}
    layer2 = {split: filter_trainable_records(records) for split, records in layer2.items()}
    phase1_train, phase1_val, test_record_ids = controller_phase1_splits(
        layer1, layer2, run_phase2
    )
    GLOBAL_CONFIG.logger.info(
        "Phase splits: phase-1 train=%d val=%d, test=%d (phase2=%s)",
        len(phase1_train), len(phase1_val), len(test_record_ids), run_phase2,
    )

    if load_checkpoint_phase1:
        phase1_best = resolve_phase_best("phase1")
        phase1_history_path = checkpoint_dir / "phase1" / "pass_history.csv"
        phase1_history = (
            []
            if not phase1_history_path.exists()
            else pd.read_csv(phase1_history_path).to_dict("records")
        )
        GLOBAL_CONFIG.logger.info("Reloading phase-1 controller from %s", phase1_best)
    else:
        policy, phase1_history, phase1_best = train_controller_phase(
            "phase1", phase1_train, phase1_val, init_from=False
        )

    if run_phase2:
        if load_checkpoint_phase2:
            phase2_best = resolve_phase_best("phase2")
            policy = load_ppo(phase2_best, env=None, device=device)
            phase2_history_path = checkpoint_dir / "phase2" / "pass_history.csv"
            phase2_history = (
                []
                if not phase2_history_path.exists()
                else pd.read_csv(phase2_history_path).to_dict("records")
            )
            GLOBAL_CONFIG.logger.info("Reloading phase-2 controller from %s", phase2_best)
        else:
            policy, phase2_history, phase2_best = train_controller_phase(
                "phase2", list(layer2["train"]), list(layer2["val"]), init_from=phase1_best
            )
    else:
        phase2_history = []
        policy = load_ppo(phase1_best, env=None, device=device)

    phase_histories = [("phase1", phase1_history)]
    if run_phase2:
        phase_histories.append(("phase2", phase2_history))
    rows = []
    for phase_name, history in phase_histories:
        for row in history:
            tagged = {"phase": phase_name}
            tagged.update(row)
            rows.append(tagged)
    pass_history_df = pd.DataFrame(rows)

    if not test_record_ids:
        return policy, {}, None, pass_history_df, None

    policy_key = "phase2_best" if run_phase2 else "phase1_best"
    controller_rollouts, second_metrics = evaluate_record_set(
        policy, test_record_ids, policy_key
    )
    second_df = pd.DataFrame(second_metrics)

    if run_phase2:
        phase1_policy = load_ppo(phase1_best, env=None, device=device)
        _, p1_second = evaluate_record_set(
            phase1_policy,
            test_record_ids,
            "phase1_best",
        )
        phase_test_df = pd.DataFrame(
            {
                "phase1_warmstart": {
                    "duration_f1": p1_second["total"]["f1"],
                },
                "phase2_best": {
                    "duration_f1": second_metrics["total"]["f1"],
                },
            }
        ).T
    else:
        phase_test_df = pd.DataFrame(
            {
                "phase1_best": {
                    "duration_f1": second_metrics["total"]["f1"],
                },
            }
        ).T

    return policy, controller_rollouts, second_df, pass_history_df, phase_test_df

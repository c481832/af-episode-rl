"""Render Figure 7: the controller's flat-profile response of paper section 7.11.

Loads the notebook's reported controller (runs/foundation/controller/phase1_best.zip)
and draws it with the same probe and plot the notebook uses.

Run: uv run python scripts/render_policy_response_figure.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ecg_af.data.common import load_records_list, split_records_two_layer
from ecg_af.rl.learn import resolve_phase_best
from ecg_af.rl.policy import flat_profile_response
from ecg_af.rl.ppo import load_ppo
from ecg_af.util import GLOBAL_CONFIG
from ecg_af.visualization import plot_flat_profile_response

EXPERIMENT = "foundation"
OUT_PATH = Path(__file__).resolve().parents[1] / "plots" / "figure_07_policy_behavior_last_state.png"


def main():
    GLOBAL_CONFIG.load_config(EXPERIMENT)
    GLOBAL_CONFIG.name = EXPERIMENT
    train_records = split_records_two_layer(load_records_list())[0]["layer1"]["train"]
    policy = load_ppo(resolve_phase_best("phase1"), env=None, device=GLOBAL_CONFIG["rl"]["train"]["device"])

    response = flat_profile_response(policy, train_records)
    # Importing ecg_af switches on a seaborn style; the paper figure uses matplotlib's default.
    plt.style.use("default")
    plot_flat_profile_response(response, "ECGFounder")
    plt.gcf().savefig(OUT_PATH, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.1)
    print(f"enter={response['enter']:.4f} remain={response['remain']:.4f} median_bpm={response['median_bpm']:.1f}")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

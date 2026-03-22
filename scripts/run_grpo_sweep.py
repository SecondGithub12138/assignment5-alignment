"""
Run 8 GRPO experiments as specified in the sweep plan.

Part 1 (E1-E4): loss_type, loss_aggregation, use_std_normalization
Part 2 (C1-C4): epochs_per_rollout_batch, train_batch_size

Each experiment: 40 steps, then eval. Based on train_eval.py.
"""
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.flag_util import get_config

SCRIPTS = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS.parent
RESULTS = PROJECT_ROOT / "results"
CONFIGS = PROJECT_ROOT / "scripts" / "config_sweep"
CONFIGS.mkdir(parents=True, exist_ok=True)

DATA_SET_SIZE = 2  # for grpo

# Base config from config_grpo.json
BASE_CONFIG = get_config("grpo")

# Part 1: E1-E4 - loss_type, loss_aggregation, use_std_normalization
# E1: no_baseline + masked_mean + use_std_normalization=True
# E2: reinforce_with_baseline + masked_mean + use_std_normalization=True
# E3: reinforce_with_baseline + masked_normalize + use_std_normalization=True
# E4: reinforce_with_baseline + masked_mean + use_std_normalization=False
EXPERIMENTS_PART1 = [
    # ("E1", {"loss_type": "no_baseline", "loss_aggregation": "masked_mean", "use_std_normalization": True}),
    # ("E2", {"loss_type": "reinforce_with_baseline", "loss_aggregation": "masked_mean", "use_std_normalization": True}),
    # ("E3", {"loss_type": "reinforce_with_baseline", "loss_aggregation": "masked_normalize", "use_std_normalization": True}),
    # ("E4", {"loss_type": "reinforce_with_baseline", "loss_aggregation": "masked_mean", "use_std_normalization": False}),
]

# Part 2: C1-C4 - epochs_per_rollout_batch, train_batch_size
# C1: epochs=1, train_batch=256 (1 update)
# C2: epochs=1, train_batch=128 (2 updates)
# C3: epochs=2, train_batch=256 (2 updates)
# C4: epochs=2, train_batch=128 (4 updates)
# Note: train_batch_size=128 requires grad_accum to divide 128. Base has grad_accum=128, so 128/128=1 (micro_batch=1)
EXPERIMENTS_PART2 = [
    ("C1", {"epochs_per_rollout_batch": 1, "train_batch_size": 256}),
    ("C2", {"epochs_per_rollout_batch": 1, "train_batch_size": 128}),
    ("C3", {"epochs_per_rollout_batch": 2, "train_batch_size": 256}),
    ("C4", {"epochs_per_rollout_batch": 2, "train_batch_size": 128}),
]

ALL_EXPERIMENTS = EXPERIMENTS_PART1 + EXPERIMENTS_PART2


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_log = RESULTS / f"run_grpo_sweep_{timestamp}.log"
    run_configs_dir = CONFIGS / f"grpo_sweep_{timestamp}"
    RESULTS.mkdir(parents=True, exist_ok=True)
    run_configs_dir.mkdir(parents=True, exist_ok=True)
    print(f"Sweep log: {sweep_log}")
    print(f"Sweep configs: {run_configs_dir}")

    for exp_id, overrides in ALL_EXPERIMENTS:
        config = {**BASE_CONFIG, **overrides}
        config_path = (run_configs_dir / f"config_grpo_{exp_id}.json").resolve()
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

        exp_log = RESULTS / f"run_grpo_{exp_id}_{timestamp}.log"
        print(f"\n{'='*60}\n[Experiment {exp_id}] {overrides}\n{'='*60}")

        with open(exp_log, "w") as f:
            f.write(f"[Experiment {exp_id}] config overrides: {overrides}\n")
            f.write(f"Config: {config_path}\n\n")

        with open(exp_log, "a") as f:
            # Train
            train_cmd = [
                sys.executable,
                str(SCRIPTS / "run_grpo_experiment.py"),
                "--data_size", str(DATA_SET_SIZE),
                "--exp_id", exp_id,
                "--config_path", str(config_path),
            ]
            subprocess.run(train_cmd, stdout=f, stderr=f, check=True)

            # Eval
            eval_cmd = [
                sys.executable,
                str(SCRIPTS / "run_eval.py"),
                "--data_size", str(DATA_SET_SIZE),
                "--train_type", "grpo",
                "--exp_id", exp_id,
                "--validation_step", str(config["n_grpo_steps"]),
            ]
            subprocess.run(eval_cmd, stdout=f, stderr=f, check=True)

        print(f"  Done. Log: {exp_log}")

    print(f"\nAll 8 experiments finished. Sweep log: {sweep_log}")


if __name__ == "__main__":
    main()

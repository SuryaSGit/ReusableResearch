"""
Ablation runner: sweeps (arm x trial) and dumps all logs to disk as JSON
and CSV.  No pandas dependency — the CSV is hand-written so it's
diff-friendly in git.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional

from residual_rl.cht_prior import ArmConfig, CANONICAL_ARMS
from residual_rl.trainer   import TrainerConfig, TrainLog, train_one_arm
from residual_rl.residual_dqn import ResidualHP


def run_ablation(
    arms:                List[ArmConfig],
    env_factory:         Callable[[int], Any],
    prior_factory:       Callable[[Any, ArmConfig], Any],
    obs_builder_factory: Callable[[Any], Any],
    episode_runner:      Callable,
    trainer_cfg:         TrainerConfig,
    n_trials:            int,
    out_dir:             str,
    hp:                  Optional[ResidualHP] = None,
    verbose:             bool = True,
) -> List[TrainLog]:
    os.makedirs(out_dir, exist_ok=True)

    all_logs: List[TrainLog] = []
    for arm in arms:
        for trial in range(n_trials):
            if verbose:
                print(f"\n=== arm={arm.name}  trial={trial}  eps={trainer_cfg.n_episodes} ===")
            logs, _ = train_one_arm(
                arm                 = arm,
                env_factory         = env_factory,
                prior_factory       = prior_factory,
                obs_builder_factory = obs_builder_factory,
                episode_runner      = episode_runner,
                trainer_cfg         = trainer_cfg,
                trial               = trial,
                hp                  = hp,
                verbose             = verbose,
            )
            all_logs.extend(logs)
            # Write incremental output in case later arms crash
            _dump(all_logs, out_dir)
    _dump(all_logs, out_dir)
    return all_logs


def _dump(logs: List[TrainLog], out_dir: str) -> None:
    rows = [asdict(l) for l in logs]
    with open(os.path.join(out_dir, "logs.json"), "w") as f:
        json.dump(rows, f, indent=2)
    if rows:
        with open(os.path.join(out_dir, "logs.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


def arms_from_strings(arm_names: List[str]) -> List[ArmConfig]:
    """'A,B,D' -> [ARM_A, ARM_B, ARM_D]."""
    return [CANONICAL_ARMS[n.strip().upper()] for n in arm_names]

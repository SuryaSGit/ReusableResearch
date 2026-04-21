"""
Ablation runner: sweeps (arm x trial), computes baselines once per env,
samples a probe set once per experiment, and dumps rich per-arm/per-trial
logs to disk.

Directory layout written under `out_dir`:

    <out_dir>/
        hyperparameters.json         (dumped by the caller — see experiments/)
        baselines.json               (baselines dict, one-shot)
        probe_set.npz                (probe_feats + probe_info count)
        logs.json                    (aggregate rows across arms/trials)
        logs.csv                     (same as logs.json, flat)
        learning_curves.png
        q_magnitudes.png
        <arm_name>/
            trial_<n>/
                log.jsonl            (per-row richer schema)
                log.csv              (same, flat)
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from residual_rl.cht_prior     import ArmConfig, CANONICAL_ARMS
from residual_rl.residual_dqn  import ResidualHP
from residual_rl.trainer       import (
    TrainerConfig, TrainLog, train_one_arm, build_probe_set,
    format_header, format_baseline_row,
)


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
    baselines:           Optional[Dict[str, Any]] = None,
    probe_feats:         Optional[np.ndarray] = None,
    probe_infos:         Optional[List[Dict[str, Any]]] = None,
) -> List[TrainLog]:
    """
    Run the sweep. Callers should pass `baselines` + `probe_feats/infos`
    (usually built from `collect_*_baselines` and `build_probe_set` in
    the experiment script) so the per-row numbers are comparable.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Dump baselines once
    if baselines is not None:
        with open(os.path.join(out_dir, "baselines.json"), "w") as f:
            # nested-dict with some numpy floats — coerce via default=float
            json.dump(baselines, f, indent=2, default=float)

    # Dump probe set shape marker
    if probe_feats is not None:
        np.savez(os.path.join(out_dir, "probe_set.npz"),
                 feats=probe_feats, n_infos=np.array([len(probe_infos or [])]))

    all_logs: List[TrainLog] = []
    all_rows: List[Dict[str, Any]] = []

    header_printed = False
    for arm in arms:
        for trial in range(n_trials):
            if verbose:
                print(f"\n=== arm={arm.name}  trial={trial}  episodes={trainer_cfg.n_episodes} ===")
                if baselines is not None:
                    print(format_baseline_row(trainer_cfg.env_name, baselines,
                                              label="[reference]"))

            arm_trial_dir = os.path.join(out_dir, arm.name, f"trial_{trial}")
            logs, _, rows = train_one_arm(
                arm                 = arm,
                env_factory         = env_factory,
                prior_factory       = prior_factory,
                obs_builder_factory = obs_builder_factory,
                episode_runner      = episode_runner,
                trainer_cfg         = trainer_cfg,
                trial               = trial,
                hp                  = hp,
                verbose             = verbose,
                baselines           = baselines,
                probe_feats         = probe_feats,
                probe_infos         = probe_infos,
                log_dir             = arm_trial_dir,
                print_header        = not header_printed,
            )
            header_printed = True
            all_logs.extend(logs)
            all_rows.extend(rows)
            # Incremental top-level dump
            _dump(all_logs, all_rows, out_dir)
    _dump(all_logs, all_rows, out_dir)
    return all_logs


def _dump(logs: List[TrainLog], rows: List[Dict[str, Any]], out_dir: str) -> None:
    # Back-compat aggregate
    rows_simple = [asdict(l) for l in logs]
    with open(os.path.join(out_dir, "logs.json"), "w") as f:
        json.dump(rows_simple, f, indent=2, default=float)
    if rows_simple:
        with open(os.path.join(out_dir, "logs.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_simple[0].keys()))
            w.writeheader()
            w.writerows(rows_simple)
    # Rich aggregate (with all columns)
    if rows:
        with open(os.path.join(out_dir, "logs_rich.json"), "w") as f:
            json.dump(rows, f, indent=2, default=float)
        # Union of all keys for a complete CSV header
        keys: List[str] = []
        seen = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k); keys.append(k)
        with open(os.path.join(out_dir, "logs_rich.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)


def arms_from_strings(arm_names: List[str]) -> List[ArmConfig]:
    """'A,B,D' -> [ARM_A, ARM_B, ARM_D]."""
    return [CANONICAL_ARMS[n.strip().upper()] for n in arm_names]

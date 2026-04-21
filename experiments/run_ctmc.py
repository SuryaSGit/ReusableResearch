"""
4-arm ablation on Surya's Xie-CTMC `ctmc_env.CTMCEnv` (Network 1).

The CTMC env is event-count-driven (no natural "episode" boundary), so
we define an episode as `EVENTS_PER_EPISODE` successive arrivals plus
whatever departures occur in between.

Usage:
    python3 experiments/run_ctmc.py                                   # defaults
    python3 experiments/run_ctmc.py --trials 1 --episodes 200 --N 1   # smoke
    python3 experiments/run_ctmc.py --arms A,D
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ============================================================
# HYPERPARAMETERS — single source of truth for this experiment
# ============================================================

# --- Experiment identity ---
EXPERIMENT_NAME = "ctmc_ablation_v1"
RESULTS_ROOT    = "results"
SEED_BASE       = 20260421

# --- Training budget ---
N_EPISODES      = 500
N_TRIALS        = 1
LOG_EVERY       = 25
EVAL_ROLLOUTS   = 10
BASELINE_REFRESH_EVERY = 10

# --- Agent / optimizer ---
LR              = 3e-4
BATCH_SIZE      = 512
BUFFER_SIZE     = 100_000
HIDDEN_DIMS     = [256, 256]
GAMMA           = 1.0
TARGET_SYNC     = 10
N_STEP          = 3
GRAD_CLIP       = 1.0
HUBER_DELTA     = 1.0
LEARN_EVERY     = 4

# --- Epsilon-greedy ---
EPS_START       = 1.0
EPS_END         = 0.02
EPS_DECAY_FRAC  = 0.6

# --- Warm-start (Arms C, D) ---
ALPHA_START     = 0.3
ALPHA_DECAY_FRAC= 0.2

# --- Residual-Q (Arm D) ---
Q_CHT_WEIGHT    = 1.0

# --- Environment ---
CTMC_N              = 1
EVENTS_PER_EPISODE  = 200

# --- Q-probe set (R3) ---
N_PROBE_STATES  = 64

# --- Arms to run ---
ARMS            = ["A", "B", "C", "D"]


# ============================================================
# Imports that depend on ROOT on sys.path
# ============================================================

from ctmc_env  import CTMCEnv

from residual_rl.ablation           import run_ablation, arms_from_strings
from residual_rl.baselines          import collect_ctmc_baselines
from residual_rl.cht_prior          import ArmConfig, CTMCCHTPrior
from residual_rl.envs.hotel_wrapper import CTMCObsBuilder
from residual_rl.plotting           import plot_learning_curves, plot_q_magnitudes
from residual_rl.residual_dqn       import ResidualHP
from residual_rl.trainer            import TrainerConfig, build_probe_set


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def make_env(seed: int, N: int, events_per_episode: int) -> CTMCEnv:
    env = CTMCEnv(N=N, max_events=events_per_episode + 100)
    return env


def make_prior(env: CTMCEnv, arm: ArmConfig) -> CTMCCHTPrior:
    return CTMCCHTPrior(env, q_cht_weight=arm.q_cht_weight)


def make_obs_builder(env: CTMCEnv) -> CTMCObsBuilder:
    return CTMCObsBuilder(env)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def _info_snapshot(info: Dict[str, Any]) -> Dict[str, Any]:
    keep = (
        "current_type", "current_type_name", "state",
        "can_accept", "resource_usage", "resource_available", "time",
    )
    d = {k: info[k] for k in keep if k in info}
    if "state" in d:
        d["state"] = np.asarray(d["state"]).copy()
    return d


def ctmc_episode(
    agent,
    env: CTMCEnv,
    prior: CTMCCHTPrior,
    greedy: bool,
    seed: Optional[int] = None,
) -> Tuple[float, Dict]:
    if seed is not None:
        np.random.seed(int(seed))
    obs, info = env.reset()
    prior.reset_allocation()

    total = 0.0
    done  = False
    last_info = info
    while not done and info.get("current_type") is not None:
        action = agent.act(info, greedy=greedy)
        info_before = _info_snapshot(info)
        feat_before = agent.build_feat(info)

        obs, r, term, trunc, next_info = env.step(action)
        total += float(r)
        done = term or trunc

        if not greedy:
            info_after = _info_snapshot(next_info)
            feat_after = agent.build_feat(next_info)
            agent.buffer.push(
                feat_before, info_before, action, float(r),
                feat_after,  info_after,  done,
            )
            if (agent._steps % agent.hp.learn_every) == 0:
                agent.learn()
            agent._steps += 1

        info = next_info
        last_info = info

    return total, last_info


# ---------------------------------------------------------------------------
# HP banner
# ---------------------------------------------------------------------------

def _resolved_hp_dict(args) -> Dict[str, Any]:
    return {
        "experiment_name":       EXPERIMENT_NAME,
        "seed_base":             args.seed,
        "n_episodes":            args.episodes,
        "n_trials":              args.trials,
        "log_every":             args.log_every,
        "eval_rollouts":         EVAL_ROLLOUTS,
        "baseline_refresh_every": BASELINE_REFRESH_EVERY,
        "lr":                    LR,
        "batch_size":            BATCH_SIZE,
        "buffer_size":           BUFFER_SIZE,
        "hidden_dims":           list(HIDDEN_DIMS),
        "gamma":                 GAMMA,
        "target_sync":           TARGET_SYNC,
        "n_step":                N_STEP,
        "grad_clip":             GRAD_CLIP,
        "huber_delta":           HUBER_DELTA,
        "learn_every":           LEARN_EVERY,
        "eps_start":             EPS_START,
        "eps_end":               EPS_END,
        "eps_decay_frac":        EPS_DECAY_FRAC,
        "alpha_start":           ALPHA_START,
        "alpha_decay_frac":      ALPHA_DECAY_FRAC,
        "q_cht_weight":          Q_CHT_WEIGHT,
        "ctmc_N":                args.N,
        "events_per_episode":    args.events_per_episode,
        "n_probe_states":        N_PROBE_STATES,
        "arms":                  [a.strip().upper() for a in args.arms.split(",") if a.strip()],
    }


def _print_hp_banner(d: Dict[str, Any]) -> None:
    print("=" * 72)
    print(f" Hyperparameters (experiment = {d['experiment_name']}) — resolved values")
    print("=" * 72)
    for k, v in d.items():
        print(f"  {k:<24s} {v}")
    print("=" * 72)


def _arms_from_letters(letters: List[str]) -> List[ArmConfig]:
    out = []
    for L in letters:
        name = {
            "A": "A_vanilla",
            "B": "B_state_aug",
            "C": "C_state_aug_warmstart",
            "D": "D_full_residual",
        }[L]
        out.append(ArmConfig(
            name                 = name,
            use_delta_features   = L in ("B", "C", "D"),
            use_warm_start       = L in ("C", "D"),
            use_residual_q       = L == "D",
            q_cht_weight         = Q_CHT_WEIGHT if L == "D" else 0.0,
            alpha_start          = ALPHA_START,
            alpha_decay_frac     = ALPHA_DECAY_FRAC,
        ))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms",     default=",".join(ARMS))
    ap.add_argument("--trials",   type=int, default=N_TRIALS)
    ap.add_argument("--episodes", type=int, default=N_EPISODES)
    ap.add_argument("--N",        type=int, default=CTMC_N)
    ap.add_argument("--events-per-episode", type=int, default=EVENTS_PER_EPISODE)
    ap.add_argument("--log-every", type=int, default=LOG_EVERY)
    ap.add_argument("--out-dir",  default=None)
    ap.add_argument("--seed",     type=int, default=SEED_BASE)
    ap.add_argument("--quiet",    action="store_true")
    args = ap.parse_args()

    # ---- HP banner ----
    hp_dict = _resolved_hp_dict(args)
    _print_hp_banner(hp_dict)

    arm_letters = [a.strip().upper() for a in args.arms.split(",") if a.strip()]
    arms = _arms_from_letters(arm_letters)

    out_dir = args.out_dir or os.path.join(
        ROOT, RESULTS_ROOT,
        f"{EXPERIMENT_NAME}_N{args.N}_t{args.trials}_ep{args.episodes}"
        f"_{time.strftime('%Y%m%d-%H%M%S')}"
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nOutput dir: {out_dir}")
    print(f"Arms: {[a.name for a in arms]}\n")

    with open(os.path.join(out_dir, "hyperparameters.json"), "w") as f:
        json.dump(hp_dict, f, indent=2, default=str)

    # ---- Baselines ----
    env_factory = lambda s: make_env(s, args.N, args.events_per_episode)
    print("Computing baselines (one-shot)…")
    baselines = collect_ctmc_baselines(
        env_factory, n_rollouts=EVAL_ROLLOUTS, seed=args.seed,
    )
    print("\nBaseline reference:")
    print(f"  CHT-only (total):  {baselines['cht_only']['mean_reward']:>9.2f}  "
          f"+/- {baselines['cht_only']['std_reward']:.2f}")
    print(f"  CHT-only (rate):   {baselines['cht_only'].get('mean_rate', 0.0):>9.2f}")
    print(f"  LP upper bound:    {baselines['lp_bound']['mean_reward']:>9.2f}  (per unit time)")
    print(f"  AcceptAll (total): {baselines['accept_all']['mean_reward']:>9.2f}")
    print(f"  RejectAll (total): {baselines['reject_all']['mean_reward']:>9.2f}")

    # ---- Probe set (same infos for all arms; agent rebuilds feats) ----
    probe_arm = ArmConfig(name="probe", use_delta_features=True,
                          use_warm_start=False, use_residual_q=False)
    probe_feats, probe_infos = build_probe_set(
        env_factory         = env_factory,
        prior_factory       = make_prior,
        obs_builder_factory = make_obs_builder,
        arm                 = probe_arm,
        episode_runner      = ctmc_episode,
        n_probe             = N_PROBE_STATES,
        seed                = args.seed,
    )

    trainer_cfg = TrainerConfig(
        n_episodes    = args.episodes,
        log_every     = args.log_every,
        eval_episodes = EVAL_ROLLOUTS,
        seed          = args.seed,
        device        = "cpu",
        baseline_refresh_every = BASELINE_REFRESH_EVERY,
        env_name      = "ctmc",
    )

    hp = ResidualHP(
        hidden_dims    = list(HIDDEN_DIMS),
        buffer_size    = BUFFER_SIZE,
        batch_size     = BATCH_SIZE,
        lr             = LR,
        gamma          = GAMMA,
        n_step         = N_STEP,
        learn_every    = LEARN_EVERY,
        target_sync    = TARGET_SYNC,
        eps_start      = EPS_START,
        eps_end        = EPS_END,
        eps_decay_frac = EPS_DECAY_FRAC,
        grad_clip      = GRAD_CLIP,
    )

    t0 = time.time()
    logs = run_ablation(
        arms                = arms,
        env_factory         = env_factory,
        prior_factory       = make_prior,
        obs_builder_factory = make_obs_builder,
        episode_runner      = ctmc_episode,
        trainer_cfg         = trainer_cfg,
        n_trials            = args.trials,
        out_dir             = out_dir,
        hp                  = hp,
        verbose             = not args.quiet,
        baselines           = baselines,
        probe_feats         = probe_feats,
        probe_infos         = probe_infos,
    )
    dt = time.time() - t0
    print(f"\nTraining done in {dt:.1f}s.  {len(logs)} log rows -> {out_dir}")

    # ---- Plots ----
    rich_path = os.path.join(out_dir, "logs_rich.json")
    if os.path.exists(rich_path):
        with open(rich_path) as f:
            rich_rows = json.load(f)
    else:
        rich_rows = []
    plot_learning_curves(
        rich_rows or [asdict(l) for l in logs],
        out_path = os.path.join(out_dir, "learning_curves.png"),
        title    = f"{EXPERIMENT_NAME} (N={args.N}) learning curves",
        baselines = baselines,
        env_name  = "ctmc",
    )
    plot_q_magnitudes(
        rich_rows,
        out_path = os.path.join(out_dir, "q_magnitudes.png"),
    )
    print(f"Plots: learning_curves.png, q_magnitudes.png in {out_dir}")


if __name__ == "__main__":
    main()

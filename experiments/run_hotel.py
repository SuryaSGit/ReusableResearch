"""
4-arm ablation on Surya's `hotel_env.HotelEnv`.

Usage:
    python3 experiments/run_hotel.py                                  # defaults
    python3 experiments/run_hotel.py --trials 1 --episodes 200        # smoke
    python3 experiments/run_hotel.py --arms A,D --difficulty hard

CLI flags override the HP block values below. Flags that are not given
stick with the values in the HP block; anything touched below is the
single source of truth.
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

# Ensure repo root is on sys.path whether run from repo root or via -m
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ============================================================
# HYPERPARAMETERS — single source of truth for this experiment
# ============================================================

# --- Experiment identity ---
EXPERIMENT_NAME = "hotel_ablation_v1"
RESULTS_ROOT    = "results"
SEED_BASE       = 20260421

# --- Training budget ---
N_EPISODES      = 3000
N_TRIALS        = 3
LOG_EVERY       = 50
EVAL_ROLLOUTS   = 20
BASELINE_REFRESH_EVERY = 10        # print baseline row every N log rows

# --- Agent / optimizer ---
LR              = 3e-4
BATCH_SIZE      = 512
BUFFER_SIZE     = 100_000
HIDDEN_DIMS     = [256, 256]
GAMMA           = 1.0
TARGET_SYNC     = 10                # episodes between hard target syncs
N_STEP          = 3
GRAD_CLIP       = 1.0
HUBER_DELTA     = 1.0               # SmoothL1 default; noted for transparency
LEARN_EVERY     = 4                 # steps between learn() calls

# --- Epsilon-greedy ---
EPS_START       = 1.0
EPS_END         = 0.02
EPS_DECAY_FRAC  = 0.6               # fraction of training over which eps decays

# --- Warm-start (Arms C, D) ---
ALPHA_START     = 0.3
ALPHA_DECAY_FRAC= 0.2

# --- Residual-Q (Arm D) ---
Q_CHT_WEIGHT    = 1.0

# --- Environment ---
HOTEL_CAPACITY       = 20
HOTEL_HORIZON        = 50           # episode_length
HOTEL_DIFFICULTY     = "hard"       # "easy" or "hard"

# --- Q-probe set (R3) ---
N_PROBE_STATES  = 64

# --- Arms to run ---
ARMS            = ["A", "B", "C", "D"]


# ============================================================
# Imports that depend on ROOT on sys.path
# ============================================================

from hotel_env  import HotelEnv, TIGHT_CUSTOMER_TYPES, DEFAULT_CUSTOMER_TYPES

from residual_rl.ablation            import run_ablation, arms_from_strings
from residual_rl.baselines           import collect_hotel_baselines
from residual_rl.cht_prior           import ArmConfig, HotelCHTPrior
from residual_rl.envs.hotel_wrapper  import HotelObsBuilder
from residual_rl.plotting            import plot_learning_curves, plot_q_magnitudes
from residual_rl.residual_dqn        import ResidualHP
from residual_rl.trainer             import TrainerConfig, build_probe_set


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def make_env(seed: int, difficulty: str) -> HotelEnv:
    ctypes = TIGHT_CUSTOMER_TYPES if difficulty == "hard" else DEFAULT_CUSTOMER_TYPES
    return HotelEnv(
        capacity       = HOTEL_CAPACITY,
        episode_length = HOTEL_HORIZON,
        customer_types = ctypes,
        render_mode    = None,
        seed           = seed,
    )


def make_prior(env: HotelEnv, arm: ArmConfig) -> HotelCHTPrior:
    return HotelCHTPrior(env, q_cht_weight=arm.q_cht_weight)


def make_obs_builder(env: HotelEnv) -> HotelObsBuilder:
    return HotelObsBuilder(env)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def hotel_episode(
    agent,
    env: HotelEnv,
    prior: HotelCHTPrior,
    greedy: bool,
    seed: Optional[int] = None,
) -> Tuple[float, Dict]:
    if seed is not None:
        obs, info = env.reset(seed=seed)
    else:
        obs, info = env.reset()
    prior.reset_allocation()

    total = 0.0
    done = False
    last_info = info
    while not done:
        action = agent.act(info, greedy=greedy)

        feat_before = agent.build_feat(info)
        info_before = _info_snapshot(info)
        prior.update_allocation(info, action)

        obs, r, term, trunc, next_info = env.step(action)
        done = term or trunc
        total += float(r)

        if not greedy:
            feat_after = agent.build_feat(next_info)
            info_after = _info_snapshot(next_info)
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


def _info_snapshot(info: Dict[str, Any]) -> Dict[str, Any]:
    keep = (
        "current_customer", "current_customer_idx", "requested_rooms",
        "reward_per_room", "rooms_occupied", "rooms_available",
        "time_step", "can_accept", "state", "current_type",
    )
    return {k: info[k] for k in keep if k in info}


# ---------------------------------------------------------------------------
# HP banner
# ---------------------------------------------------------------------------

def _resolved_hp_dict(args) -> Dict[str, Any]:
    """Everything printed below is exactly what the run used."""
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
        "hotel_capacity":        HOTEL_CAPACITY,
        "hotel_horizon":         HOTEL_HORIZON,
        "hotel_difficulty":      args.difficulty,
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


# ---------------------------------------------------------------------------
# Build a per-arm ArmConfig from the HP constants (no hidden defaults)
# ---------------------------------------------------------------------------

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
    ap.add_argument("--arms",       default=",".join(ARMS))
    ap.add_argument("--trials",     type=int, default=N_TRIALS)
    ap.add_argument("--episodes",   type=int, default=N_EPISODES)
    ap.add_argument("--log-every",  type=int, default=LOG_EVERY)
    ap.add_argument("--difficulty", choices=["easy", "hard"], default=HOTEL_DIFFICULTY)
    ap.add_argument("--out-dir",    default=None)
    ap.add_argument("--seed",       type=int, default=SEED_BASE)
    ap.add_argument("--quiet",      action="store_true")
    args = ap.parse_args()

    # ---- HP banner ----
    hp_dict = _resolved_hp_dict(args)
    _print_hp_banner(hp_dict)

    arm_letters = [a.strip().upper() for a in args.arms.split(",") if a.strip()]
    arms = _arms_from_letters(arm_letters)

    out_dir = args.out_dir or os.path.join(
        ROOT, RESULTS_ROOT,
        f"{EXPERIMENT_NAME}_{args.difficulty}_t{args.trials}_ep{args.episodes}"
        f"_{time.strftime('%Y%m%d-%H%M%S')}"
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nOutput dir: {out_dir}")
    print(f"Arms: {[a.name for a in arms]}\n")

    # Dump resolved HP to disk (R5)
    with open(os.path.join(out_dir, "hyperparameters.json"), "w") as f:
        json.dump(hp_dict, f, indent=2, default=str)

    # ---- Baselines ----
    env_factory = lambda s: make_env(s, args.difficulty)
    print("Computing baselines (one-shot)…")
    baselines = collect_hotel_baselines(
        env_factory, n_rollouts=EVAL_ROLLOUTS, seed=args.seed,
    )
    print("\nBaseline reference:")
    print(f"  CHT-only:  {baselines['cht_only']['mean_reward']:>9.2f}  "
          f"+/- {baselines['cht_only']['std_reward']:.2f}")
    print(f"  Hindsight: {baselines['hindsight']['mean_reward']:>9.2f}  "
          f"+/- {baselines['hindsight']['std_reward']:.2f}")
    print(f"  AcceptAll: {baselines['accept_all']['mean_reward']:>9.2f}")
    print(f"  RejectAll: {baselines['reject_all']['mean_reward']:>9.2f}")

    # ---- Probe set (shared across all arms; delta features depend on arm) ----
    # Build it with the most-featureful arm (delta-on) so probe_obs dim matches
    # any arm that uses delta features; arms without delta features will get
    # a smaller obs, so we build probe set *per-arm* inside the trainer call
    # below. To keep "same states across arms", we build the probe info list
    # ONCE (env state), and reconstruct feats per-arm.
    probe_arm = ArmConfig(name="probe", use_delta_features=True,
                          use_warm_start=False, use_residual_q=False)
    probe_feats_D, probe_infos = build_probe_set(
        env_factory         = env_factory,
        prior_factory       = make_prior,
        obs_builder_factory = make_obs_builder,
        arm                 = probe_arm,
        episode_runner      = hotel_episode,
        n_probe             = N_PROBE_STATES,
        seed                = args.seed,
    )
    # Build same-state probe feats WITHOUT delta for arms A (use_delta=False).
    # The agent.q_magnitudes expects feats matching its own obs_dim, so we
    # pass the full-delta feats and, for non-delta arms, regenerate a
    # zero-delta version inside the probe.
    # Simpler: pass probe_infos and the specific arm's build uses whatever
    # it needs at log time. The trainer currently accepts a pre-built feats
    # matrix; to keep the shape matching, we hand the trainer the arm-A
    # variant and the arm-D variant lazily... but that's complex. Easy fix:
    # rebuild feats per-arm inside the trainer by using the arm's own build.
    # For now we pass the delta-on feats; arms without delta will still see
    # meaningful magnitudes because obs_dim differs — so instead we let the
    # agent rebuild feats from infos.

    # The cleanest path: pass probe_infos ONLY and let the agent rebuild
    # feats via its own build_feat. We do that inside trainer via a small
    # wrapper before q_magnitudes is called.

    trainer_cfg = TrainerConfig(
        n_episodes    = args.episodes,
        log_every     = args.log_every,
        eval_episodes = EVAL_ROLLOUTS,
        seed          = args.seed,
        device        = "cpu",
        baseline_refresh_every = BASELINE_REFRESH_EVERY,
        env_name      = "hotel",
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
        episode_runner      = hotel_episode,
        trainer_cfg         = trainer_cfg,
        n_trials            = args.trials,
        out_dir             = out_dir,
        hp                  = hp,
        verbose             = not args.quiet,
        baselines           = baselines,
        probe_feats         = probe_feats_D,        # arms with delta consume this
        probe_infos         = probe_infos,
    )
    dt = time.time() - t0
    print(f"\nTraining done in {dt:.1f}s.  {len(logs)} log rows -> {out_dir}")

    # ---- Plots (R5) ----
    import json as _json
    rich_path = os.path.join(out_dir, "logs_rich.json")
    if os.path.exists(rich_path):
        with open(rich_path) as f:
            rich_rows = _json.load(f)
    else:
        rich_rows = []
    plot_learning_curves(
        rich_rows or [asdict(l) for l in logs],
        out_path = os.path.join(out_dir, "learning_curves.png"),
        title    = f"{EXPERIMENT_NAME} ({args.difficulty}) learning curves",
        baselines = baselines,
        env_name  = "hotel",
    )
    plot_q_magnitudes(
        rich_rows,
        out_path = os.path.join(out_dir, "q_magnitudes.png"),
    )
    print(f"Plots: learning_curves.png, q_magnitudes.png in {out_dir}")


if __name__ == "__main__":
    main()

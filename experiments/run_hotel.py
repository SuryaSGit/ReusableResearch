"""
4-arm ablation on Surya's `hotel_env.HotelEnv`.

Usage:
    python3 experiments/run_hotel.py --trials 3 --episodes 3000 --arms A,B,C,D
    python3 experiments/run_hotel.py --trials 1 --episodes 200  --arms A,D   # smoke test
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure repo root is on sys.path whether run from repo root or via -m
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from hotel_env  import HotelEnv, TIGHT_CUSTOMER_TYPES, DEFAULT_CUSTOMER_TYPES

from residual_rl.ablation            import run_ablation, arms_from_strings
from residual_rl.cht_prior           import ArmConfig, HotelCHTPrior
from residual_rl.envs.hotel_wrapper  import HotelObsBuilder
from residual_rl.residual_dqn        import ResidualHP
from residual_rl.trainer             import TrainerConfig


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def make_env(seed: int, difficulty: str) -> HotelEnv:
    ctypes = TIGHT_CUSTOMER_TYPES if difficulty == "hard" else DEFAULT_CUSTOMER_TYPES
    return HotelEnv(
        capacity       = 20,
        episode_length = 50,
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
            # Learn every `learn_every` steps
            if (agent._steps % agent.hp.learn_every) == 0:
                agent.learn()
            agent._steps += 1

        info = next_info
        last_info = info

    return total, last_info


def _info_snapshot(info: Dict[str, Any]) -> Dict[str, Any]:
    """Copy the fields the prior actually reads, nothing else."""
    keep = (
        "current_customer", "current_customer_idx", "requested_rooms",
        "reward_per_room", "rooms_occupied", "rooms_available",
        "time_step", "can_accept", "state", "current_type",
    )
    return {k: info[k] for k in keep if k in info}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms",     default="A,B,C,D")
    ap.add_argument("--trials",   type=int, default=3)
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--difficulty", choices=["easy", "hard"], default="hard")
    ap.add_argument("--out-dir",  default=None)
    ap.add_argument("--seed",     type=int, default=0)
    ap.add_argument("--quiet",    action="store_true")
    args = ap.parse_args()

    arms = arms_from_strings([a.strip() for a in args.arms.split(",") if a.strip()])
    out_dir = args.out_dir or os.path.join(
        ROOT, "results",
        f"hotel_{args.difficulty}_t{args.trials}_ep{args.episodes}_{time.strftime('%Y%m%d-%H%M%S')}"
    )
    print(f"\nOutput dir: {out_dir}")
    print(f"Arms: {[a.name for a in arms]}")
    print(f"Trials: {args.trials}  Episodes/trial: {args.episodes}  Difficulty: {args.difficulty}\n")

    trainer_cfg = TrainerConfig(
        n_episodes    = args.episodes,
        log_every     = args.log_every,
        eval_episodes = 20,
        seed          = args.seed,
        device        = "cpu",
    )

    t0 = time.time()
    logs = run_ablation(
        arms                = arms,
        env_factory         = lambda s: make_env(s, args.difficulty),
        prior_factory       = make_prior,
        obs_builder_factory = make_obs_builder,
        episode_runner      = hotel_episode,
        trainer_cfg         = trainer_cfg,
        n_trials            = args.trials,
        out_dir             = out_dir,
        verbose             = not args.quiet,
    )
    print(f"\nDone in {time.time() - t0:.1f}s.  {len(logs)} log rows -> {out_dir}")


if __name__ == "__main__":
    main()

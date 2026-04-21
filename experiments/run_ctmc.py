"""
4-arm ablation on Surya's Xie-CTMC `ctmc_env.CTMCEnv` (Network 1).

The CTMC env is event-count-driven (no natural "episode" boundary), so
we define an episode as `--events-per-episode` successive arrivals,
resetting after each. This is coarse but gives the residual-RL agent
finite-horizon returns to learn from.

Usage:
    python3 experiments/run_ctmc.py --trials 1 --episodes 200 --N 1 --arms A,D
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ctmc_env  import CTMCEnv

from residual_rl.ablation           import run_ablation, arms_from_strings
from residual_rl.cht_prior          import ArmConfig, CTMCCHTPrior
from residual_rl.envs.hotel_wrapper import CTMCObsBuilder
from residual_rl.residual_dqn       import ResidualHP
from residual_rl.trainer            import TrainerConfig


# ---------------------------------------------------------------------------
# Factories (closed over CLI args via partials at main())
# ---------------------------------------------------------------------------

def make_env(seed: int, N: int, events_per_episode: int) -> CTMCEnv:
    env = CTMCEnv(N=N, max_events=events_per_episode + 100)
    # Seed the env's RNG explicitly. CTMCEnv uses np.random, not env.np_random,
    # so we fall back to global seeding at each reset inside the runner.
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
        "can_accept", "resource_usage", "resource_available",
    )
    d = {k: info[k] for k in keep if k in info}
    # arrays need to be copied so in-place changes in env don't corrupt buffer
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
    # CTMCEnv uses np.random at the module level; reseed per episode
    # so arrivals are reproducible across arms.
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
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms",     default="A,B,C,D")
    ap.add_argument("--trials",   type=int, default=1)
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--N",        type=int, default=1)
    ap.add_argument("--events-per-episode", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--out-dir",  default=None)
    ap.add_argument("--seed",     type=int, default=0)
    ap.add_argument("--quiet",    action="store_true")
    args = ap.parse_args()

    arms = arms_from_strings([a.strip() for a in args.arms.split(",") if a.strip()])
    out_dir = args.out_dir or os.path.join(
        ROOT, "results",
        f"ctmc_N{args.N}_t{args.trials}_ep{args.episodes}_{time.strftime('%Y%m%d-%H%M%S')}"
    )
    print(f"\nOutput dir: {out_dir}")
    print(f"Arms: {[a.name for a in arms]}")
    print(f"Trials: {args.trials}  Episodes/trial: {args.episodes}  N={args.N}\n")

    trainer_cfg = TrainerConfig(
        n_episodes    = args.episodes,
        log_every     = args.log_every,
        eval_episodes = 10,
        seed          = args.seed,
        device        = "cpu",
    )

    t0 = time.time()
    logs = run_ablation(
        arms                = arms,
        env_factory         = lambda s: make_env(s, args.N, args.events_per_episode),
        prior_factory       = make_prior,
        obs_builder_factory = make_obs_builder,
        episode_runner      = ctmc_episode,
        trainer_cfg         = trainer_cfg,
        n_trials            = args.trials,
        out_dir             = out_dir,
        verbose             = not args.quiet,
    )
    print(f"\nDone in {time.time() - t0:.1f}s.  {len(logs)} log rows -> {out_dir}")


if __name__ == "__main__":
    main()

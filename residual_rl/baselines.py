"""
Baseline reference policies.

All functions evaluate a non-learning policy on the same env factory as
the residual-RL trainer, so numbers are directly comparable. Seeding
matches the trainer's eval loop: each rollout k uses
    seed = seed_base + 13 * k
(the same offset the trainer uses inside `train_one_arm`).

Functions:

    cht_only_policy_hotel(env_factory, n_rollouts, seed)
        Greedy `accept iff Delta_i > 0` using HotelCHTPrior directly.

    cht_only_policy_ctmc(env_factory, n_rollouts, seed)
        Wraps `cht_policy.CHTPolicy` from the surya repo.

    lp_upper_bound_ctmc(env)
        Reads `env.lp_bound` (already computed by ctmc_env.CTMCEnv.__init__).
        Falls back to linprog if unset.

    hindsight_optimal_hotel(env_factory, n_rollouts, seed)
        Consumes `info["ideal_revenue"]` exposed by HotelEnv on episode end.

    random_baseline_accept_all(env_factory, n_rollouts, seed)
    random_baseline_reject_all(env_factory, n_rollouts, seed)
        Trivial floors.

All return a dict with keys {mean_reward, std_reward, per_rollout_rewards}.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _stats(rewards: List[float]) -> Dict[str, Any]:
    arr = np.asarray(rewards, dtype=float)
    return {
        "mean_reward":         float(arr.mean()) if arr.size else 0.0,
        "std_reward":          float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "per_rollout_rewards": [float(r) for r in rewards],
        "n_rollouts":          int(arr.size),
    }


def _rollout_seeds(seed_base: int, n_rollouts: int) -> List[int]:
    """Match the offset used by the trainer's eval loop."""
    return [int(seed_base + 13 * k) for k in range(n_rollouts)]


# ===========================================================================
# Hotel baselines
# ===========================================================================

def _run_hotel_policy(
    env_factory: Callable[[int], Any],
    decide:      Callable[[Any, Dict[str, Any]], int],
    rollout_seed: int,
    reset_cb:    Optional[Callable[[], None]] = None,
) -> float:
    """One greedy rollout of a deterministic hotel policy."""
    env = env_factory(rollout_seed)
    if reset_cb is not None:
        reset_cb()
    obs, info = env.reset(seed=rollout_seed)
    total = 0.0
    done = False
    while not done:
        a = decide(env, info)
        obs, r, term, trunc, info = env.step(a)
        total += float(r)
        done = term or trunc
    return total


def cht_only_policy_hotel(
    env_factory: Callable[[int], Any],
    n_rollouts:  int,
    seed:        int,
) -> Dict[str, Any]:
    """Greedy CHT rule: accept iff Delta_i > 0 (same rule HotelCHTPrior uses)."""
    from residual_rl.cht_prior import HotelCHTPrior, ArmConfig

    rewards: List[float] = []
    for rs in _rollout_seeds(seed, n_rollouts):
        env = env_factory(rs)
        prior = HotelCHTPrior(env, q_cht_weight=1.0)
        prior.reset_allocation()
        obs, info = env.reset(seed=rs)
        total = 0.0
        done = False
        while not done:
            a = prior.warm_start_action(info)
            # Hard capacity mask (same as agent.act)
            if info.get("rooms_available", 0) < info.get("requested_rooms", 0):
                a = 0
            prior.update_allocation(info, a)
            obs, r, term, trunc, info = env.step(a)
            total += float(r)
            done = term or trunc
        rewards.append(total)
    return _stats(rewards)


def hindsight_optimal_hotel(
    env_factory: Callable[[int], Any],
    n_rollouts:  int,
    seed:        int,
) -> Dict[str, Any]:
    """
    HotelEnv attaches `info["ideal_revenue"]` on episode termination
    (hotel_env.py:326). We run any quick policy that just drives the env
    through a full episode on the target seed, then read that field.

    We use accept-all as the driver because it's cheap; the hindsight
    optimum is a property of the arrival stream, not the policy.
    """
    rewards: List[float] = []
    for rs in _rollout_seeds(seed, n_rollouts):
        env = env_factory(rs)
        obs, info = env.reset(seed=rs)
        done = False
        while not done:
            obs, r, term, trunc, info = env.step(1)
            done = term or trunc
        ideal = float(info.get("ideal_revenue", 0.0))
        rewards.append(ideal)
    return _stats(rewards)


def random_baseline_accept_all(
    env_factory: Callable[[int], Any],
    n_rollouts:  int,
    seed:        int,
) -> Dict[str, Any]:
    rewards: List[float] = []
    for rs in _rollout_seeds(seed, n_rollouts):
        env = env_factory(rs)
        obs, info = env.reset(seed=rs)
        total = 0.0
        done = False
        while not done:
            obs, r, term, trunc, info = env.step(1)
            total += float(r)
            done = term or trunc
        rewards.append(total)
    return _stats(rewards)


def random_baseline_reject_all(
    env_factory: Callable[[int], Any],
    n_rollouts:  int,
    seed:        int,
) -> Dict[str, Any]:
    rewards: List[float] = []
    for rs in _rollout_seeds(seed, n_rollouts):
        env = env_factory(rs)
        obs, info = env.reset(seed=rs)
        total = 0.0
        done = False
        while not done:
            obs, r, term, trunc, info = env.step(0)
            total += float(r)
            done = term or trunc
        rewards.append(total)
    return _stats(rewards)


# ===========================================================================
# CTMC baselines
# ===========================================================================

def cht_only_policy_ctmc(
    env_factory: Callable[[int], Any],
    n_rollouts:  int,
    seed:        int,
) -> Dict[str, Any]:
    """
    Use the reference `cht_policy.CHTPolicy` directly, as a standalone
    greedy policy. No learning. One episode = one arrival stream of
    max_events steps (the env_factory decides how long).

    Returns *per-episode total reward* (same scale as the RL trainer's
    eval_reward), plus `mean_rate` = reward/time on the side for
    comparison against `lp_bound` (which is a rate).
    """
    from cht_policy import CHTPolicy

    rewards: List[float] = []
    rates:   List[float] = []
    for rs in _rollout_seeds(seed, n_rollouts):
        np.random.seed(int(rs))
        env = env_factory(rs)
        # Safe to build CHTPolicy after env (needs env.lp_solution).
        policy = CHTPolicy(env, delta=3.0)
        obs, info = env.reset(seed=int(rs))
        total = 0.0
        done = False
        while not done and info.get("current_type") is not None:
            a = policy(obs, info)
            obs, r, term, trunc, info = env.step(a)
            total += float(r)
            done = term or trunc
        rewards.append(total)
        t_end = float(info.get("time", 0.0))
        rates.append(total / t_end if t_end > 0 else 0.0)
    out = _stats(rewards)
    out["mean_rate"] = float(np.mean(rates)) if rates else 0.0
    out["std_rate"]  = float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0
    return out


def lp_upper_bound_ctmc(env) -> float:
    """
    LP bound for the Xie network. `CTMCEnv.__init__` already calls
    `self._compute_lp_bound()` and stashes the result on `env.lp_bound`
    (ctmc_env.py:108-128). We read that directly to avoid duplicating
    the optimisation.

    If for some reason `env.lp_bound is None` (scipy missing at env
    construction), fall back to computing it here with the same
    formulation.
    """
    val = getattr(env, "lp_bound", None)
    if val is not None:
        return float(val)

    # Fallback (should not be needed with stock CTMCEnv)
    from scipy.optimize import linprog
    r_mu = env.rewards * env.service_rates
    offered_load = env.arrival_rates / env.service_rates
    res = linprog(
        c=-r_mu,
        A_ub=env.A,
        b_ub=env.capacities.astype(float),
        bounds=[(0, ol) for ol in offered_load],
        method="highs",
    )
    if not res.success:
        raise RuntimeError("LP for CTMC upper bound failed: " + str(res.message))
    return float(-res.fun)


def _run_ctmc_fixed_action(env_factory, n_rollouts, seed, action: int) -> Dict[str, Any]:
    rewards: List[float] = []
    rates:   List[float] = []
    for rs in _rollout_seeds(seed, n_rollouts):
        np.random.seed(int(rs))
        env = env_factory(rs)
        obs, info = env.reset(seed=int(rs))
        total = 0.0
        done = False
        while not done and info.get("current_type") is not None:
            obs, r, term, trunc, info = env.step(action)
            total += float(r)
            done = term or trunc
        rewards.append(total)
        t_end = float(info.get("time", 0.0))
        rates.append(total / t_end if t_end > 0 else 0.0)
    out = _stats(rewards)
    out["mean_rate"] = float(np.mean(rates)) if rates else 0.0
    out["std_rate"]  = float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0
    return out


def accept_all_ctmc(env_factory, n_rollouts, seed):
    return _run_ctmc_fixed_action(env_factory, n_rollouts, seed, action=1)


def reject_all_ctmc(env_factory, n_rollouts, seed):
    return _run_ctmc_fixed_action(env_factory, n_rollouts, seed, action=0)


# ===========================================================================
# Convenience: run all baselines for an env and return a flat dict
# ===========================================================================

def collect_hotel_baselines(
    env_factory: Callable[[int], Any],
    n_rollouts:  int,
    seed:        int,
) -> Dict[str, Any]:
    return {
        "cht_only":   cht_only_policy_hotel(env_factory, n_rollouts, seed),
        "hindsight":  hindsight_optimal_hotel(env_factory, n_rollouts, seed),
        "accept_all": random_baseline_accept_all(env_factory, n_rollouts, seed),
        "reject_all": random_baseline_reject_all(env_factory, n_rollouts, seed),
    }


def collect_ctmc_baselines(
    env_factory: Callable[[int], Any],
    n_rollouts:  int,
    seed:        int,
) -> Dict[str, Any]:
    # Build one env just to read the LP bound (cheap).
    probe_env = env_factory(seed)
    return {
        "cht_only":   cht_only_policy_ctmc(env_factory, n_rollouts, seed),
        "lp_bound":   {"mean_reward": lp_upper_bound_ctmc(probe_env),
                       "std_reward": 0.0, "per_rollout_rewards": []},
        "accept_all": accept_all_ctmc(env_factory, n_rollouts, seed),
        "reject_all": reject_all_ctmc(env_factory, n_rollouts, seed),
    }

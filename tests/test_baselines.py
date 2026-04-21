"""
Tests for residual_rl.baselines.

Checks:
  * LP bound is positive and matches Xie Example 3.1 analytical value
    for N=1. Paper / env docs state lp_bound = 19.333... for N=1.
  * Hindsight revenue >= any greedy policy on the same seed (sanity:
    optimal-in-hindsight is an upper bound on all online policies).
  * cht_only_policy_hotel returns a dict with the expected keys.
  * cht_only_policy_ctmc returns a dict with the expected keys.
"""

import math
import numpy as np
import pytest

from hotel_env  import HotelEnv, TIGHT_CUSTOMER_TYPES
from ctmc_env   import CTMCEnv

from residual_rl.baselines import (
    cht_only_policy_hotel,
    cht_only_policy_ctmc,
    hindsight_optimal_hotel,
    lp_upper_bound_ctmc,
    random_baseline_accept_all,
    random_baseline_reject_all,
    collect_hotel_baselines,
    collect_ctmc_baselines,
)


def _hotel_env(seed):
    return HotelEnv(
        capacity=20, episode_length=50,
        customer_types=TIGHT_CUSTOMER_TYPES, render_mode=None, seed=seed,
    )


def _ctmc_env(seed, N=1):
    return CTMCEnv(N=N, max_events=220)


def test_lp_upper_bound_positive_and_matches_xie_example_3_1():
    env = _ctmc_env(0, N=1)
    val = lp_upper_bound_ctmc(env)
    assert val > 0, f"LP bound must be positive, got {val}"
    # Xie et al. Example 3.1 / Network 1: 19.333... for N=1
    assert math.isclose(val, 58.0 / 3.0, rel_tol=1e-6), (
        f"LP bound N=1 expected 19.3333..., got {val}"
    )


def test_lp_upper_bound_scales_with_N():
    # Linear scaling property: lp_bound(N) = N * lp_bound(1)
    env1 = _ctmc_env(0, N=1)
    env2 = _ctmc_env(0, N=2)
    v1 = lp_upper_bound_ctmc(env1)
    v2 = lp_upper_bound_ctmc(env2)
    assert math.isclose(v2, 2.0 * v1, rel_tol=1e-6)


def test_hindsight_ge_any_greedy_hotel():
    """
    Hindsight-optimal revenue is an upper bound on any realisable policy
    on the same arrival realisation.  Check hindsight >= CHT-only and
    hindsight >= accept-all averaged over multiple rollouts.
    """
    hind = hindsight_optimal_hotel(_hotel_env, n_rollouts=10, seed=1234)
    cht  = cht_only_policy_hotel(_hotel_env,   n_rollouts=10, seed=1234)
    aa   = random_baseline_accept_all(_hotel_env, n_rollouts=10, seed=1234)
    assert hind["mean_reward"] >= cht["mean_reward"] - 1e-6
    assert hind["mean_reward"] >= aa["mean_reward"]  - 1e-6


def test_cht_only_hotel_returns_expected_keys():
    d = cht_only_policy_hotel(_hotel_env, n_rollouts=3, seed=42)
    for k in ("mean_reward", "std_reward", "per_rollout_rewards", "n_rollouts"):
        assert k in d, f"missing key {k}"
    assert d["n_rollouts"] == 3
    assert len(d["per_rollout_rewards"]) == 3
    # Greedy CHT should be non-negative
    assert d["mean_reward"] >= 0.0


def test_cht_only_ctmc_returns_expected_keys():
    d = cht_only_policy_ctmc(_ctmc_env, n_rollouts=3, seed=42)
    for k in ("mean_reward", "std_reward", "per_rollout_rewards", "n_rollouts"):
        assert k in d, f"missing key {k}"
    # mean_rate surfaced for apples-to-apples LP comparison
    assert "mean_rate" in d
    assert d["mean_reward"] >= 0.0


def test_reject_all_is_zero():
    rr_hotel = random_baseline_reject_all(_hotel_env, n_rollouts=3, seed=0)
    assert rr_hotel["mean_reward"] == 0.0


def test_collect_hotel_baselines_dict_shape():
    b = collect_hotel_baselines(_hotel_env, n_rollouts=3, seed=0)
    for k in ("cht_only", "hindsight", "accept_all", "reject_all"):
        assert k in b
        assert "mean_reward" in b[k]


def test_collect_ctmc_baselines_dict_shape():
    b = collect_ctmc_baselines(_ctmc_env, n_rollouts=3, seed=0)
    for k in ("cht_only", "lp_bound", "accept_all", "reject_all"):
        assert k in b
        assert "mean_reward" in b[k]

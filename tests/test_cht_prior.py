"""
Q_CHT returns sane values on known hotel states, and the allocation
tracker behaves correctly.
"""

import numpy as np
import pytest

from hotel_env import HotelEnv, TIGHT_CUSTOMER_TYPES
from residual_rl.cht_prior import HotelCHTPrior


@pytest.fixture
def env():
    return HotelEnv(
        capacity       = 20,
        episode_length = 50,
        customer_types = TIGHT_CUSTOMER_TYPES,
        render_mode    = None,
        seed           = 7,
    )


def test_prior_returns_correct_shape(env):
    prior = HotelCHTPrior(env)
    info = {
        "current_customer":      "Premium",
        "current_customer_idx":  2,
        "requested_rooms":       3,
        "reward_per_room":       200.0,
        "rooms_occupied":        0,
        "time_step":             0,
    }
    q = prior.q_cht(info)
    assert q.shape == (2,)
    assert q.dtype == np.float32


def test_prior_accepts_when_under_allocated(env):
    """If delta > 0 (under-allocated), Q_CHT nudges toward accept."""
    prior = HotelCHTPrior(env)
    # Premium has a huge stationary target (top-priced, unbounded demand
    # within capacity). With zero cumulative allocation it is definitely
    # under-allocated.
    info = {
        "current_customer":     "Premium",
        "current_customer_idx": 2,
        "requested_rooms":      3,
        "reward_per_room":      200.0,
        "rooms_occupied":       0,
        "time_step":            0,
    }
    q = prior.q_cht(info)
    assert q[1] > q[0], f"Expected accept nudge, got {q}"
    assert q[0] == 0.0


def test_prior_rejects_when_over_allocated(env):
    """If delta < 0 (over-allocated), Q_CHT nudges toward reject."""
    prior = HotelCHTPrior(env)
    prior.allocation["Budget"] = 9999.0   # massively over the target
    info = {
        "current_customer":     "Budget",
        "current_customer_idx": 0,
        "requested_rooms":      1,
        "reward_per_room":      80.0,
        "rooms_occupied":       0,
        "time_step":            0,
    }
    q = prior.q_cht(info)
    assert q[0] > q[1], f"Expected reject nudge, got {q}"
    assert q[1] == 0.0


def test_prior_zero_weight_returns_zeros(env):
    prior = HotelCHTPrior(env, q_cht_weight=0.0)
    info = {
        "current_customer":     "Premium",
        "current_customer_idx": 2,
        "requested_rooms":      3,
        "reward_per_room":      200.0,
        "rooms_occupied":       0,
        "time_step":            0,
    }
    assert np.allclose(prior.q_cht(info), [0.0, 0.0])


def test_allocation_tracker_accumulates_on_accept(env):
    prior = HotelCHTPrior(env)
    prior.reset_allocation()
    info = {
        "current_customer":     "Budget",
        "requested_rooms":      2,
        "reward_per_room":      80.0,
        "rooms_occupied":       0,
        "time_step":            0,
    }
    prior.update_allocation(info, action=0)
    assert prior.allocation["Budget"] == 0.0
    prior.update_allocation(info, action=1)
    assert prior.allocation["Budget"] == 2.0
    prior.update_allocation(info, action=1)
    assert prior.allocation["Budget"] == 4.0


def test_delta_features_shape(env):
    prior = HotelCHTPrior(env)
    info = {
        "current_customer":     "Standard",
        "current_customer_idx": 1,
        "requested_rooms":      2,
        "reward_per_room":      120.0,
        "rooms_occupied":       0,
        "time_step":            0,
    }
    f = prior.delta_features(info)
    assert f.shape == (2,)
    assert f.dtype == np.float32


def test_horizon_aware_differs_from_stationary(env):
    """
    Late-in-episode, remaining capacity + remaining horizon matter. For
    a class that actually receives allocation under greedy LP (Premium),
    the target should shrink when the horizon is mostly consumed, so
    `horizon_aware` should produce a different delta from `stationary`.
    """
    prior_stat = HotelCHTPrior(env, horizon_aware=False)
    prior_haw  = HotelCHTPrior(env, horizon_aware=True)
    info = {
        "current_customer":     "Premium",
        "current_customer_idx": 2,
        "requested_rooms":      3,
        "reward_per_room":      200.0,
        "rooms_occupied":       15,   # near capacity
        "time_step":             45,   # near end of episode_length=50
    }
    f_stat = prior_stat.delta_features(info)
    f_haw  = prior_haw.delta_features(info)
    assert not np.allclose(f_stat, f_haw), f"Got stat={f_stat} vs haw={f_haw}"


def test_reward_scale_derived_from_env_not_global(env):
    """Bug B7: reward_scale must reflect THIS env's customer_types."""
    # Default tight types: max rpr=200, max req=max_rooms scale=8 -> scale=1/(200*8)=6.25e-4
    prior = HotelCHTPrior(env)
    expected = 1.0 / (200.0 * max(c.max_rooms for c in TIGHT_CUSTOMER_TYPES))
    assert prior.reward_scale == pytest.approx(expected)

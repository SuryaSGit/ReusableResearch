"""
Both envs produce a consistent (obs, info) tuple shape through our
wrappers and can run at least one step without crashing.
"""

import numpy as np
import pytest

from hotel_env  import HotelEnv, TIGHT_CUSTOMER_TYPES
from ctmc_env   import CTMCEnv

from residual_rl.cht_prior             import HotelCHTPrior, CTMCCHTPrior, ARM_A, ARM_D
from residual_rl.envs.hotel_wrapper    import HotelObsBuilder, CTMCObsBuilder


def test_hotel_obs_builder_shape():
    env = HotelEnv(customer_types=TIGHT_CUSTOMER_TYPES, render_mode=None, seed=0)
    _, info = env.reset()
    assert "current_customer_idx" in info, "hotel_env.py fix B5 must expose this key"
    b = HotelObsBuilder(env)
    obs = b(info)
    assert obs.shape == (b.base_obs_dim,)
    assert obs.dtype == np.float32
    # All scalars normalised to sensible range
    assert np.all(obs >= -1.0) and np.all(obs <= 2.0)


def test_hotel_full_step_cycle():
    env = HotelEnv(customer_types=TIGHT_CUSTOMER_TYPES, render_mode=None, seed=0)
    _, info = env.reset()
    b = HotelObsBuilder(env)
    prior = HotelCHTPrior(env)
    _ = b(info)
    _ = prior.delta_features(info)
    _ = prior.q_cht(info)
    obs, r, term, trunc, info2 = env.step(1)
    assert isinstance(r, float) or np.issubdtype(type(r), np.floating) or isinstance(r, int)
    if not term:
        _ = b(info2)
        _ = prior.q_cht(info2)


def test_ctmc_obs_builder_shape():
    env = CTMCEnv(N=1, max_events=200)
    _, info = env.reset(seed=0)
    b = CTMCObsBuilder(env)
    obs = b(info)
    assert obs.shape == (b.base_obs_dim,)
    assert obs.dtype == np.float32


def test_ctmc_prior_delta_shape():
    env = CTMCEnv(N=1, max_events=200)
    _, info = env.reset(seed=0)
    prior = CTMCCHTPrior(env)
    f = prior.delta_features(info)
    assert f.shape == (prior.delta_dim(),)
    assert f.shape[0] == 2 * env.n_types


def test_ctmc_prior_q_shape():
    env = CTMCEnv(N=1, max_events=200)
    _, info = env.reset(seed=0)
    prior = CTMCCHTPrior(env)
    q = prior.q_cht(info)
    assert q.shape == (2,)


def test_hotel_monkey_patch_is_not_required():
    """
    Audit bug B5: originally `dqn_agent.py` monkey-patched
    `HotelEnv._get_info` to add `current_customer_idx`.  With our fix
    to `hotel_env.py`, importing only `hotel_env` (never `dqn_agent`)
    must still expose the key.
    """
    # Force a fresh import order: only hotel_env.
    import importlib
    import hotel_env as he
    importlib.reload(he)   # make sure no side-effects survive from other tests
    env = he.HotelEnv(render_mode=None, seed=0)
    _, info = env.reset()
    assert "current_customer_idx" in info
    assert info["current_customer_idx"] in range(env.num_types)

"""
Tests for the R3 q_magnitudes probe on the residual-DQN agent.

Checks:
  * Returns positive floats for |Q_theta|, |Q_CHT|, |Q_final|.
  * For an Arm-D agent, |Q_final| ≈ mean |Q_theta + w·Q_CHT| numerically.
  * For Arm-A agent (no residual_q), |Q_final| == |Q_theta|.
"""

import numpy as np
import pytest
import torch

from hotel_env import HotelEnv, TIGHT_CUSTOMER_TYPES
from residual_rl.cht_prior    import ARM_A, ARM_D, ArmConfig, HotelCHTPrior
from residual_rl.envs.hotel_wrapper import HotelObsBuilder
from residual_rl.residual_dqn import ResidualDQNAgent, ResidualHP


@pytest.fixture
def env():
    return HotelEnv(
        capacity=20, episode_length=50,
        customer_types=TIGHT_CUSTOMER_TYPES, render_mode=None, seed=0,
    )


def _fake_infos(env, n=12):
    infos = []
    for i in range(n):
        ct = env.customer_types[i % len(env.customer_types)]
        infos.append({
            "current_customer":     ct.name,
            "current_customer_idx": i % len(env.customer_types),
            "requested_rooms":      ct.min_rooms,
            "reward_per_room":      ct.reward_per_room,
            "rooms_occupied":       i,
            "rooms_available":      env.capacity - i,
            "time_step":            i,
        })
    return infos


def _build_agent(env, arm):
    return ResidualDQNAgent(
        env         = env,
        obs_builder = HotelObsBuilder(env),
        prior       = HotelCHTPrior(env, q_cht_weight=arm.q_cht_weight),
        arm         = arm,
        hp          = ResidualHP(batch_size=8, buffer_size=64, n_step=1, gamma=1.0),
        n_episodes  = 10,
    )


def test_q_magnitudes_returns_positive_floats(env):
    torch.manual_seed(0)
    agent = _build_agent(env, ARM_D)
    infos = _fake_infos(env, n=12)
    feats = np.stack([agent.build_feat(i) for i in infos]).astype(np.float32)
    qm = agent.q_magnitudes(feats, infos)
    for k in ("q_theta", "q_cht", "q_final", "q_theta_over_final"):
        assert k in qm
    assert qm["q_theta"] >= 0.0
    assert qm["q_cht"]   >= 0.0
    assert qm["q_final"] >= 0.0


def test_q_final_equals_q_theta_plus_q_cht_arm_D(env):
    """
    For Arm D: |Q_final| computed by the probe MUST equal
    mean |Q_theta + Q_CHT| reconstructed by hand.
    """
    torch.manual_seed(1)
    agent = _build_agent(env, ARM_D)
    infos = _fake_infos(env, n=12)
    feats = np.stack([agent.build_feat(i) for i in infos]).astype(np.float32)

    qm = agent.q_magnitudes(feats, infos)

    with torch.no_grad():
        t = torch.tensor(feats, dtype=torch.float32)
        qt = agent.online(t)
    qc = agent.prior.q_cht_batch(infos)
    expected_final = float((qt + qc).abs().mean().item())

    assert np.isclose(qm["q_final"], expected_final, rtol=1e-5, atol=1e-6), (
        f"q_final {qm['q_final']} vs reconstructed {expected_final}"
    )


def test_q_final_equals_q_theta_for_arm_A(env):
    """
    Arm A has use_residual_q=False so Q_final = Q_theta.
    """
    torch.manual_seed(2)
    agent = _build_agent(env, ARM_A)
    infos = _fake_infos(env, n=12)
    feats = np.stack([agent.build_feat(i) for i in infos]).astype(np.float32)

    qm = agent.q_magnitudes(feats, infos)
    assert np.isclose(qm["q_final"], qm["q_theta"], rtol=1e-6, atol=1e-8)


def test_q_magnitudes_empty_probe_returns_zeros(env):
    agent = _build_agent(env, ARM_D)
    qm = agent.q_magnitudes(np.zeros((0, 1), dtype=np.float32), [])
    for k in ("q_theta", "q_cht", "q_final", "q_theta_over_final"):
        assert qm[k] == 0.0

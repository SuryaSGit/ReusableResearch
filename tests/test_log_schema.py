"""
Tests for the persistent log schema (R5).

Runs a very short training run per env and verifies:
  * log.jsonl has one row per log-point.
  * Each row contains the required columns for that env.
  * Rows parse as valid JSON.
"""

import json
import os
import tempfile
from typing import Any, Dict, List

import numpy as np
import pytest

from hotel_env import HotelEnv, TIGHT_CUSTOMER_TYPES
from ctmc_env  import CTMCEnv

from residual_rl.ablation    import run_ablation
from residual_rl.baselines   import collect_hotel_baselines, collect_ctmc_baselines
from residual_rl.cht_prior   import ARM_A, ARM_D, ArmConfig, HotelCHTPrior, CTMCCHTPrior
from residual_rl.envs.hotel_wrapper import HotelObsBuilder, CTMCObsBuilder
from residual_rl.residual_dqn import ResidualHP
from residual_rl.trainer      import TrainerConfig, build_probe_set


# Shared episode runners copied from the experiment scripts (kept minimal)

def _info_snapshot_hotel(info):
    keep = ("current_customer", "current_customer_idx", "requested_rooms",
            "reward_per_room", "rooms_occupied", "rooms_available",
            "time_step", "can_accept", "state", "current_type")
    return {k: info[k] for k in keep if k in info}


def _hotel_episode(agent, env, prior, greedy, seed):
    if seed is not None:
        obs, info = env.reset(seed=seed)
    else:
        obs, info = env.reset()
    prior.reset_allocation()
    total = 0.0
    done = False
    last_info = info
    while not done:
        a = agent.act(info, greedy=greedy)
        feat_before = agent.build_feat(info)
        info_before = _info_snapshot_hotel(info)
        prior.update_allocation(info, a)
        obs, r, term, trunc, next_info = env.step(a)
        done = term or trunc
        total += float(r)
        if not greedy:
            feat_after = agent.build_feat(next_info)
            info_after = _info_snapshot_hotel(next_info)
            agent.buffer.push(feat_before, info_before, a, float(r),
                              feat_after, info_after, done)
            if (agent._steps % agent.hp.learn_every) == 0:
                agent.learn()
            agent._steps += 1
        info = next_info
        last_info = info
    return total, last_info


def _info_snapshot_ctmc(info):
    keep = ("current_type", "current_type_name", "state",
            "can_accept", "resource_usage", "resource_available", "time")
    d = {k: info[k] for k in keep if k in info}
    if "state" in d:
        d["state"] = np.asarray(d["state"]).copy()
    return d


def _ctmc_episode(agent, env, prior, greedy, seed):
    if seed is not None:
        np.random.seed(int(seed))
    obs, info = env.reset()
    prior.reset_allocation()
    total = 0.0
    done = False
    last_info = info
    while not done and info.get("current_type") is not None:
        a = agent.act(info, greedy=greedy)
        info_before = _info_snapshot_ctmc(info)
        feat_before = agent.build_feat(info)
        obs, r, term, trunc, next_info = env.step(a)
        total += float(r)
        done = term or trunc
        if not greedy:
            info_after = _info_snapshot_ctmc(next_info)
            feat_after = agent.build_feat(next_info)
            agent.buffer.push(feat_before, info_before, a, float(r),
                              feat_after, info_after, done)
            if (agent._steps % agent.hp.learn_every) == 0:
                agent.learn()
            agent._steps += 1
        info = next_info
        last_info = info
    return total, last_info


HOTEL_REQUIRED_COLS = {
    "arm", "trial", "episode",
    "train_reward", "eval_reward",
    "cht_ref", "hindsight", "gap_pct",
    "q_theta", "q_cht", "q_final", "q_theta_over_final",
    "epsilon", "alpha", "lr", "avg_loss", "wallclock_s",
}
CTMC_REQUIRED_COLS = {
    "arm", "trial", "episode",
    "train_reward", "eval_reward", "eval_rate",
    "cht_ref", "lp_ub", "regret", "gap_pct",
    "q_theta", "q_cht", "q_final", "q_theta_over_final",
    "epsilon", "alpha", "lr", "avg_loss", "wallclock_s",
}


def test_hotel_log_jsonl_schema(tmp_path):
    env_factory = lambda s: HotelEnv(
        capacity=20, episode_length=20,
        customer_types=TIGHT_CUSTOMER_TYPES, render_mode=None, seed=s,
    )
    prior_factory = lambda env, arm: HotelCHTPrior(env, q_cht_weight=arm.q_cht_weight)
    obs_factory   = lambda env: HotelObsBuilder(env)
    baselines = collect_hotel_baselines(env_factory, n_rollouts=2, seed=0)
    probe_arm = ArmConfig(name="probe", use_delta_features=True,
                          use_warm_start=False, use_residual_q=False)
    probe_feats, probe_infos = build_probe_set(
        env_factory, prior_factory, obs_factory, probe_arm, _hotel_episode,
        n_probe=8, seed=0,
    )

    cfg = TrainerConfig(n_episodes=12, log_every=4, eval_episodes=2,
                        seed=0, env_name="hotel")
    out_dir = str(tmp_path)
    run_ablation(
        arms=[ARM_A, ARM_D],
        env_factory=env_factory,
        prior_factory=prior_factory,
        obs_builder_factory=obs_factory,
        episode_runner=_hotel_episode,
        trainer_cfg=cfg,
        n_trials=1,
        out_dir=out_dir,
        hp=ResidualHP(batch_size=16, buffer_size=256, n_step=1, gamma=1.0),
        verbose=False,
        baselines=baselines,
        probe_feats=probe_feats, probe_infos=probe_infos,
    )

    for arm_name in ("A_vanilla", "D_full_residual"):
        jsonl = os.path.join(out_dir, arm_name, "trial_0", "log.jsonl")
        assert os.path.exists(jsonl), f"missing {jsonl}"
        rows = []
        with open(jsonl) as f:
            for line in f:
                rows.append(json.loads(line))
        assert len(rows) >= 1, f"no rows in {jsonl}"
        for r in rows:
            missing = HOTEL_REQUIRED_COLS - set(r.keys())
            assert not missing, f"missing cols in {arm_name}: {missing}"


def test_ctmc_log_jsonl_schema(tmp_path):
    env_factory = lambda s: CTMCEnv(N=1, max_events=60)
    prior_factory = lambda env, arm: CTMCCHTPrior(env, q_cht_weight=arm.q_cht_weight)
    obs_factory   = lambda env: CTMCObsBuilder(env)
    baselines = collect_ctmc_baselines(env_factory, n_rollouts=2, seed=0)
    probe_arm = ArmConfig(name="probe", use_delta_features=True,
                          use_warm_start=False, use_residual_q=False)
    probe_feats, probe_infos = build_probe_set(
        env_factory, prior_factory, obs_factory, probe_arm, _ctmc_episode,
        n_probe=8, seed=0,
    )

    cfg = TrainerConfig(n_episodes=6, log_every=3, eval_episodes=2,
                        seed=0, env_name="ctmc")
    out_dir = str(tmp_path)
    run_ablation(
        arms=[ARM_A, ARM_D],
        env_factory=env_factory,
        prior_factory=prior_factory,
        obs_builder_factory=obs_factory,
        episode_runner=_ctmc_episode,
        trainer_cfg=cfg,
        n_trials=1,
        out_dir=out_dir,
        hp=ResidualHP(batch_size=16, buffer_size=256, n_step=1, gamma=1.0),
        verbose=False,
        baselines=baselines,
        probe_feats=probe_feats, probe_infos=probe_infos,
    )
    for arm_name in ("A_vanilla", "D_full_residual"):
        jsonl = os.path.join(out_dir, arm_name, "trial_0", "log.jsonl")
        assert os.path.exists(jsonl), f"missing {jsonl}"
        rows = []
        with open(jsonl) as f:
            for line in f:
                rows.append(json.loads(line))
        assert len(rows) >= 1
        for r in rows:
            missing = CTMC_REQUIRED_COLS - set(r.keys())
            assert not missing, f"missing cols in {arm_name}: {missing}"


def test_baselines_json_written(tmp_path):
    env_factory = lambda s: HotelEnv(
        capacity=20, episode_length=20,
        customer_types=TIGHT_CUSTOMER_TYPES, render_mode=None, seed=s,
    )
    baselines = collect_hotel_baselines(env_factory, n_rollouts=2, seed=0)
    out_dir = str(tmp_path)
    run_ablation(
        arms=[ARM_A],
        env_factory=env_factory,
        prior_factory=lambda e, a: HotelCHTPrior(e, q_cht_weight=a.q_cht_weight),
        obs_builder_factory=lambda e: HotelObsBuilder(e),
        episode_runner=_hotel_episode,
        trainer_cfg=TrainerConfig(n_episodes=4, log_every=2, eval_episodes=2,
                                  seed=0, env_name="hotel"),
        n_trials=1, out_dir=out_dir,
        hp=ResidualHP(batch_size=16, buffer_size=256, n_step=1, gamma=1.0),
        verbose=False,
        baselines=baselines,
    )
    assert os.path.exists(os.path.join(out_dir, "baselines.json"))
    with open(os.path.join(out_dir, "baselines.json")) as f:
        b = json.load(f)
    assert "cht_only" in b and "hindsight" in b

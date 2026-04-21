"""
Generic train/eval loop for the residual-RL agent. Env-agnostic: callers
pass env factories and an obs-builder + prior factory.

Returned logs are plain dataclasses (pickle-friendly) so the ablation
driver can stack them into a DataFrame-equivalent without depending on
pandas.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from residual_rl.cht_prior   import ArmConfig
from residual_rl.residual_dqn import ResidualDQNAgent, ResidualHP


@dataclass
class TrainerConfig:
    n_episodes: int = 3000
    log_every:  int = 50
    eval_episodes: int = 20
    seed:       int = 0
    device:     str = "cpu"


@dataclass
class TrainLog:
    arm:              str
    trial:            int
    episode:          int
    epsilon:          float
    alpha:            float
    train_reward:     float
    eval_reward:      float
    eval_reward_std:  float
    avg_loss:         float
    q_theta_abs:      float
    q_cht_abs:        float
    wallclock_s:      float


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# One training run (one arm, one trial)
# ---------------------------------------------------------------------------

def train_one_arm(
    arm:             ArmConfig,
    env_factory:     Callable[[int], Any],        # seed -> env
    prior_factory:   Callable[[Any, ArmConfig], Any],  # (env, arm) -> prior
    obs_builder_factory: Callable[[Any], Any],    # env -> obs_builder
    episode_runner:  Callable[[Any, Any, Any, bool, Optional[int]], Tuple[float, Dict]],
                                                   # (agent, env, prior, greedy, seed)
                                                   #   -> (episode_reward, info_last)
    trainer_cfg:     TrainerConfig,
    trial:           int = 0,
    hp:              Optional[ResidualHP] = None,
    verbose:         bool = False,
) -> Tuple[List[TrainLog], ResidualDQNAgent]:
    """
    Train one (arm, trial) pair. Deterministic given `trainer_cfg.seed + trial`.

    The caller supplies three factories plus an `episode_runner`:
      - `env_factory(seed)`              -> a fresh env seeded to `seed`
      - `prior_factory(env, arm)`        -> a CHT prior for that env
      - `obs_builder_factory(env)`       -> the vanilla obs builder
      - `episode_runner(agent, env, prior, greedy, seed)` -> one episode
           returns `(episode_reward, final_info)`

    `episode_runner` is env-specific because the hotel env is finite-
    horizon step-driven while the CTMC env is event-count-driven.
    """
    root_seed = trainer_cfg.seed + 100_003 * trial
    set_seeds(root_seed)

    # Separate deterministic seeds for the arrival/eval sequences that are
    # shared across arms: we want arm A and arm D in trial k to see the
    # SAME arrival stream so differences come from policy, not luck.
    env_seed = 7 * (trainer_cfg.seed + trial)        # shared across arms
    eval_seed = 99991 + 13 * (trainer_cfg.seed + trial)

    env = env_factory(env_seed)
    eval_env = env_factory(eval_seed)
    prior = prior_factory(env, arm)
    obs_builder = obs_builder_factory(env)
    eval_prior = prior_factory(eval_env, arm)  # independent prior with its own allocation tracker
    eval_obs_builder = obs_builder_factory(eval_env)

    agent = ResidualDQNAgent(
        env          = env,
        obs_builder  = obs_builder,
        prior        = prior,
        arm          = arm,
        hp           = hp or ResidualHP(),
        n_episodes   = trainer_cfg.n_episodes,
        device       = trainer_cfg.device,
    )

    logs:  List[TrainLog] = []

    t0 = time.time()
    for ep in range(trainer_cfg.n_episodes):
        train_reward, _ = episode_runner(agent, env, prior, False, None)
        agent.end_of_episode()

        if (ep + 1) % trainer_cfg.log_every == 0 or ep == 0:
            eval_rewards = []
            for ev in range(trainer_cfg.eval_episodes):
                r, _ = episode_runner(agent, eval_env, eval_prior, True, None)
                eval_rewards.append(r)
            eval_mean = float(np.mean(eval_rewards))
            eval_std  = float(np.std(eval_rewards))

            # Magnitudes (arm D diagnostic)
            q_theta_mag, q_cht_mag = _q_magnitudes(agent, env, prior, obs_builder, n=32)

            avg_loss = (
                float(np.mean(agent._recent_losses[-50:]))
                if agent._recent_losses else 0.0
            )
            logs.append(TrainLog(
                arm             = arm.name,
                trial           = trial,
                episode         = ep + 1,
                epsilon         = float(agent.epsilon),
                alpha           = float(agent.alpha),
                train_reward    = float(train_reward),
                eval_reward     = eval_mean,
                eval_reward_std = eval_std,
                avg_loss        = avg_loss,
                q_theta_abs     = q_theta_mag,
                q_cht_abs       = q_cht_mag,
                wallclock_s     = time.time() - t0,
            ))
            if verbose:
                print(
                    f"  [{arm.name} t={trial} ep={ep+1:>5}] "
                    f"eps={agent.epsilon:.2f} a={agent.alpha:.2f}  "
                    f"train={train_reward:>8.2f}  eval={eval_mean:>8.2f}  "
                    f"|Qth|={q_theta_mag:.3f} |Qcht|={q_cht_mag:.3f}"
                )

    return logs, agent


def _q_magnitudes(agent, env, prior, obs_builder, n: int = 32) -> Tuple[float, float]:
    """Sample `n` states from the replay buffer, report |Q_theta| and |Q_CHT|."""
    if len(agent.buffer) < n:
        return 0.0, 0.0
    obs, actions, _, _, _, infos, _ = agent.buffer.sample(n)
    with torch.no_grad():
        q_theta = agent.online(obs.to(agent.device))                # (n, A)
        q_theta_mag = float(q_theta.abs().mean().item())
    q_cht = prior.q_cht_batch(infos)
    q_cht_mag = float(q_cht.abs().mean().item())
    return q_theta_mag, q_cht_mag

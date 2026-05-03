"""
Residual-Q DQN — one agent class, four arms.

Given an `ArmConfig` and a `Prior` (Hotel or CTMC), the same class runs:

    arm A: vanilla Double Dueling DQN
    arm B: + Delta features in obs
    arm C: + alpha-annealed CHT-guided exploration (warm-start)
    arm D: + residual TD target   Q_final = Q_CHT + Q_theta

The residual-Q math (arm D):

    Q_theta(s,a) = r + gamma^n * max_{a'} [Q_CHT(s',a') + Q_theta(s',a')]
                                         - Q_CHT(s, a)

and the argmax in `act()` uses Q_final = Q_CHT + Q_theta, matching the
target (fix for audit bug B1).

When `arm.use_residual_q == False` the agent degenerates to ordinary
Double-DQN over the (possibly augmented) obs, so this single file covers
all four ablations.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from residual_rl.cht_prior import ArmConfig
from residual_rl.networks  import DuelingMLP
from residual_rl.replay_buffer import NStepReplayBuffer


# ---------------------------------------------------------------------------
# Hyperparameters (env-agnostic; experiment scripts can override)
# ---------------------------------------------------------------------------

@dataclass
class ResidualHP:
    hidden_dims:    List[int] = None          # type: ignore[assignment]
    buffer_size:    int       = 100_000
    batch_size:     int       = 512
    lr:             float     = 3e-4
    gamma:          float     = 1.0
    n_step:         int       = 3
    learn_every:    int       = 4
    target_sync:    int       = 10            # episodes between hard target syncs
    eps_start:      float     = 1.0
    eps_end:        float     = 0.02
    eps_decay_frac: float     = 0.6           # fraction of training over which eps linearly decays
    grad_clip:      float     = 1.0

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ResidualDQNAgent:

    def __init__(
        self,
        env,
        obs_builder,                            # callable: info -> vanilla obs vector
        prior,                                  # HotelCHTPrior or CTMCCHTPrior
        arm:         ArmConfig,
        hp:          Optional[ResidualHP] = None,
        n_episodes:  int    = 1000,
        device:      str    = "cpu",
        torch_gen:   Optional[torch.Generator] = None,
    ):
        self.env         = env
        self.obs_builder = obs_builder
        self.prior       = prior
        self.arm         = arm
        self.hp          = hp or ResidualHP()
        self.device      = torch.device(device)
        self.n_episodes  = int(n_episodes)
        self._episode    = 0
        self._steps      = 0

        # Derived obs dim
        delta_dim     = prior.delta_dim() if arm.use_delta_features else 0
        self.obs_dim  = obs_builder.base_obs_dim + delta_dim
        self.n_actions = prior.n_actions

        # Networks (seeded deterministically via torch global state by trainer)
        self.online = DuelingMLP(self.obs_dim, self.n_actions, self.hp.hidden_dims).to(self.device)
        self.target = DuelingMLP(self.obs_dim, self.n_actions, self.hp.hidden_dims).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=self.hp.lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(self.n_episodes, 1), eta_min=self.hp.lr / 20.0
        )
        self.loss_fn = nn.SmoothL1Loss()

        self.buffer  = NStepReplayBuffer(self.hp.buffer_size, self.hp.n_step, self.hp.gamma)
        self.epsilon = self.hp.eps_start
        self.alpha   = arm.alpha_start if arm.use_warm_start else 0.0

        # Per-episode loss accumulator — drained by the trainer
        self._recent_losses: List[float] = []

    # ------ schedules ------

    def _update_epsilon(self) -> None:
        decay_eps = max(int(self.hp.eps_decay_frac * self.n_episodes), 1)
        frac = min(self._episode / decay_eps, 1.0)
        self.epsilon = self.hp.eps_start + frac * (self.hp.eps_end - self.hp.eps_start)

    def _update_alpha(self) -> None:
        if not self.arm.use_warm_start:
            self.alpha = 0.0
            return
        decay_eps = max(int(self.arm.alpha_decay_frac * self.n_episodes), 1)
        frac = min(self._episode / decay_eps, 1.0)
        self.alpha = self.arm.alpha_start + frac * (0.0 - self.arm.alpha_start)

    # ------ obs ------

    def build_feat(self, info: Dict[str, Any]) -> np.ndarray:
        base = self.obs_builder(info)
        if self.arm.use_delta_features:
            return np.concatenate([base, self.prior.delta_features(info)])
        return base

    # ------ action selection ------

    def act(self, info: Dict[str, Any], greedy: bool = False) -> int:
        # 1. hard capacity mask (env-specific)
        if "rooms_available" in info and "requested_rooms" in info:
            if info["rooms_available"] < info["requested_rooms"]:
                return 0
        if "can_accept" in info and not info["can_accept"] and info.get("current_type") is not None:
            # CTMC env: accept is infeasible -> forced reject
            return 0

        if not greedy:
            r = random.random()
            # warm-start branch (arm C/D)
            if self.arm.use_warm_start and r < self.alpha:
                return self.prior.warm_start_action(info)
            # epsilon-greedy on the remaining mass
            boundary = self.alpha + self.epsilon * (1.0 - self.alpha)
            if r < boundary:
                return random.randint(0, self.n_actions - 1)

        feat = self.build_feat(info)
        with torch.no_grad():
            t = torch.tensor(feat, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_theta = self.online(t).squeeze(0)              # (n_actions,)
            if self.arm.use_residual_q:
                q_cht = torch.tensor(self.prior.q_cht(info), dtype=torch.float32, device=self.device)
                q_final = q_theta + q_cht
            else:
                q_final = q_theta
        return int(q_final.argmax().item())

    # ------ learning ------

    def learn(self) -> Optional[float]:
        if len(self.buffer) < self.hp.batch_size:
            return None

        obs, actions, rewards, next_obs, dones, infos, next_infos = self.buffer.sample(self.hp.batch_size)
        obs      = obs.to(self.device)
        actions  = actions.to(self.device)
        rewards  = rewards.to(self.device)
        next_obs = next_obs.to(self.device)
        dones    = dones.to(self.device)

        q_sa = self.online(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.arm.use_residual_q:
                # Residual-Q TD target (corrected). The argmax is over
                # Q_final(s', .) = Q_theta(s', .) + Q_CHT(s', .), and the
                # target value is Q_theta(s', a*) + Q_CHT(s', a*). Finally
                # we subtract Q_CHT(s, a) so that Q_theta learns the
                # residual.  See cht_prior.ArmConfig docstring.
                q_cht_next = self.prior.q_cht_batch(next_infos).to(self.device)   # (B, n)
                q_cht_curr = self.prior.q_cht_batch(infos).to(self.device)        # (B, n)
                q_online_next = self.online(next_obs) + q_cht_next                # (B, n)
                best_a = q_online_next.argmax(dim=1, keepdim=True)                # (B, 1)
                q_target_next = self.target(next_obs).gather(1, best_a).squeeze(1)
                q_cht_next_best = q_cht_next.gather(1, best_a).squeeze(1)
                q_cht_sa = q_cht_curr.gather(1, actions.unsqueeze(1)).squeeze(1)
                td_target = (
                    rewards
                    + (self.hp.gamma ** self.hp.n_step)
                    * (q_target_next + q_cht_next_best)
                    * (1.0 - dones)
                    - q_cht_sa
                )
            else:
                # Double-DQN target, unchanged
                best_a = self.online(next_obs).argmax(dim=1, keepdim=True)
                q_target_next = self.target(next_obs).gather(1, best_a).squeeze(1)
                td_target = rewards + (self.hp.gamma ** self.hp.n_step) * q_target_next * (1.0 - dones)

        loss = self.loss_fn(q_sa, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=self.hp.grad_clip)
        self.optimizer.step()
        self._steps += 1
        loss_f = float(loss.item())
        self._recent_losses.append(loss_f)
        if len(self._recent_losses) > 200:
            self._recent_losses = self._recent_losses[-200:]
        return loss_f

    # ------ diagnostics ------

    def q_magnitudes(
        self,
        probe_obs:   np.ndarray,
        probe_infos: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Report average |Q_theta|, |Q_CHT|, |Q_final| over a *fixed* probe set.

        The probe set is sampled ONCE at experiment start (by the trainer)
        and reused across all arms/log-points so magnitudes are directly
        comparable.

        Returns keys:
            q_theta              - mean |Q_theta| over probe states+actions
            q_cht                - mean |Q_CHT|   over probe states+actions
            q_final              - mean |Q_theta + Q_CHT|  (or |Q_theta| when
                                   arm.use_residual_q is False — Q_final is
                                   what the agent actually consumes)
            q_theta_over_final   - ratio |Q_theta| / |Q_final| (passenger check)
        """
        if probe_obs.shape[0] == 0:
            return {"q_theta": 0.0, "q_cht": 0.0, "q_final": 0.0,
                    "q_theta_over_final": 0.0}

        with torch.no_grad():
            t = torch.tensor(probe_obs, dtype=torch.float32, device=self.device)
            q_theta = self.online(t)                                 # (B, A)
            q_theta_mag = float(q_theta.abs().mean().item())

        q_cht_t = self.prior.q_cht_batch(probe_infos).to(self.device)  # (B, A)
        q_cht_mag = float(q_cht_t.abs().mean().item())

        if self.arm.use_residual_q:
            q_final = q_theta + q_cht_t
        else:
            q_final = q_theta
        q_final_mag = float(q_final.abs().mean().item())
        ratio = q_theta_mag / q_final_mag if q_final_mag > 1e-8 else 0.0
        return {
            "q_theta":             q_theta_mag,
            "q_cht":               q_cht_mag,
            "q_final":             q_final_mag,
            "q_theta_over_final":  ratio,
        }

    # ------ episode boundary hooks ------

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def end_of_episode(self) -> None:
        self._episode += 1
        self._update_epsilon()
        self._update_alpha()
        # Only step the LR scheduler once the optimizer has taken at least
        # one update — avoids the torch warning + skipped first LR value.
        if self._steps > 0 and len(self.buffer) >= self.hp.batch_size:
            self.scheduler.step()
        if self._episode % self.hp.target_sync == 0:
            self.sync_target()

"""
N-step replay buffer that also stores per-transition ``info`` dicts.

We need infos on both sides of a transition because the residual-Q TD
target (arm D) evaluates `Q_CHT(s, a)` and `Q_CHT(s', a')`, and the CHT
prior reads structured fields from info (customer type, request,
state, etc.) — not from the (s, s') feature vectors that go into the
Q-net.

The episode-boundary flushing logic is copied verbatim from
`dqn_agent.NStepReplayBuffer` (which had already fixed Surya's original
"stale pending transition" bug).
"""

from __future__ import annotations

import collections
import random
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Tuple

import numpy as np
import torch


@dataclass
class Transition:
    obs:       np.ndarray
    info:      Dict[str, Any]
    action:    int
    reward:    float           # n-step return
    next_obs:  np.ndarray
    next_info: Dict[str, Any]
    done:      bool


class NStepReplayBuffer:

    def __init__(self, capacity: int, n: int, gamma: float):
        self.buf:     Deque[Transition] = collections.deque(maxlen=capacity)
        self.pending: List[Tuple]       = []
        self.n     = int(n)
        self.gamma = float(gamma)

    def _commit(self, start_idx: int, final_next_obs, final_next_info, final_done: bool) -> None:
        first_obs, first_info, first_action, _, _, _, _ = self.pending[start_idx]
        n_ret = 0.0
        g     = 1.0
        for _, _, _, r, n_obs, n_info, d in self.pending[start_idx:]:
            n_ret += g * r
            g     *= self.gamma
            final_next_obs  = n_obs
            final_next_info = n_info
            final_done      = d
            if d:
                break
        self.buf.append(Transition(
            first_obs, first_info, first_action, n_ret,
            final_next_obs, final_next_info, final_done,
        ))

    def push(
        self,
        obs:       np.ndarray,
        info:      Dict[str, Any],
        action:    int,
        reward:    float,
        next_obs:  np.ndarray,
        next_info: Dict[str, Any],
        done:      bool,
    ) -> None:
        self.pending.append((obs, info, action, reward, next_obs, next_info, done))
        if len(self.pending) >= self.n:
            self._commit(0, next_obs, next_info, done)
            self.pending.pop(0)
        if done:
            for i in range(len(self.pending)):
                self._commit(i, next_obs, next_info, True)
            self.pending.clear()

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        obs       = torch.tensor(np.stack([t.obs      for t in batch]), dtype=torch.float32)
        actions   = torch.tensor([t.action  for t in batch],            dtype=torch.long)
        rewards   = torch.tensor([t.reward  for t in batch],            dtype=torch.float32)
        next_obs  = torch.tensor(np.stack([t.next_obs for t in batch]), dtype=torch.float32)
        dones     = torch.tensor([t.done    for t in batch],            dtype=torch.float32)
        infos     = [t.info      for t in batch]
        next_infos= [t.next_info for t in batch]
        return obs, actions, rewards, next_obs, dones, infos, next_infos

    def __len__(self) -> int:
        return len(self.buf)

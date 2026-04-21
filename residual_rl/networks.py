"""
Dueling MLP Q-network (shared across all four arms).

Kept minimal — the purpose is not architecture research, it's the
residual-Q math and the ablation structure.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class DuelingMLP(nn.Module):
    """MLP with a dueling value + advantage head.  obs -> Q(s, .)."""

    def __init__(
        self,
        obs_dim:      int,
        n_actions:    int,
        hidden_dims:  List[int] | None = None,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [256, 256]

        trunk: List[nn.Module] = []
        in_dim = obs_dim
        for i, h in enumerate(hidden_dims[:-1]):
            trunk.append(nn.Linear(in_dim, h))
            if i == 0:
                trunk.append(nn.LayerNorm(h))
            trunk.append(nn.ReLU())
            in_dim = h
        self.trunk = nn.Sequential(*trunk)

        last_h = hidden_dims[-1]
        self.value_stream     = nn.Sequential(
            nn.Linear(in_dim, last_h), nn.ReLU(), nn.Linear(last_h, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(in_dim, last_h), nn.ReLU(), nn.Linear(last_h, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.trunk(x)
        V = self.value_stream(shared)                    # (B, 1)
        A = self.advantage_stream(shared)                # (B, n)
        return V + A - A.mean(dim=1, keepdim=True)       # (B, n)

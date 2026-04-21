"""
Thin wrappers that give `HotelEnv` and `CTMCEnv` a uniform API for the
residual-RL agent.

What the agent needs per step:

    obs:    np.ndarray (already normalised to roughly [0, 1])
    info:   dict carrying whatever the CHT prior needs
    n_actions: int (= 2 for both envs in this repo)
    build_obs(info): vanilla observation vector
    base_obs_dim: int

This module does NOT re-implement the env physics — it just wraps
Surya's existing `HotelEnv` / `CTMCEnv` instances.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Hotel
# ---------------------------------------------------------------------------

class HotelObsBuilder:
    """
    Replicates `dqn_agent.build_obs`: 6 normalised scalars + one-hot
    customer type. Lives in the wrapper so we don't import the reference
    implementation (`dqn_agent.py`) — we keep that file as evidence of
    the "before" state.
    """

    def __init__(self, env):
        self.env = env
        self.num_types = env.num_types
        self._total_prob = sum(c.arrival_prob for c in env.customer_types)
        self._exp_rpr = sum(
            (c.arrival_prob / self._total_prob) * c.reward_per_room
            for c in env.customer_types
        )
        self._max_rpr = max(c.reward_per_room for c in env.customer_types)
        self._max_req = max(c.max_rooms       for c in env.customer_types) * env.scale

    @property
    def base_obs_dim(self) -> int:
        return 6 + self.num_types

    def __call__(self, info: Dict[str, Any]) -> np.ndarray:
        env = self.env
        time_rem  = max(env.episode_length - info["time_step"], 0) / env.episode_length
        cur_rpr   = info["reward_per_room"]
        opp_ratio = float(np.clip(cur_rpr / self._exp_rpr, 0.0, 2.0)) / 2.0

        one_hot = np.zeros(self.num_types, dtype=np.float32)
        one_hot[info.get("current_customer_idx", 0)] = 1.0

        scalars = np.array([
            info["rooms_occupied"]  / env.capacity,
            info["rooms_available"] / env.capacity,
            time_rem,
            info["requested_rooms"] / self._max_req,
            cur_rpr                 / self._max_rpr,
            opp_ratio,
        ], dtype=np.float32)

        return np.concatenate([scalars, one_hot])


# ---------------------------------------------------------------------------
# CTMC
# ---------------------------------------------------------------------------

class CTMCObsBuilder:
    """
    Normalised obs for the CTMC env:
        [x_1..x_K / max_cap,  one_hot_event_type,  is_arrival]
    """

    def __init__(self, env):
        self.env = env
        self.num_types = env.n_types
        self._max_cap  = float(np.max(env.capacities))

    @property
    def base_obs_dim(self) -> int:
        return self.num_types + self.num_types + 1   # state + one-hot type + is_arrival

    def __call__(self, info: Dict[str, Any]) -> np.ndarray:
        env = self.env
        state = np.asarray(info.get("state", np.zeros(self.num_types)), dtype=np.float32)
        state_norm = state / max(self._max_cap, 1.0)

        type_idx = int(info.get("current_type", 0))
        one_hot = np.zeros(self.num_types, dtype=np.float32)
        one_hot[type_idx] = 1.0

        is_arr = 1.0 if info.get("current_type") is not None else 0.0

        return np.concatenate([state_norm, one_hot, np.asarray([is_arr], dtype=np.float32)])

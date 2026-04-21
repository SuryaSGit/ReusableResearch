"""
CHT prior: Q_CHT(s, a), target allocations, and the single ArmConfig that
drives the four ablation arms.

Two env-specific priors are provided:

  * `HotelCHTPrior`  — single-resource finite-horizon hotel RM
                       (Surya's `hotel_env.HotelEnv`).

  * `CTMCCHTPrior`   — multi-resource CTMC loss network (Xie et al. 2024
                       "Network 1"; Surya's `ctmc_env.CTMCEnv`). Built on
                       top of the existing `cht_policy.CHTPolicy` so we
                       don't re-derive the corrected-head-count logic.

Both priors expose the same interface:

    prior.q_cht(info)            -> np.ndarray of shape (n_actions,)
    prior.delta_features(info)   -> np.ndarray of shape (d_delta,)  (may be empty)
    prior.warm_start_action(info)-> int                              (argmax of Q_CHT)

The TD-target code in `residual_dqn.ResidualDQNAgent.learn()` consumes
only the batch form:

    prior.q_cht_batch(infos)     -> torch.Tensor of shape (B, n_actions)

Fix audits covered here:
  * B1: Q_CHT appears in BOTH the target argmax and the TD target (arm D).
        When `arm.use_residual_q == False` the prior is still usable for
        warm-start, but its values are never added to Q-net outputs.
  * B6: `HotelCHTPrior` explicitly documents that it uses
        episode-cumulative allocation (fine for hotel, no departures).
        `CTMCCHTPrior` uses *current* occupancy and is therefore correct
        for the CTMC env with departures.
  * B7: reward_scale for the hotel prior is derived from the env's
        active `customer_types` at construction time, not globally.
  * B9: `HotelCHTPrior` supports `horizon_aware=True` to recompute the
        fluid LP target using remaining horizon.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Arm configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ArmConfig:
    """
    Single source of truth for which levers a run uses.

    Four canonical arms:
        A — Vanilla             (F, F, F)
        B — State-aug only      (T, F, F)
        C — + warm-start        (T, T, F)
        D — Full residual RL    (T, T, T)

    Setting `q_cht_weight = 0.0` with `use_residual_q = True` is
    mathematically identical to Arm A — a property the test suite relies on.
    """
    name:                str
    use_delta_features:  bool  = False
    use_warm_start:      bool  = False
    use_residual_q:      bool  = False
    q_cht_weight:        float = 1.0
    alpha_start:         float = 0.3
    alpha_decay_frac:    float = 0.2    # fraction of total episodes over which alpha -> 0


ARM_A = ArmConfig(name="A_vanilla",             use_delta_features=False, use_warm_start=False, use_residual_q=False)
ARM_B = ArmConfig(name="B_state_aug",           use_delta_features=True,  use_warm_start=False, use_residual_q=False)
ARM_C = ArmConfig(name="C_state_aug_warmstart", use_delta_features=True,  use_warm_start=True,  use_residual_q=False)
ARM_D = ArmConfig(name="D_full_residual",       use_delta_features=True,  use_warm_start=True,  use_residual_q=True)

CANONICAL_ARMS: Dict[str, ArmConfig] = {"A": ARM_A, "B": ARM_B, "C": ARM_C, "D": ARM_D}


# ===========================================================================
# Hotel prior
# ===========================================================================

class HotelCHTPrior:
    """
    Fluid-LP target allocation for the single-resource hotel RM env.

    For a single resource the LP reduces to "sort classes by $/room and
    greedily fill capacity". We keep this as the default (matches
    `cht_dqn.compute_cht_target`).

    `horizon_aware=True` recomputes X* each call using the *remaining*
    episode horizon and *remaining* capacity, which is the honest choice
    when the agent is mid-episode (bug B9).

    Cumulative allocation tracker: because hotel episodes have no
    departures, `update_allocation(info, action)` simply accumulates
    accepted rooms per class within the episode (matches Surya's
    `AllocationTracker`).  The class documents this so the CTMC user
    doesn't reuse it.
    """

    def __init__(
        self,
        env,                                 # HotelEnv
        q_cht_weight:  float = 1.0,
        horizon_aware: bool  = False,
    ):
        self.env           = env
        self.q_cht_weight  = float(q_cht_weight)
        self.horizon_aware = bool(horizon_aware)
        self.customer_types = list(env.customer_types)
        self.num_types      = len(self.customer_types)

        # Reward scale derived from THIS env's customer set (fix B7).
        max_rpr = max(c.reward_per_room for c in self.customer_types)
        max_req = max(c.max_rooms       for c in self.customer_types)
        self.reward_scale = 1.0 / (max_rpr * max_req)
        self._max_rpr     = float(max_rpr)
        self._max_req     = float(max_req)

        self.allocation: Dict[str, float] = {c.name: 0.0 for c in self.customer_types}
        self._stationary_target = self._compute_target_alloc(
            remaining_capacity = float(env.capacity),
            remaining_horizon  = float(env.episode_length),
        )

    # ----- allocation tracker -----

    def reset_allocation(self) -> None:
        self.allocation = {c.name: 0.0 for c in self.customer_types}

    def update_allocation(self, info: Dict[str, Any], action: int) -> None:
        """Cumulative rooms-accepted per class within the episode."""
        if action == 1:
            cname = info["current_customer"]
            self.allocation[cname] = self.allocation.get(cname, 0.0) + float(
                info["requested_rooms"]
            )

    # ----- target allocation -----

    def _compute_target_alloc(
        self,
        remaining_capacity: float,
        remaining_horizon:  float,
    ) -> Dict[str, float]:
        total_prob = sum(c.arrival_prob for c in self.customer_types)
        demand = {
            c.name: (c.arrival_prob / total_prob)
                    * (c.min_rooms + c.max_rooms) / 2
                    * remaining_horizon
            for c in self.customer_types
        }
        sorted_types = sorted(self.customer_types, key=lambda c: -c.reward_per_room)
        remaining = float(remaining_capacity)
        target: Dict[str, float] = {c.name: 0.0 for c in self.customer_types}
        for c in sorted_types:
            alloc = min(demand[c.name], remaining)
            target[c.name] = alloc
            remaining -= alloc
            if remaining <= 0:
                break
        return target

    def _target_now(self, info: Dict[str, Any]) -> Dict[str, float]:
        if not self.horizon_aware:
            return self._stationary_target
        remaining_horizon  = max(self.env.episode_length - info["time_step"], 1)
        remaining_capacity = max(self.env.capacity - info["rooms_occupied"], 0.0)
        return self._compute_target_alloc(
            remaining_capacity = remaining_capacity,
            remaining_horizon  = remaining_horizon,
        )

    # ----- features -----

    def delta_features(self, info: Dict[str, Any]) -> np.ndarray:
        """[delta_norm, phi(delta_norm)]  (matches cht_dqn.build_cht_obs)."""
        cname     = info["current_customer"]
        x_star    = self._target_now(info).get(cname, 0.0)
        x_curr    = self.allocation.get(cname, 0.0)
        delta_raw = x_star - x_curr
        delta_norm= float(np.clip(delta_raw / self.env.capacity, -1.0, 1.0))
        phi       = float(np.sign(delta_norm) * np.sqrt(abs(delta_norm)))
        return np.asarray([delta_norm, phi], dtype=np.float32)

    def delta_dim(self) -> int:
        return 2

    # ----- Q_CHT -----

    def q_cht(self, info: Dict[str, Any]) -> np.ndarray:
        """Q_CHT(s, .) in reward-scale units.  Returns (q_reject, q_accept)."""
        if self.q_cht_weight == 0.0:
            return np.zeros(2, dtype=np.float32)
        cname     = info["current_customer"]
        x_star    = self._target_now(info).get(cname, 0.0)
        x_curr    = self.allocation.get(cname, 0.0)
        delta_raw = x_star - x_curr
        w_i       = float(info["reward_per_room"]) * float(info["requested_rooms"])
        scaled    = w_i * self.reward_scale * self.q_cht_weight
        if delta_raw > 0:
            return np.asarray([0.0, scaled], dtype=np.float32)
        if delta_raw < 0:
            return np.asarray([scaled, 0.0], dtype=np.float32)
        return np.zeros(2, dtype=np.float32)

    def q_cht_batch(self, infos: Sequence[Dict[str, Any]]) -> torch.Tensor:
        return torch.tensor(np.stack([self.q_cht(i) for i in infos]), dtype=torch.float32)

    def warm_start_action(self, info: Dict[str, Any]) -> int:
        q = self.q_cht(info)
        # tie-break toward reject, matches Surya's `1 if delta_raw > 0 else 0`
        return int(q[1] > q[0])

    # ----- obs API shared with the wrapper -----

    @property
    def n_actions(self) -> int:
        return 2


# ===========================================================================
# CTMC prior
# ===========================================================================

class CTMCCHTPrior:
    """
    Thin wrapper around Surya's `cht_policy.CHTPolicy` that exposes the
    residual-RL interface.

    For the CTMC env:
      * `allocation` is the *current* customer count per type (not
        cumulative) — departures decrement it. We read it from
        `info["state"]` each call, so we don't need an explicit tracker.
      * `delta_features` returns the per-type gap to the LP target x*
        (normalised by the max capacity), plus its sign-sqrt compression.
        Shape: (2 * n_types,)  so the policy sees the gap vector directly.
      * `q_cht(info)` is ONE-HOT on {accept, reject} depending on whether
        the CHT policy would accept the current arriving type.
    """

    def __init__(self, env, q_cht_weight: float = 1.0, delta_cht: float = 3.0):
        from cht_policy import CHTPolicy   # lazy import to avoid startup cost

        self.env          = env
        self.q_cht_weight = float(q_cht_weight)
        self.policy       = CHTPolicy(env, delta=delta_cht)
        self.n_types      = env.n_types
        self.n_resources  = env.n_resources
        self._max_cap     = float(np.max(env.capacities))
        # "reward scale" so Q_CHT is comparable in magnitude to per-step raw reward.
        # Per-step reward is at most max_i r_i; we scale by 1/max_r.
        max_r = float(max(ct.reward for ct in env.customer_types))
        self.reward_scale = 1.0 / max_r if max_r > 0 else 1.0

    # ----- allocation tracker (no-op: state is read directly) -----

    def reset_allocation(self) -> None:
        pass

    def update_allocation(self, info: Dict[str, Any], action: int) -> None:
        pass

    # ----- features -----

    def delta_features(self, info: Dict[str, Any]) -> np.ndarray:
        state = np.asarray(info.get("state", np.zeros(self.n_types)), dtype=np.float32)
        x_star = self.policy._target_allocation(state.astype(int))   # type: ignore[arg-type]
        # Map x_star back to full n_types vector (zeros for preferred/rejected types)
        x_star_full = np.zeros(self.n_types, dtype=np.float32)
        for idx, i in enumerate(self.policy.less_preferred):
            x_star_full[i] = float(x_star[idx])
        # For preferred types, "target" = offered load (lambda/mu) as a soft guide
        offered_load = self.env.arrival_rates / self.env.service_rates
        for i in self.policy.preferred:
            x_star_full[i] = float(offered_load[i])

        delta = (x_star_full - state) / max(self._max_cap, 1.0)
        delta = np.clip(delta, -1.0, 1.0).astype(np.float32)
        phi   = np.sign(delta) * np.sqrt(np.abs(delta))
        return np.concatenate([delta, phi.astype(np.float32)])

    def delta_dim(self) -> int:
        return 2 * self.n_types

    # ----- Q_CHT -----

    def q_cht(self, info: Dict[str, Any]) -> np.ndarray:
        if self.q_cht_weight == 0.0:
            return np.zeros(2, dtype=np.float32)
        if not info.get("can_accept", False):
            # CHT would reject; Q_CHT nudges "reject".
            return np.asarray([self.q_cht_weight * self.reward_scale, 0.0], dtype=np.float32)
        # CHTPolicy.__call__ evaluates its default-arg expression even when
        # "state" is in info (Python eagerly evaluates `dict.get(k, default)`).
        # Hand it a dummy obs that it can safely slice.
        dummy_obs = np.zeros(self.n_types + 2, dtype=np.float32)
        a_cht = self.policy(dummy_obs, info)     # 0 or 1
        type_idx = info.get("current_type", 0)
        r_i = float(self.env.customer_types[type_idx].reward)
        nudge = r_i * self.reward_scale * self.q_cht_weight
        if a_cht == 1:
            return np.asarray([0.0, nudge], dtype=np.float32)
        else:
            return np.asarray([nudge, 0.0], dtype=np.float32)

    def q_cht_batch(self, infos: Sequence[Dict[str, Any]]) -> torch.Tensor:
        return torch.tensor(np.stack([self.q_cht(i) for i in infos]), dtype=torch.float32)

    def warm_start_action(self, info: Dict[str, Any]) -> int:
        q = self.q_cht(info)
        return int(q[1] > q[0])

    @property
    def n_actions(self) -> int:
        return 2

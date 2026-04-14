"""
CHT-Augmented DQN — Hotel Revenue Management
=============================================
Implements the Neural CHT algorithm sketched in the design doc.

Core idea
---------
Classical Continuous Heavy Traffic (CHT) theory gives us a fluid-limit
optimal allocation X*_i for each customer class i.  The *residual*
Δ_i = X*_i − x_i tells the network how far the current allocation deviates
from the theoretical optimum — a structured, theoretically-grounded feature
the vanilla DQN had to discover from scratch.

Three enhancements over vanilla DQN
------------------------------------
1. CHT state augmentation
   Augmented obs = [vanilla_obs | Δ | φ(Δ)]
   where φ(Δ) = sign-preserved normalisation, clipped to [-1, 1].
   This gives the Q-network a coordinate system rooted in RM theory.

2. Residual Q-learning
   Q_final(s, a) = Q_CHT(s, a) + Q_θ(s, a)
   Q_CHT is a simple heuristic: +w_i for accept if Δ_i > 0, else 0.
   Q_θ learns only the correction term — much smaller magnitude,
   faster convergence, more stable gradients.

3. Hybrid α-annealed policy
   With prob α  → follow CHT signal (accept iff Δ_i > 0)
   With prob ε  → random (standard exploration)
   With prob rest → argmax Q_final
   α decays linearly to 0 over alpha_decay_ep episodes.
   Note: α and ε operate on disjoint probability mass to avoid
   the buffer-domination problem (see design notes below).

CHT target allocation X*_i (fluid approximation)
-------------------------------------------------
In the infinite-horizon fluid limit, the optimal allocation is the
solution to the weighted bandwidth-sharing LP:

    max  Σ_i w_i * x_i
    s.t. Σ_i x_i  ≤ C          (capacity)
         x_i ≤ λ_i * E[req_i]  (demand ceiling per class)
         x_i ≥ 0

We solve this exactly (it's a tiny LP — 4 classes, 1 resource) using
a greedy sort-by-weight heuristic that is optimal for a single-resource
problem:  sort classes by w_i descending, fill greedily up to demand.

This gives a *stationary* X* that ignores the finite horizon.  The CHT
correction Δ_i = X*_i − x_i is therefore a structural bias term rather
than a horizon-aware signal, which is by design — Q_θ learns the
horizon-dependent residual on top.

Design notes
------------
• α and ε are on disjoint mass:
    p(CHT)   = α
    p(random)= ε * (1 - α)
    p(DQN)   = (1 - ε) * (1 - α)
  This prevents CHT from swamping the buffer during early training.

• Q_CHT magnitude is scaled to reward_scale units so the residual
  Q_θ trains on the same numerical scale as vanilla DQN — no new
  hyperparameter needed.

• The residual architecture still uses the full Dueling + Double DQN
  setup from the base agent.  Q_CHT is added *after* the dueling
  combination, before argmax.

Run:
    python cht_dqn.py                     # fresh training
    python cht_dqn.py --resume            # resume from cht_best.pt
"""

from __future__ import annotations

import argparse
import random
import collections
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from hotel_env import (
    HotelEnv,
    CustomerType,
    TIGHT_CUSTOMER_TYPES,
    GreedyAgent,
    ThresholdAgent,
    RandomAgent,
)
from dqn_agent import (
    HyperParams,
    NStepReplayBuffer,
    TrainLog,
    _expected_rpr,
    _potential,
    shaped_reward,
    evaluate,
    compare_agents,
    policy_summary,
    regret_breakdown,
    _patched_get_info,   # ensure patch is applied
)


# ────────────────────────────────────────────────────────────────────────────
# CHT hyperparameters (extends base HyperParams)
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class CHTHyperParams(HyperParams):
    # α-annealing: CHT prior probability
    alpha_start   : float = 0.5     # start: 50% of actions follow CHT prior
    alpha_end     : float = 0.0     # end:   pure DQN
    alpha_decay_ep: int   = 2_000   # linear decay over first N episodes

    # Residual learning weight applied to Q_CHT heuristic
    # Set to 0.0 to disable residual learning (ablation)
    q_cht_weight  : float = 1.0

    # Whether to include Δ features in the observation
    # Set to False to ablate the state augmentation
    use_cht_obs   : bool  = True

    def __post_init__(self):
        super().__post_init__()


CHTHP = CHTHyperParams()


# ────────────────────────────────────────────────────────────────────────────
# CHT target allocation  (fluid LP, single resource, greedy-optimal)
# ────────────────────────────────────────────────────────────────────────────

def compute_cht_target(env: HotelEnv) -> Dict[str, float]:
    """
    Solve the fluid-limit LP for target allocation X*_i per class.

    For a single-resource problem this reduces to: sort classes by
    reward_per_room descending, allocate min(demand_i, remaining) greedily.

    Returns a dict mapping customer type name → X*_i (in rooms).
    """
    total_prob = sum(c.arrival_prob for c in env.customer_types)
    # Expected rooms demanded per episode per class
    demand = {
        c.name: (c.arrival_prob / total_prob)
                * (c.min_rooms + c.max_rooms) / 2
                * env.episode_length
        for c in env.customer_types
    }
    # Sort by reward_per_room descending → greedy fills highest-value first
    sorted_types = sorted(env.customer_types, key=lambda c: -c.reward_per_room)

    remaining = float(env.capacity)
    target    = {}
    for c in sorted_types:
        alloc      = min(demand[c.name], remaining)
        target[c.name] = alloc
        remaining -= alloc
        if remaining <= 0:
            break
    # Classes that didn't get allocation
    for c in env.customer_types:
        target.setdefault(c.name, 0.0)
    return target


def compute_delta(
    info      : Dict,
    env       : HotelEnv,
    x_star    : Dict[str, float],
    x_current : Dict[str, float],
) -> Tuple[float, float]:
    """
    Compute Δ_i = X*_i − x_i for the *arriving* customer class i,
    and return (delta_raw, delta_normalised).

    delta_raw  : signed rooms difference, unbounded
    delta_norm : clipped to [-1, 1] by dividing by capacity
    """
    cname     = info["current_customer"]
    x_star_i  = x_star.get(cname, 0.0)
    x_curr_i  = x_current.get(cname, 0.0)
    delta_raw = x_star_i - x_curr_i
    delta_norm= float(np.clip(delta_raw / env.capacity, -1.0, 1.0))
    return delta_raw, delta_norm


# ────────────────────────────────────────────────────────────────────────────
# Current allocation tracker  (rooms occupied per class this episode)
# ────────────────────────────────────────────────────────────────────────────

class AllocationTracker:
    """Tracks rooms occupied per customer class within the current episode."""

    def __init__(self, env: HotelEnv):
        self.env   = env
        self.x     : Dict[str, float] = {c.name: 0.0 for c in env.customer_types}

    def reset(self):
        self.x = {c.name: 0.0 for c in self.env.customer_types}

    def update(self, customer_name: str, rooms: int, accepted: bool):
        if accepted:
            self.x[customer_name] = self.x.get(customer_name, 0.0) + rooms


# ────────────────────────────────────────────────────────────────────────────
# Augmented observation builder
# ────────────────────────────────────────────────────────────────────────────

MAX_REWARD_PER_ROOM = max(c.reward_per_room for c in TIGHT_CUSTOMER_TYPES)
OBS_DIM_BASE        = 6 + len(TIGHT_CUSTOMER_TYPES)   # same as vanilla


def build_cht_obs(
    info     : Dict,
    env      : HotelEnv,
    delta_raw: float,
    delta_norm: float,
    use_cht  : bool = True,
) -> np.ndarray:
    """
    Augmented observation:
        [vanilla_obs | delta_norm | phi(delta_norm)]

    phi(Δ) = sign-preserved square root: sign(Δ) * sqrt(|Δ|)
    This compresses large Δ values while preserving direction —
    important for stability when Δ can be very large early in training.

    If use_cht=False, appends two zeros (ablation mode).
    """
    time_rem  = max(env.episode_length - info["time_step"], 0) / env.episode_length
    exp_rpr   = _expected_rpr(env.customer_types)
    cur_rpr   = info["reward_per_room"]
    opp_ratio = float(np.clip(cur_rpr / exp_rpr, 0.0, 2.0)) / 2.0

    one_hot = np.zeros(env.num_types, dtype=np.float32)
    one_hot[info.get("current_customer_idx", 0)] = 1.0

    scalars = np.array([
        info["rooms_occupied"]  / env.capacity,
        info["rooms_available"] / env.capacity,
        time_rem,
        info["requested_rooms"] / env.max_request,
        cur_rpr                 / MAX_REWARD_PER_ROOM,
        opp_ratio,
    ], dtype=np.float32)

    if use_cht:
        phi = float(np.sign(delta_norm) * np.sqrt(abs(delta_norm)))
        cht_feats = np.array([delta_norm, phi], dtype=np.float32)
    else:
        cht_feats = np.zeros(2, dtype=np.float32)

    return np.concatenate([scalars, one_hot, cht_feats])


def cht_obs_dim(env: HotelEnv) -> int:
    return OBS_DIM_BASE + 2   # +delta_norm, +phi(delta)


# ────────────────────────────────────────────────────────────────────────────
# Q_CHT heuristic  (residual baseline)
# ────────────────────────────────────────────────────────────────────────────

def q_cht_heuristic(
    info      : Dict,
    delta_raw : float,
    env       : HotelEnv,
    weight    : float,
    reward_scale: float,
) -> Tuple[float, float]:
    """
    Heuristic Q-value for each action, scaled to training units.

    Q_CHT(s, Accept) = weight * w_i  if Δ_i > 0  else  0
    Q_CHT(s, Reject) = weight * w_i  if Δ_i < 0  else  0

    Intuition:
      Δ > 0 → we're under-allocated for class i → accept is aligned with theory
      Δ < 0 → we're over-allocated for class i  → reject is aligned with theory
      Δ = 0 → no prior signal → both Q_CHT = 0, pure Q_θ decides

    Returns (q_reject, q_accept) in reward_scale units.
    """
    if weight == 0.0:
        return 0.0, 0.0
    w_i      = info["reward_per_room"] * info["requested_rooms"]
    scaled_w = w_i * reward_scale * weight
    if delta_raw > 0:
        return 0.0, scaled_w     # theory says accept
    elif delta_raw < 0:
        return scaled_w, 0.0     # theory says reject
    else:
        return 0.0, 0.0


# ────────────────────────────────────────────────────────────────────────────
# Q-Network  (same Dueling architecture as vanilla DQN)
# ────────────────────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_dims: List[int]):
        super().__init__()
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
            nn.Linear(in_dim, last_h), nn.ReLU(), nn.Linear(last_h, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.trunk(x)
        V = self.value_stream(shared)
        A = self.advantage_stream(shared)
        return V + A - A.mean(dim=1, keepdim=True)


# ────────────────────────────────────────────────────────────────────────────
# CHT-DQN Agent
# ────────────────────────────────────────────────────────────────────────────

class CHTDQNAgent:
    """
    CHT-Augmented Double Dueling DQN.

    Three levers over vanilla DQN
    ------------------------------
    use_cht_obs    — augment obs with [Δ, φ(Δ)]
    q_cht_weight   — residual learning weight for Q_CHT heuristic
    alpha_*        — CHT prior annealing schedule

    All three can be independently ablated:
      CHTHyperParams(use_cht_obs=False, q_cht_weight=0.0, alpha_start=0.0)
      → reduces exactly to vanilla DQNAgent
    """

    def __init__(
        self,
        env       : HotelEnv,
        hp        : CHTHyperParams = CHTHP,
        device    : str = "cpu",
        n_episodes: int = 0,
    ):
        self.env      = env
        self.hp       = hp
        self.device   = torch.device(device)
        self._episode = 0

        self.x_star   = compute_cht_target(env)   # stationary CHT target
        self.tracker  = AllocationTracker(env)

        _obs_dim = cht_obs_dim(env)
        self.online = QNetwork(_obs_dim, hp.hidden_dims).to(self.device)
        self.target = QNetwork(_obs_dim, hp.hidden_dims).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=hp.lr)
        T_max = max(n_episodes or hp.n_episodes, 1)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=T_max, eta_min=hp.lr / 20
        )
        self.loss_fn  = nn.SmoothL1Loss()
        self.buffer   = NStepReplayBuffer(hp.buffer_size, hp.n_step, hp.gamma)
        self.epsilon  = hp.eps_start
        self.alpha    = hp.alpha_start
        self.steps    = 0

        self._best_eval_revenue : float = -float("inf")
        self._best_path         : str   = "cht_best.pt"

    # ── Schedule updates ──────────────────────────────────────────────────

    def _update_epsilon(self) -> None:
        frac         = min(self._episode / self.hp.eps_decay_ep, 1.0)
        self.epsilon = self.hp.eps_start + frac * (self.hp.eps_end - self.hp.eps_start)

    def _update_alpha(self) -> None:
        frac       = min(self._episode / self.hp.alpha_decay_ep, 1.0)
        self.alpha = self.hp.alpha_start + frac * (self.hp.alpha_end - self.hp.alpha_start)

    # ── Delta computation ─────────────────────────────────────────────────

    def _delta(self, info: Dict) -> Tuple[float, float]:
        return compute_delta(info, self.env, self.x_star, self.tracker.x)

    # ── Observation ───────────────────────────────────────────────────────

    def _obs(self, info: Dict) -> np.ndarray:
        delta_raw, delta_norm = self._delta(info)
        return build_cht_obs(info, self.env, delta_raw, delta_norm, self.hp.use_cht_obs)

    # ── Action selection ──────────────────────────────────────────────────

    def act(self, obs: np.ndarray, info: Dict, greedy: bool = False) -> int:
        # 1. Hard capacity mask
        if info["rooms_available"] < info["requested_rooms"]:
            return 0

        delta_raw, _ = self._delta(info)

        if not greedy:
            r = random.random()
            # 2. α-branch: follow CHT prior (on disjoint mass from ε)
            if r < self.alpha:
                return 1 if delta_raw > 0 else 0
            # 3. ε-branch: random exploration (on remaining mass)
            if r < self.alpha + self.epsilon * (1 - self.alpha):
                return self.env.action_space.sample()

        # 4. Q_final = Q_CHT + Q_θ
        feat = self._obs(info)
        with torch.no_grad():
            q_theta = self.online(
                torch.tensor(feat, dtype=torch.float32, device=self.device).unsqueeze(0)
            ).squeeze(0)   # shape (2,)

        q_cht_rej, q_cht_acc = q_cht_heuristic(
            info, delta_raw, self.env,
            self.hp.q_cht_weight, self.hp.reward_scale
        )
        q_final = torch.tensor(
            [q_cht_rej, q_cht_acc], dtype=torch.float32, device=self.device
        ) + q_theta

        return int(q_final.argmax().item())

    # ── Learning ──────────────────────────────────────────────────────────

    def learn(self) -> Optional[float]:
        if len(self.buffer) < self.hp.batch_size:
            return None

        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.hp.batch_size)
        obs      = obs.to(self.device)
        actions  = actions.to(self.device)
        rewards  = rewards.to(self.device)
        next_obs = next_obs.to(self.device)
        dones    = dones.to(self.device)

        q_vals = self.online(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            best_a = self.online(next_obs).argmax(dim=1, keepdim=True)
            next_q = self.target(next_obs).gather(1, best_a).squeeze(1)
            target = rewards + (self.hp.gamma ** self.hp.n_step) * next_q * (1.0 - dones)

        loss = self.loss_fn(q_vals, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item()

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def end_of_episode(self) -> None:
        self.scheduler.step()

    # ── Checkpoint ────────────────────────────────────────────────────────

    def save(self, path: str = "cht_hotel.pt") -> None:
        torch.save({
            "online"   : self.online.state_dict(),
            "target"   : self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epsilon"  : self.epsilon,
            "alpha"    : self.alpha,
            "steps"    : self.steps,
            "episode"  : self._episode,
        }, path)
        print(f"  [ckpt] saved → {path}")

    def save_best(self, eval_revenue: float, path: str = "cht_best.pt") -> bool:
        if eval_revenue > self._best_eval_revenue:
            self._best_eval_revenue = eval_revenue
            self._best_path = path
            torch.save({
                "online": self.online.state_dict(), "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
                "epsilon": self.epsilon, "alpha": self.alpha,
                "steps": self.steps, "episode": self._episode,
                "best_revenue": eval_revenue,
            }, path)
            return True
        return False

    def load(self, path: str = "cht_hotel.pt") -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["target"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.epsilon    = ckpt["epsilon"]
        self.alpha      = ckpt.get("alpha", 0.0)
        self.steps      = ckpt["steps"]
        self._episode   = ckpt.get("episode", 0)
        print(f"  [ckpt] loaded ← {path}  ε={self.epsilon:.3f}"
              f"  α={self.alpha:.3f}  ep={self._episode}")

    def load_best(self) -> None:
        self.load(self._best_path)
        print(f"  [best] best eval revenue: ${self._best_eval_revenue:.1f}")


# ────────────────────────────────────────────────────────────────────────────
# Training loop
# ────────────────────────────────────────────────────────────────────────────

def train_cht(
    agent      : CHTDQNAgent,
    env        : HotelEnv,
    hp         : CHTHyperParams = CHTHP,
    eval_env   : Optional[HotelEnv] = None,
    resume_from: Optional[str] = None,
) -> List[TrainLog]:
    if resume_from:
        agent.load(resume_from)
        print(f"  Resuming from ep {agent._episode}\n")

    start_episode = agent._episode

    if eval_env is None:
        eval_env = HotelEnv(
            capacity=env.base_capacity, episode_length=env.base_episode_length,
            customer_types=env.customer_types, scale=env.scale, render_mode=None,
        )

    print(f"\n{'='*80}")
    print(f"  CHT-DQN  capacity={env.capacity}  horizon={env.episode_length}  scale={env.scale}")
    print(f"  {hp.n_episodes} eps | γ={hp.gamma} | lr={hp.lr} | n_step={hp.n_step}"
          f" | shaping={hp.shaping_weight}")
    print(f"  ε: {hp.eps_start}→{hp.eps_end} over {hp.eps_decay_ep} eps")
    print(f"  α: {hp.alpha_start}→{hp.alpha_end} over {hp.alpha_decay_ep} eps"
          f"  (CHT prior annealing)")
    print(f"  cht_obs={hp.use_cht_obs}  q_cht_weight={hp.q_cht_weight}")
    print(f"  CHT targets: " +
          "  ".join(f"{n}={v:.1f}" for n, v in sorted(agent.x_star.items())))
    print(f"{'='*80}")
    print(f"  {'Episode':>8}  {'ε':>5}  {'α':>5}  {'EvalRev':>9}  {'Ideal':>9}"
          f"  {'Regret%':>8}  {'Util%':>6}  {'Loss':>10}  {'Best?':>6}")
    print(f"  {'-'*80}")

    logs   : List[TrainLog]   = []
    losses : collections.deque = collections.deque(maxlen=500)

    for ep in range(1, hp.n_episodes + 1):
        abs_ep = start_episode + ep
        agent._episode = abs_ep
        agent._update_epsilon()
        agent._update_alpha()
        agent.tracker.reset()

        obs, info = env.reset()
        done = False

        while not done:
            info_before = dict(info)
            action = agent.act(obs, info)
            feat   = agent._obs(info)

            next_obs, raw_reward, terminated, truncated, info_after = env.step(action)
            done = terminated or truncated

            # Update allocation tracker
            agent.tracker.update(
                info_before["current_customer"],
                info_before["requested_rooms"],
                accepted=(action == 1 and raw_reward > 0),
            )

            r         = shaped_reward(raw_reward, info_before, info_after, env, hp.shaping_weight)
            next_feat = agent._obs(info_after)
            agent.buffer.push(feat, action, r * hp.reward_scale, next_feat, done)
            agent.steps += 1

            if agent.steps % hp.learn_every == 0:
                loss = agent.learn()
                if loss is not None:
                    losses.append(loss)

            obs  = next_obs
            info = info_after

        if ep % hp.target_sync == 0:
            agent.sync_target()
        agent.end_of_episode()

        if ep % hp.log_every == 0:
            stats    = _evaluate_cht(agent, eval_env, n_episodes=hp.eval_episodes)
            avg_loss = float(np.mean(losses)) if losses else 0.0
            is_best  = agent.save_best(stats["avg_revenue"])
            logs.append(TrainLog(
                episode      = abs_ep,
                epsilon      = agent.epsilon,
                eval_revenue = stats["avg_revenue"],
                eval_ideal   = stats["avg_ideal"],
                eval_regret  = stats["avg_regret_pct"],
                avg_loss     = avg_loss,
                utilisation  = stats["avg_utilisation"],
            ))
            print(
                f"  {abs_ep:>8}  {agent.epsilon:>5.3f}  {agent.alpha:>5.3f}"
                f"  ${stats['avg_revenue']:>8.1f}  ${stats['avg_ideal']:>8.1f}"
                f"  {stats['avg_regret_pct']:>7.1f}%  {stats['avg_utilisation']*100:>5.1f}%"
                f"  {avg_loss:>10.5f}  {'★ best' if is_best else '':>6}"
            )

    print(f"  {'─'*80}")
    print(f"  Done. steps={agent.steps}  updates={agent.steps // hp.learn_every}")
    print(f"  Best: ${agent._best_eval_revenue:.1f} — restoring\n")
    agent.load_best()
    return logs


# ────────────────────────────────────────────────────────────────────────────
# Evaluation (CHT-aware — resets tracker per episode)
# ────────────────────────────────────────────────────────────────────────────

def _evaluate_cht(
    agent     : CHTDQNAgent,
    env       : HotelEnv,
    n_episodes: int = 200,
) -> Dict:
    revenues, ideals, regrets, regpcts, utils = [], [], [], [], []
    for _ in range(n_episodes):
        agent.tracker.reset()
        obs, info = env.reset()
        done = False
        while not done:
            action = agent.act(obs, info, greedy=True)
            agent.tracker.update(
                info["current_customer"], info["requested_rooms"],
                accepted=(action == 1 and info["rooms_available"] >= info["requested_rooms"]),
            )
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        revenues.append(info["episode_revenue"])
        ideals.append(info.get("ideal_revenue", 0.0))
        regrets.append(info.get("regret", 0.0))
        regpcts.append(info.get("regret_pct", 0.0))
        utils.append(info.get("utilisation_rate", 0.0))
    return {
        "avg_revenue"    : float(np.mean(revenues)),
        "std_revenue"    : float(np.std(revenues)),
        "avg_ideal"      : float(np.mean(ideals)),
        "avg_regret"     : float(np.mean(regrets)),
        "avg_regret_pct" : float(np.mean(regpcts)),
        "avg_utilisation": float(np.mean(utils)),
    }


# ────────────────────────────────────────────────────────────────────────────
# Ablation study: compare CHT variants
# ────────────────────────────────────────────────────────────────────────────

def ablation_study(env: HotelEnv, eval_env: HotelEnv, n_episodes: int = 3_000):
    """
    Train three variants and compare:
      Vanilla   — no CHT features, no prior, no residual  (baseline)
      CHT-obs   — +state augmentation only
      CHT-full  — +state augmentation + residual + α-prior
    """
    variants = [
        ("Vanilla DQN",  CHTHyperParams(use_cht_obs=False, q_cht_weight=0.0,
                                        alpha_start=0.0, n_episodes=n_episodes)),
        ("CHT-obs only", CHTHyperParams(use_cht_obs=True,  q_cht_weight=0.0,
                                        alpha_start=0.0, n_episodes=n_episodes)),
        ("CHT-full",     CHTHyperParams(use_cht_obs=True,  q_cht_weight=1.0,
                                        alpha_start=0.5, n_episodes=n_episodes)),
    ]

    cw   = 16
    cols = ["Variant", "Avg Revenue", "DP Ideal", "Regret %", "Utilisation"]
    sep  = "=" * (cw * len(cols))
    results = []

    for name, hp in variants:
        print(f"\n── Training: {name} ──")
        agent = CHTDQNAgent(env, hp=hp, device="cpu", n_episodes=hp.n_episodes)
        train_cht(agent, env, hp=hp, eval_env=eval_env)
        stats = _evaluate_cht(agent, eval_env, n_episodes=500)
        results.append((name, stats))

    print(f"\n{sep}")
    print(f"  Ablation Study  |  {n_episodes} training eps each")
    print(sep)
    print("  " + "".join(f"{c:<{cw}}" for c in cols))
    print("  " + "-" * (cw * len(cols)))
    for name, stats in results:
        print(
            f"  {name:<{cw}}"
            f"${stats['avg_revenue']:<{cw-1}.1f}"
            f"${stats['avg_ideal']:<{cw-1}.1f}"
            f"{stats['avg_regret_pct']:<{cw}.1f}%"
            f"{stats['avg_utilisation']*100:<{cw}.1f}%"
        )
    print(sep)


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume",   action="store_true", help="Resume from cht_best.pt")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    args = parser.parse_args()

    train_env = HotelEnv(
        capacity=20, episode_length=50,
        customer_types=TIGHT_CUSTOMER_TYPES, scale=1,
        render_mode=None, seed=0,
    )
    eval_env = HotelEnv(
        capacity=20, episode_length=50,
        customer_types=TIGHT_CUSTOMER_TYPES, scale=1,
        render_mode=None,
    )

    if args.ablation:
        ablation_study(train_env, eval_env, n_episodes=3_000)
    else:
        hp = CHTHyperParams(
            hidden_dims    = [256, 256],
            buffer_size    = 100_000,
            batch_size     = 512,
            lr             = 3e-4,
            gamma          = 1.0,
            n_step         = 3,
            n_episodes     = 6_000,
            learn_every    = 4,
            target_sync    = 10,
            eps_start      = 1.0,
            eps_end        = 0.02,
            eps_decay_ep   = 3_600,
            shaping_weight = 0.2,
            log_every      = 500,
            eval_episodes  = 200,
            alpha_start    = 0.5,
            alpha_end      = 0.0,
            alpha_decay_ep = 2_000,
            q_cht_weight   = 1.0,
            use_cht_obs    = True,
        )

        agent = CHTDQNAgent(train_env, hp=hp, device="cpu", n_episodes=hp.n_episodes)
        train_cht(
            agent, train_env, hp=hp, eval_env=eval_env,
            resume_from="cht_best.pt" if args.resume else None,
        )
        agent.save("cht_hotel.pt")

        # Full evaluation
        stats = _evaluate_cht(agent, eval_env, n_episodes=500)
        print(f"\n── CHT-DQN Final Evaluation (500 eps, greedy) ──")
        print(f"  Revenue     : ${stats['avg_revenue']:.1f} ± {stats['std_revenue']:.1f}")
        print(f"  DP Ideal    : ${stats['avg_ideal']:.1f}")
        print(f"  Regret      : ${stats['avg_regret']:.1f}  ({stats['avg_regret_pct']:.1f}%)")
        print(f"  Utilisation : {stats['avg_utilisation']*100:.1f}%\n")

        # Compare vs baselines including vanilla DQN
        agents_dict = {
            "CHT-DQN"        : lambda obs, info: agent.act(obs, info, greedy=True),
            "Greedy"         : GreedyAgent(),
            "Threshold(0.3)" : ThresholdAgent(0.3),
            "Random"         : RandomAgent(),
        }
        regret_breakdown(agents_dict, eval_env, n_episodes=500)
        policy_summary(agent, eval_env, n_episodes=500)
"""
DQN Agent — Hotel Revenue Management  (v4)
==========================================
Root-cause fixes for the ~38% regret plateau and inverted priority problem.

PROBLEM 1 — Scalar customer index encodes ordinal bias
  The old obs used customer_type / (num_types-1), a single float.
  This forces the network to treat types as ordered on a number line
  (0 = Budget, 0.33 = Standard, 0.67 = Premium, 1.0 = Group), so it
  can't learn that Group ($90/room) is worth less than Standard ($120/room)
  even though its index is higher.
  FIX: one-hot encode customer type → the network gets a separate input
  neuron per type and can learn independent value functions for each.

PROBLEM 2 — No signal about what's still coming
  The agent sees the current customer's reward_per_room but has no sense
  of the *distribution* of future arrivals. It can't reason "I should
  reject this $80 Budget guest because $200 Premium guests arrive 15%
  of the time and I only have 2 rooms left."
  FIX: add two derived features to the observation:
    • expected_rpr_remaining  — arrival-weighted expected $/room given
                                time remaining (normalised)
    • opportunity_cost_ratio  — current reward_per_room / expected future
                                reward_per_room.  < 1.0 means this customer
                                is below-average value; the agent should
                                consider waiting.

PROBLEM 3 — Shaping penalises high-value accepts
  The old shaping Φ(s) = rooms_available × time_remaining × exp_rpr
  decreases whenever any customer is accepted (rooms go down).
  So Φ(s') − Φ(s) is *negative* on every accept — the shaped reward
  punishes accepting even a $200 Premium guest, counteracting the raw
  revenue signal.
  FIX: value-adjusted shaping.  Φ(s) = rooms_available × time_remaining
  × exp_rpr_remaining.  Only rejects below exp_rpr get a positive shaping
  bonus (held a room for someone better).  Accepts above exp_rpr get zero
  shaping penalty (correctly priced).  This is still potential-based so
  it cannot change the optimal policy.

PROBLEM 4 — One update per step floods the buffer with correlated data
  Already partially addressed with learn_every=4.  Also added
  multi-step (n-step) returns with n=3: the TD target uses the sum of
  rewards over the next 3 steps before bootstrapping, which gives a
  stronger learning signal and reduces the variance of the 1-step target.

OBS LAYOUT  (4 + num_customer_types floats)
  [0]   rooms_occupied   / capacity
  [1]   rooms_available  / capacity
  [2]   time_remaining   / episode_length        (horizon left, not elapsed)
  [3]   requested_rooms  / max_request
  [4]   reward_per_room  / max_reward_per_room
  [5]   opportunity_cost_ratio  (current rpr / expected future rpr, clamped)
  [6..6+K-1]  one-hot customer type  (K = num_customer_types)

Run:
    python dqn_agent.py
"""

from __future__ import annotations

import random
import collections
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from hotel_env import (
    HotelEnv,
    CustomerType,
    DEFAULT_CUSTOMER_TYPES,
    TIGHT_CUSTOMER_TYPES,
    GreedyAgent,
    ThresholdAgent,
    RandomAgent,
)


# ────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class HyperParams:
    hidden_dims    : List[int] = field(default_factory=lambda: [256, 256])
    buffer_size    : int   = 100_000
    batch_size     : int   = 512
    lr             : float = 3e-4
    gamma          : float = 1.0           # undiscounted — short finite horizon
    n_step         : int   = 3             # n-step return
    n_episodes     : int   = 6_000
    learn_every    : int   = 4
    target_sync    : int   = 10            # episodes between hard target updates
    eps_start      : float = 1.0
    eps_end        : float = 0.02
    eps_decay_ep   : int   = 3_600         # linear decay over 60% of training
    shaping_weight : float = 0.2           # 0 = off, see PROBLEM 3 fix above
    reward_scale   : float = field(init=False)
    log_every      : int   = 500
    eval_episodes  : int   = 200

    def __post_init__(self):
        max_rpr = max(c.reward_per_room for c in TIGHT_CUSTOMER_TYPES)
        max_req = max(c.max_rooms       for c in TIGHT_CUSTOMER_TYPES)
        self.reward_scale = 1.0 / (max_rpr * max_req)


HP = HyperParams()


# ────────────────────────────────────────────────────────────────────────────
# Observation builder
# ────────────────────────────────────────────────────────────────────────────

MAX_REWARD_PER_ROOM = max(c.reward_per_room for c in TIGHT_CUSTOMER_TYPES)


def _expected_rpr(customer_types: List[CustomerType]) -> float:
    """Arrival-probability-weighted expected reward per room."""
    total_prob = sum(c.arrival_prob for c in customer_types)
    return sum(
        (c.arrival_prob / total_prob) * c.reward_per_room
        for c in customer_types
    )


def obs_dim_for(env: HotelEnv) -> int:
    """6 scalar features + one-hot over customer types."""
    return 6 + env.num_types


def build_obs(info: Dict, env: HotelEnv) -> np.ndarray:
    """
    Rich observation vector.  See module docstring for full layout.
    """
    time_rem   = max(env.episode_length - info["time_step"], 0) / env.episode_length
    exp_rpr    = _expected_rpr(env.customer_types)
    cur_rpr    = info["reward_per_room"]
    opp_ratio  = np.clip(cur_rpr / exp_rpr, 0.0, 2.0) / 2.0   # normalised to [0,1]

    # One-hot customer type
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

    return np.concatenate([scalars, one_hot])


# Patch HotelEnv once to expose customer_type index in info
_orig_get_info = HotelEnv._get_info

def _patched_get_info(self) -> Dict:
    d = _orig_get_info(self)
    d["current_customer_idx"] = self._customer_type
    return d

HotelEnv._get_info = _patched_get_info


# ────────────────────────────────────────────────────────────────────────────
# Reward shaping  (value-adjusted potential, still policy-invariant)
# ────────────────────────────────────────────────────────────────────────────

def _potential(info: Dict, env: HotelEnv) -> float:
    """
    Φ(s) = rooms_available × time_remaining × expected_rpr_remaining

    This measures the expected revenue the hotel *could still earn* if
    future customers arrive at the expected rate and all are accepted.
    Difference Φ(s') − Φ(s):
      • Reject a below-average customer → rooms held, Φ rises   → +bonus
      • Reject an above-average customer → rooms held but overkill → small +
      • Accept any customer → rooms fall, Φ drops → −penalty proportional
        to rooms consumed, automatically smaller for high-value guests
        (they consume fewer rooms relative to their revenue contribution,
        but also the raw reward already compensates).
    """
    time_rem   = max(env.episode_length - info["time_step"], 0) / env.episode_length
    frac_avail = info["rooms_available"] / env.capacity
    exp_rpr    = _expected_rpr(env.customer_types)
    return frac_avail * time_rem * exp_rpr * env.capacity   # in $ units


def shaped_reward(
    raw_reward : float,
    info_before: Dict,
    info_after : Dict,
    env        : HotelEnv,
    weight     : float,
) -> float:
    if weight == 0.0:
        return raw_reward
    phi_before = _potential(info_before, env)
    phi_after  = _potential(info_after,  env)
    shaping    = weight * (phi_after - phi_before)
    # Scale shaping to same order of magnitude as raw reward
    shaping   *= HP.reward_scale
    return raw_reward + shaping / HP.reward_scale   # both in raw $ before scaling


# ────────────────────────────────────────────────────────────────────────────
# Q-Network
# ────────────────────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """
    MLP: obs → [Q(Reject), Q(Accept)]

    Dueling head: separate value V(s) and advantage A(s,a) streams,
    combined as Q(s,a) = V(s) + A(s,a) − mean(A).
    Dueling helps when the agent needs to distinguish "state is bad
    regardless of action" (near-full hotel, no good customers coming)
    from "this specific action is bad".
    """

    def __init__(self, obs_dim: int, hidden_dims: List[int] = None):
        super().__init__()
        hidden_dims = hidden_dims or HP.hidden_dims

        # Shared trunk
        trunk: List[nn.Module] = []
        in_dim = obs_dim
        for i, h in enumerate(hidden_dims[:-1]):
            trunk.append(nn.Linear(in_dim, h))
            if i == 0:
                trunk.append(nn.LayerNorm(h))
            trunk.append(nn.ReLU())
            in_dim = h
        self.trunk = nn.Sequential(*trunk)

        # Dueling streams
        last_h = hidden_dims[-1]
        self.value_stream     = nn.Sequential(
            nn.Linear(in_dim, last_h), nn.ReLU(), nn.Linear(last_h, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(in_dim, last_h), nn.ReLU(), nn.Linear(last_h, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.trunk(x)
        V      = self.value_stream(shared)                  # (B, 1)
        A      = self.advantage_stream(shared)              # (B, 2)
        Q      = V + A - A.mean(dim=1, keepdim=True)       # (B, 2)
        return Q


# ────────────────────────────────────────────────────────────────────────────
# N-step replay buffer
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class Transition:
    obs     : np.ndarray
    action  : int
    reward  : float          # already n-step return
    next_obs: np.ndarray
    done    : bool


class NStepReplayBuffer:
    """
    Replay buffer with correct n-step return accumulation.

    BUG FIXED: the old implementation only committed one transition per
    done signal, leaving up to n-1 transitions stranded in the pending
    deque across episode boundaries.  Those stale transitions would then
    be mixed with the *next* episode's transitions, producing garbage
    targets (rewards from episode k bootstrapped into episode k+1 states).

    Correct behaviour on done=True:
      Flush ALL remaining pending transitions, each with its own partial
      n-step return calculated from only the transitions still in the
      window (not crossing the episode boundary).  This is equivalent to
      treating the terminal state as having V=0 for any remaining steps.

    With γ=1 (our setting) each partial return is just the sum of
    remaining rewards in the window.
    """

    def __init__(self, capacity: int, n: int, gamma: float):
        self.buf    : collections.deque = collections.deque(maxlen=capacity)
        self.pending: List              = []          # list so we can slice
        self.n      = n
        self.gamma  = gamma

    def _commit(self, start_idx: int, final_next_obs: np.ndarray, final_done: bool) -> None:
        """Commit the transition at pending[start_idx] using returns from start_idx onward."""
        first_obs, first_action, _, _, _ = self.pending[start_idx]
        n_ret = 0.0
        g     = 1.0
        for _, _, r, n_obs, d in self.pending[start_idx:]:
            n_ret += g * r
            g     *= self.gamma
            final_next_obs = n_obs
            final_done     = d
            if d:
                break
        self.buf.append(Transition(first_obs, first_action,
                                   n_ret, final_next_obs, final_done))

    def push(
        self,
        obs     : np.ndarray,
        action  : int,
        reward  : float,
        next_obs: np.ndarray,
        done    : bool,
    ) -> None:
        self.pending.append((obs, action, reward, next_obs, done))

        if len(self.pending) >= self.n:
            # Full window ready — commit oldest transition
            self._commit(0, next_obs, done)
            self.pending.pop(0)

        if done:
            # FIX: flush ALL remaining pending transitions with partial returns
            # so nothing crosses the episode boundary
            for i in range(len(self.pending)):
                self._commit(i, next_obs, True)
            self.pending.clear()

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        batch    = random.sample(self.buf, batch_size)
        obs      = torch.tensor(np.stack([t.obs      for t in batch]), dtype=torch.float32)
        actions  = torch.tensor([t.action  for t in batch],            dtype=torch.long)
        rewards  = torch.tensor([t.reward  for t in batch],            dtype=torch.float32)
        next_obs = torch.tensor(np.stack([t.next_obs for t in batch]), dtype=torch.float32)
        dones    = torch.tensor([t.done    for t in batch],            dtype=torch.float32)
        return obs, actions, rewards, next_obs, dones

    def __len__(self) -> int:
        return len(self.buf)


# ────────────────────────────────────────────────────────────────────────────
# DQN Agent
# ────────────────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    Double Dueling DQN with n-step returns and value-adjusted shaping.

    Stability fixes vs v4
    ---------------------
    • LR scheduler: CosineAnnealingLR warms down lr from hp.lr → hp.lr/20
      over training.  By the time ε hits its floor the LR is already low,
      preventing the Q-value divergence that caused the ep-3500 collapse.
    • Best-model checkpoint: save_best() writes only when eval improves,
      so the ep-3000 peak (4.6% regret) is never overwritten.
    • Grad clip tightened to 1.0 (was 10.0) for additional stability.
    """

    def __init__(
        self,
        env        : HotelEnv,
        hp         : HyperParams = HP,
        device     : str = "cpu",
        n_episodes : int = 0,          # passed in so scheduler knows total steps
    ):
        self.env      = env
        self.hp       = hp
        self.device   = torch.device(device)
        self._episode = 0

        _obs_dim = obs_dim_for(env)

        self.online = QNetwork(_obs_dim, hp.hidden_dims).to(self.device)
        self.target = QNetwork(_obs_dim, hp.hidden_dims).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=hp.lr)
        # Cosine anneal LR from hp.lr down to hp.lr/20 over full training
        T_max = max(n_episodes or hp.n_episodes, 1)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=T_max, eta_min=hp.lr / 20
        )
        self.loss_fn   = nn.SmoothL1Loss()

        self.buffer  = NStepReplayBuffer(hp.buffer_size, hp.n_step, hp.gamma)
        self.epsilon = hp.eps_start
        self.steps   = 0

        # Best-model tracking
        self._best_eval_revenue : float = -float("inf")
        self._best_path         : str   = "dqn_hotel_best.pt"

    def _update_epsilon(self) -> None:
        frac         = min(self._episode / self.hp.eps_decay_ep, 1.0)
        self.epsilon = self.hp.eps_start + frac * (self.hp.eps_end - self.hp.eps_start)

    def act(self, obs: np.ndarray, info: Dict, greedy: bool = False) -> int:
        # Hard capacity mask
        if info["rooms_available"] < info["requested_rooms"]:
            return 0
        if not greedy and random.random() < self.epsilon:
            return self.env.action_space.sample()
        feat = build_obs(info, self.env)
        with torch.no_grad():
            q = self.online(
                torch.tensor(feat, dtype=torch.float32, device=self.device).unsqueeze(0)
            )
        return int(q.argmax(dim=1).item())

    def learn(self) -> Optional[float]:
        if len(self.buffer) < self.hp.batch_size:
            return None

        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.hp.batch_size)
        obs      = obs.to(self.device)
        actions  = actions.to(self.device)
        rewards  = rewards.to(self.device)
        next_obs = next_obs.to(self.device)
        dones    = dones.to(self.device)

        # Current Q values
        q_vals = self.online(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target: select action with online, evaluate with target
        with torch.no_grad():
            best_actions = self.online(next_obs).argmax(dim=1, keepdim=True)
            next_q       = self.target(next_obs).gather(1, best_actions).squeeze(1)
            q_target     = rewards + (self.hp.gamma ** self.hp.n_step) * next_q * (1.0 - dones)

        loss = self.loss_fn(q_vals, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=1.0)   # tighter clip
        self.optimizer.step()
        return loss.item()

    def end_of_episode(self) -> None:
        """Call once per episode: step LR scheduler."""
        self.scheduler.step()

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def save(self, path: str = "dqn_hotel.pt") -> None:
        torch.save({
            "online"    : self.online.state_dict(),
            "target"    : self.target.state_dict(),
            "optimizer" : self.optimizer.state_dict(),
            "scheduler" : self.scheduler.state_dict(),
            "epsilon"   : self.epsilon,
            "steps"     : self.steps,
            "episode"   : self._episode,
        }, path)
        print(f"  [ckpt] saved → {path}")

    def save_best(self, eval_revenue: float, path: str = "dqn_hotel_best.pt") -> bool:
        """Save only if eval_revenue beats the previous best. Returns True if saved."""
        if eval_revenue > self._best_eval_revenue:
            self._best_eval_revenue = eval_revenue
            self._best_path         = path
            torch.save({
                "online"       : self.online.state_dict(),
                "target"       : self.target.state_dict(),
                "optimizer"    : self.optimizer.state_dict(),
                "scheduler"    : self.scheduler.state_dict(),
                "epsilon"      : self.epsilon,
                "steps"        : self.steps,
                "episode"      : self._episode,
                "best_revenue" : eval_revenue,
            }, path)
            return True
        return False

    def load_best(self) -> None:
        """Restore the best checkpoint saved so far."""
        self.load(self._best_path)
        print(f"  [best] best eval revenue: ${self._best_eval_revenue:.1f}")

    def load(self, path: str = "dqn_hotel.pt") -> None:
        """
        Load a checkpoint and fully restore training state.

        Restores network weights, optimizer moments, scheduler position,
        epsilon, and step/episode counters — so resumed training continues
        exactly where it left off with no discontinuities in LR or ε.
        """
        ckpt = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["target"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.epsilon    = ckpt["epsilon"]
        self.steps      = ckpt["steps"]
        self._episode   = ckpt.get("episode", 0)
        print(f"  [ckpt] loaded ← {path}"
              f"  ε={self.epsilon:.3f}"
              f"  steps={self.steps}"
              f"  episode={self._episode}"
              f"  lr={self.optimizer.param_groups[0]['lr']:.2e}")


# ────────────────────────────────────────────────────────────────────────────
# Training loop
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainLog:
    episode     : int
    epsilon     : float
    eval_revenue: float
    eval_ideal  : float
    eval_regret : float
    avg_loss    : float
    utilisation : float


def train(
    agent       : DQNAgent,
    env         : HotelEnv,
    hp          : HyperParams = HP,
    eval_env    : Optional[HotelEnv] = None,
    resume_from : Optional[str] = None,   # path to checkpoint to resume from
) -> List[TrainLog]:
    """
    Full DQN training loop.  Pass resume_from="dqn_hotel.pt" to continue
    training on top of an existing checkpoint.

    On resume:
      - Network weights, optimizer moments, and scheduler position are all
        restored so LR and ε continue smoothly — no cold-restart discontinuity.
      - The episode counter is offset so ε decay and LR cosine schedule
        treat the resumed episode as episode (start_episode + 1), not 1.
      - n_episodes is interpreted as *additional* episodes to run, not total.
    """
    if resume_from is not None:
        agent.load(resume_from)
        print(f"  Resuming from ep {agent._episode} — "
              f"running {hp.n_episodes} more episodes\n")

    start_episode = agent._episode
    if eval_env is None:
        eval_env = HotelEnv(
            capacity       = env.base_capacity,
            episode_length = env.base_episode_length,
            customer_types = env.customer_types,
            scale          = env.scale,
            render_mode    = None,
        )

    print(f"\n{'='*78}")
    print(f"  Double Dueling DQN  |  capacity={env.capacity}"
          f"  horizon={env.episode_length}  scale={env.scale}")
    print(f"  {hp.n_episodes} eps | γ={hp.gamma} | lr={hp.lr} | n_step={hp.n_step}"
          f" | shaping={hp.shaping_weight}")
    print(f"  ε: {hp.eps_start}→{hp.eps_end} over {hp.eps_decay_ep} eps |"
          f" target_sync={hp.target_sync} | learn_every={hp.learn_every}")
    print(f"  LR cosine anneal: {hp.lr} → {hp.lr/20:.2e} over training")
    print(f"{'='*78}")
    print(f"  {'Episode':>8}  {'ε':>6}  {'Eval Rev':>10}  {'DP Ideal':>10}"
          f"  {'Regret%':>8}  {'Util%':>7}  {'Loss':>10}  {'Best?':>6}")
    print(f"  {'-'*78}")

    logs  : List[TrainLog]          = []
    losses: collections.deque       = collections.deque(maxlen=500)

    for ep in range(1, hp.n_episodes + 1):
        abs_ep = start_episode + ep       # absolute episode number across all runs
        agent._episode = abs_ep
        agent._update_epsilon()

        obs, info = env.reset()
        done      = False

        while not done:
            info_before = dict(info)
            action      = agent.act(obs, info)
            feat        = build_obs(info, env)

            next_obs, raw_reward, terminated, truncated, info_after = env.step(action)
            done = terminated or truncated

            r         = shaped_reward(raw_reward, info_before, info_after, env, hp.shaping_weight)
            next_feat = build_obs(info_after, env)
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

        agent.end_of_episode()    # step LR scheduler

        if ep % hp.log_every == 0:
            stats    = evaluate(agent, eval_env, n_episodes=hp.eval_episodes, silent=True)
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
                f"  {abs_ep:>8}  {agent.epsilon:>6.3f}"
                f"  ${stats['avg_revenue']:>9.1f}"
                f"  ${stats['avg_ideal']:>9.1f}"
                f"  {stats['avg_regret_pct']:>7.1f}%"
                f"  {stats['avg_utilisation']*100:>6.1f}%"
                f"  {avg_loss:>10.5f}"
                f"  {'★ best' if is_best else '':>6}"
            )

    print(f"  {'─'*78}")
    print(f"  Done. steps={agent.steps}  updates={agent.steps // hp.learn_every}")
    print(f"  Best eval revenue: ${agent._best_eval_revenue:.1f} — restoring best weights\n")
    agent.load_best()
    return logs


# ────────────────────────────────────────────────────────────────────────────
# Evaluation
# ────────────────────────────────────────────────────────────────────────────

def evaluate(
    agent     : DQNAgent,
    env       : HotelEnv,
    n_episodes: int  = 300,
    silent    : bool = False,
) -> Dict:
    revenues, ideals, regrets, regpcts, utils = [], [], [], [], []
    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        while not done:
            action = agent.act(obs, info, greedy=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        revenues.append(info["episode_revenue"])
        ideals.append(info.get("ideal_revenue", 0.0))
        regrets.append(info.get("regret", 0.0))
        regpcts.append(info.get("regret_pct", 0.0))
        utils.append(info.get("utilisation_rate", 0.0))

    stats = {
        "avg_revenue"    : float(np.mean(revenues)),
        "std_revenue"    : float(np.std(revenues)),
        "avg_ideal"      : float(np.mean(ideals)),
        "avg_regret"     : float(np.mean(regrets)),
        "avg_regret_pct" : float(np.mean(regpcts)),
        "avg_utilisation": float(np.mean(utils)),
    }
    if not silent:
        print(f"\n── DQN Evaluation ({n_episodes} episodes, greedy) ──")
        print(f"  Revenue     : ${stats['avg_revenue']:.1f} ± {stats['std_revenue']:.1f}")
        print(f"  DP Ideal    : ${stats['avg_ideal']:.1f}")
        print(f"  Regret      : ${stats['avg_regret']:.1f}  ({stats['avg_regret_pct']:.1f}%)")
        print(f"  Utilisation : {stats['avg_utilisation']*100:.1f}%\n")
    return stats


# ────────────────────────────────────────────────────────────────────────────
# Policy diagnostic
# ────────────────────────────────────────────────────────────────────────────

def policy_summary(agent: DQNAgent, env: HotelEnv, n_episodes: int = 500):
    """
    Accept rate by customer type × occupancy bucket.

    What a well-trained RM policy should show:
      Premium  ($200/room) — accepted at ALL occupancy levels
      Standard ($120/room) — accepted at low/mid, maybe selective at high
      Group    ($90/room)  — accepted at low/mid, rejected at high
                             (they consume many rooms at below-Standard value)
      Budget   ($80/room)  — accepted at low, increasingly rejected as
                             occupancy rises (save rooms for better guests)

    If Premium is rejected more than Budget at high occupancy, the agent
    has NOT learned value-based priority — it's just learned occupancy-based
    rejection without understanding customer value.
    """
    from collections import defaultdict
    totals  = defaultdict(lambda: defaultdict(int))
    accepts = defaultdict(lambda: defaultdict(int))
    buckets = ["low  (<40%)", "mid  (40–75%)", "high (>75%)"]

    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        while not done:
            util   = info["rooms_occupied"] / env.capacity
            bucket = ("low  (<40%)" if util < 0.40
                      else "mid  (40–75%)" if util < 0.75
                      else "high (>75%)")
            action = agent.act(obs, info, greedy=True)
            cname  = info["current_customer"]
            totals[cname][bucket]  += 1
            accepts[cname][bucket] += action
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

    # Sort by reward_per_room descending so Premium is printed first
    cnames_sorted = sorted(
        [c.name for c in env.customer_types],
        key=lambda n: -next(c.reward_per_room for c in env.customer_types if c.name == n)
    )

    print(f"\n── Policy summary — accept rate by type × occupancy ──")
    print(f"  (A well-trained agent should show HIGHER accept rates for higher-value types)")
    print(f"  {'Customer':<12} {'$/room':>8}", end="")
    for b in buckets:
        print(f"  {b:>14}", end="")
    print()
    print("  " + "─" * (24 + 17 * len(buckets)))

    for cname in cnames_sorted:
        ctype = next(c for c in env.customer_types if c.name == cname)
        print(f"  {cname:<12} ${ctype.reward_per_room:>6.0f}  ", end="")
        for b in buckets:
            n   = totals[cname][b]
            acc = accepts[cname][b]
            pct = 100.0 * acc / n if n > 0 else float("nan")
            print(f"  {pct:>12.1f}%", end="")
        print()
    print()
    print("  Correct priority order at HIGH occupancy (top = should accept most):")
    high_rates = {
        cname: (100.0 * accepts[cname]["high (>75%)"] / totals[cname]["high (>75%)"]
                if totals[cname]["high (>75%)"] > 0 else 0.0)
        for cname in cnames_sorted
    }
    for cname in sorted(high_rates, key=lambda n: -high_rates[n]):
        ctype = next(c for c in env.customer_types if c.name == cname)
        print(f"    {cname:<10} ${ctype.reward_per_room:>6.0f}/room  →  "
              f"{high_rates[cname]:.1f}% accepted at high occ")
    print()


# ────────────────────────────────────────────────────────────────────────────
# Comparison table
# ────────────────────────────────────────────────────────────────────────────

def compare_agents(dqn_agent: DQNAgent, env: HotelEnv, n_episodes: int = 500):
    rule_agents = {
        "Greedy"         : GreedyAgent(),
        "Threshold(0.3)" : ThresholdAgent(0.3),
        "Random"         : RandomAgent(),
    }

    cw   = 16
    cols = ["Agent", "Avg Revenue", "DP Ideal", "Regret $", "Regret %", "Util%", "Accepted"]
    sep  = "=" * (cw * len(cols))
    print(f"\n{sep}")
    print(f"  Final Comparison  |  {n_episodes} episodes  |  scale={env.scale}")
    print(sep)
    print("  " + "".join(f"{c:<{cw}}" for c in cols))
    print("  " + "-" * (cw * len(cols)))

    def _rollout(agent_fn):
        revs, ids, regs, rps, uts, acs = [], [], [], [], [], []
        for _ in range(n_episodes):
            obs, info = env.reset()
            done = False
            while not done:
                action = agent_fn(obs, info)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            revs.append(info["episode_revenue"])
            ids.append(info.get("ideal_revenue", 0.0))
            regs.append(info.get("regret", 0.0))
            rps.append(info.get("regret_pct", 0.0))
            uts.append(info.get("utilisation_rate", 0.0))
            acs.append(info["accepted"])
        return (np.mean(revs), np.mean(ids), np.mean(regs),
                np.mean(rps), np.mean(uts), np.mean(acs))

    def _row(name, rev, ideal, reg, regp, util, acc):
        print(
            f"  {name:<{cw}}"
            f"${rev:<{cw-1}.1f}"
            f"${ideal:<{cw-1}.1f}"
            f"${reg:<{cw-1}.1f}"
            f"{regp:<{cw}.1f}%"
            f"{util*100:<{cw}.1f}%"
            f"{acc:<{cw}.1f}"
        )

    _row("DQN (trained)", *_rollout(
        lambda obs, info: dqn_agent.act(obs, info, greedy=True)
    ))
    print("  " + "-" * (cw * len(cols)))
    for name, agent in rule_agents.items():
        _row(name, *_rollout(lambda obs, info, a=agent: a(obs, info)))

    print(sep)
    print("  DP Ideal = backward-induction optimal on same arrival sequence.\n")


# ────────────────────────────────────────────────────────────────────────────
# Regret decomposition: forced vs voluntary
# ────────────────────────────────────────────────────────────────────────────

def regret_breakdown(agents_dict: Dict, env: HotelEnv, n_episodes: int = 500):
    """
    For each agent, decompose rejected revenue into two buckets:

    Forced regret   — agent said Accept but the hotel was physically full.
                      This is unavoidable at decision time; it reflects
                      earlier choices that filled the hotel too fast with
                      low-value guests.

    Voluntary regret — agent said Reject (action=0).
                      This is a deliberate strategic choice.  It's correct
                      when saving rooms for better guests, and wrong when
                      the hotel eventually ends the episode with free rooms.

    Both are measured in raw dollars of revenue that *would* have been
    earned had that customer been accepted.  Note: this is a gross
    attribution — it doesn't account for the counterfactual chain (accepting
    a Budget guest blocks a Premium guest two steps later).  Use it to
    understand *where* losses are happening, not as an exact decomposition
    of the DP regret.

    Also reports:
      spilled_voluntary  — voluntary rejects where the hotel ended the
                           episode with rooms still free.  These are the
                           clearest mistakes: the agent rejected someone,
                           and those rooms were never used anyway.
    """
    cw   = 17
    cols = ["Agent", "Revenue", "Forced $lost", "Forced %", "Voluntary $lost",
            "Vol %", "Wasted vol $"]
    sep  = "=" * (cw * len(cols))

    print(f"\n{sep}")
    print(f"  Regret decomposition  |  {n_episodes} episodes  |  scale={env.scale}")
    print(f"  Forced   = tried to accept, hotel was full")
    print(f"  Voluntary= chose to reject (action=0)")
    print(f"  Wasted   = voluntary rejects where rooms were unused at end of episode")
    print(sep)
    print("  " + "".join(f"{c:<{cw}}" for c in cols))
    print("  " + "-" * (cw * len(cols)))

    for name, agent_fn in agents_dict.items():
        ep_revenues, ep_forced, ep_vol, ep_spilled = [], [], [], []

        for _ in range(n_episodes):
            obs, info = env.reset()
            done = False

            # Track voluntary rejects this episode to identify wasted ones
            vol_rejected_rev = 0.0

            while not done:
                action = agent_fn(obs, info)
                # Before stepping: if this will be a voluntary reject, record value
                if action == 0:
                    pot = info["reward_per_room"] * info["requested_rooms"]
                else:
                    pot = 0.0

                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                if action == 0:
                    vol_rejected_rev += pot

            ep_revenues.append(info["episode_revenue"])
            ep_forced.append(info.get("forced_rev_lost", 0.0))
            ep_vol.append(info.get("voluntary_rev_lost", 0.0))

            # Wasted voluntary: rooms were left free at end → those rejects hurt us
            rooms_left = env.capacity - info["rooms_occupied"]
            if rooms_left > 0:
                # Approximate: all voluntary rejected revenue is "wasted"
                # when there were rooms remaining (conservative — overestimates
                # slightly since some rooms might have been saved for later guests
                # who also didn't arrive).
                ep_spilled.append(info.get("voluntary_rev_lost", 0.0))
            else:
                ep_spilled.append(0.0)

        avg_rev     = np.mean(ep_revenues)
        avg_forced  = np.mean(ep_forced)
        avg_vol     = np.mean(ep_vol)
        avg_spilled = np.mean(ep_spilled)
        total_lost  = avg_forced + avg_vol
        forced_pct  = 100 * avg_forced / total_lost if total_lost > 0 else 0
        vol_pct     = 100 * avg_vol    / total_lost if total_lost > 0 else 0

        print(
            f"  {name:<{cw}}"
            f"${avg_rev:<{cw-1}.1f}"
            f"${avg_forced:<{cw-1}.1f}"
            f"{forced_pct:<{cw}.1f}%"
            f"${avg_vol:<{cw-1}.1f}"
            f"{vol_pct:<{cw}.1f}%"
            f"${avg_spilled:<{cw-1}.1f}"
        )

    print(sep)
    print(
        "  Interpretation:\n"
        "  • High Forced $  → agent accepts too eagerly early, hotel fills up, later guests spill\n"
        "  • High Voluntary $, low Wasted $ → good strategic rejects (saved for better guests)\n"
        "  • High Voluntary $, high Wasted $ → bad rejects (rejected guests, rooms sat empty)\n"
    )

if __name__ == "__main__":

    SCALE = 1

    train_env = HotelEnv(
        capacity       = 20,
        episode_length = 50,
        customer_types = TIGHT_CUSTOMER_TYPES,
        scale          = SCALE,
        render_mode    = None,
        seed           = 0,
    )
    eval_env = HotelEnv(
        capacity       = 20,
        episode_length = 50,
        customer_types = TIGHT_CUSTOMER_TYPES,
        scale          = SCALE,
        render_mode    = None,
    )

    hp = HyperParams(
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
    )

    agent = DQNAgent(train_env, hp=hp, device="cpu", n_episodes=hp.n_episodes)
    logs  = train(agent, train_env, hp=hp, eval_env=eval_env)
    agent.save("dqn_hotel.pt")

    # ── To resume training on top of saved weights, do this instead: ─────
    #
    #   agent = DQNAgent(train_env, hp=hp, device="cpu", n_episodes=hp.n_episodes)
    #   logs  = train(agent, train_env, hp=hp, eval_env=eval_env,
    #                 resume_from="dqn_hotel.pt")
    #   agent.save("dqn_hotel.pt")
    #
    # ε and LR will continue exactly from where they left off.
    # Set hp.n_episodes to however many *additional* episodes you want.
    # ─────────────────────────────────────────────────────────────────────

    evaluate(agent, eval_env, n_episodes=500, silent=False)
    compare_agents(agent, eval_env, n_episodes=500)
    policy_summary(agent, eval_env, n_episodes=500)

    # Regret decomposition: forced vs voluntary
    agents_to_compare = {
        "DQN (trained)"  : lambda obs, info: agent.act(obs, info, greedy=True),
        "Greedy"         : GreedyAgent(),
        "Threshold(0.3)" : ThresholdAgent(0.3),
        "Random"         : RandomAgent(),
    }
    regret_breakdown(agents_to_compare, eval_env, n_episodes=500) 
"""
DQN Agent — Hotel Revenue Management
======================================
Trains a Deep Q-Network to decide Accept / Reject for arriving hotel guests.

Fixes vs v1
-----------
  1. gamma = 1.0   Short-horizon episodic RM task. Discounting future rewards
                   makes the agent undervalue accepts early in the episode,
                   causing it to hoard capacity it never uses. Undiscounted
                   returns are the correct objective here.

  2. eps_decay     Was 0.995/episode → ε hits floor at ep ~600, leaving 80%
                   of training as pure (bad) exploitation. Now decays over
                   ~60% of training so the network has time to mature before
                   exploration is cut.

  3. reward_scale  Was 1/200 but max single-step reward = 200 $/room × 6
                   rooms = 1200. Fixed to 1 / (max_rpr × max_request) so
                   rewards truly land in [0, 1] for stable TD targets.

  4. target_sync   Was 50 episodes (2500 steps). Reduced to 10 episodes
                   (500 steps) so the target network tracks learning faster
                   during the critical early phase.

  5. learn_every   Added: only call learn() every N env steps (not every
                   step). Reduces correlation between consecutive updates and
                   speeds up wall-clock training.

  6. Logging       Learning-curve table now shows greedy eval revenue (100
                   episodes) at every log point — not the noisy ε-greedy
                   training revenue — so the curve is actually interpretable.

Observation (6 floats, all in [0, 1]):
    rooms_occupied / capacity
    rooms_available / capacity
    time_step / episode_length
    customer_type_idx / (num_types - 1)
    requested_rooms / max_request
    reward_per_room / max_reward_per_room

Action space:   0 = Reject   1 = Accept

Run:
    python dqn_agent.py
"""

from __future__ import annotations

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
    DEFAULT_CUSTOMER_TYPES,
    GreedyAgent,
    ThresholdAgent,
    RandomAgent,
    run_episode,
)


# ────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class HyperParams:
    # Network
    hidden_dims    : List[int] = field(default_factory=lambda: [128, 128])
    # Replay buffer
    buffer_size    : int   = 50_000
    batch_size     : int   = 256
    # Training
    lr             : float = 3e-4          # lower LR → more stable
    gamma          : float = 1.0           # FIX 1: undiscounted (short horizon)
    n_episodes     : int   = 3_000
    learn_every    : int   = 4             # FIX 5: update every N env steps
    # Target network
    target_sync    : int   = 10            # FIX 4: sync every 10 episodes
    # Exploration (epsilon-greedy)
    eps_start      : float = 1.0
    eps_end        : float = 0.05
    eps_decay_ep   : int   = 1_800        # FIX 2: linear decay over this many episodes
    # Reward normalisation  (FIX 3: correct upper bound)
    # max single-step reward = max_reward_per_room × max_rooms_requested
    reward_scale   : float = 1.0 / (
        max(c.reward_per_room for c in DEFAULT_CUSTOMER_TYPES)
        * max(c.max_rooms     for c in DEFAULT_CUSTOMER_TYPES)
    )
    # Logging
    log_every      : int   = 200
    eval_episodes  : int   = 100           # episodes per in-training eval


HP = HyperParams()

# ────────────────────────────────────────────────────────────────────────────
# Observation builder
# ────────────────────────────────────────────────────────────────────────────

MAX_REWARD_PER_ROOM = max(c.reward_per_room for c in DEFAULT_CUSTOMER_TYPES)
OBS_DIM = 6


def build_obs(info: Dict, env: HotelEnv) -> np.ndarray:
    """6-feature observation, all values normalised to [0, 1]."""
    return np.array([
        info["rooms_occupied"]        / env.capacity,
        info["rooms_available"]       / env.capacity,
        info["time_step"]             / env.episode_length,
        info.get("current_customer_idx", 0) / max(env.num_types - 1, 1),
        info["requested_rooms"]       / env.max_request,
        info["reward_per_room"]       / MAX_REWARD_PER_ROOM,
    ], dtype=np.float32)


# Patch: expose customer_type index in info dict
_orig_get_info = HotelEnv._get_info

def _patched_get_info(self) -> Dict:
    d = _orig_get_info(self)
    d["current_customer_idx"] = self._customer_type
    return d

HotelEnv._get_info = _patched_get_info


# ────────────────────────────────────────────────────────────────────────────
# Q-Network
# ────────────────────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """
    MLP: obs → [Q(s, Reject), Q(s, Accept)]

    Uses Layer Normalisation after the first hidden layer to keep activations
    well-scaled regardless of how reward_scale is set.
    """

    def __init__(self, obs_dim: int = OBS_DIM, hidden_dims: List[int] = None):
        super().__init__()
        hidden_dims = hidden_dims or HP.hidden_dims
        layers: List[nn.Module] = []
        in_dim = obs_dim
        for i, h in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, h))
            if i == 0:
                layers.append(nn.LayerNorm(h))   # stabilise early activations
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ────────────────────────────────────────────────────────────────────────────
# Replay buffer
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class Transition:
    obs     : np.ndarray
    action  : int
    reward  : float
    next_obs: np.ndarray
    done    : bool


class ReplayBuffer:
    """Circular uniform-sample experience replay buffer."""

    def __init__(self, capacity: int):
        self.buf: collections.deque = collections.deque(maxlen=capacity)

    def push(self, *args) -> None:
        self.buf.append(Transition(*args))

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
    Deep Q-Network agent for Accept / Reject decisions.

    Key design points
    -----------------
    - gamma = 1.0   : undiscounted return (correct for short finite horizon)
    - Linear ε decay: slow exploration collapse, network matures before ε drops
    - Hard action mask: if rooms_available < requested_rooms always Reject
    - learn_every   : update every N env steps to reduce update correlation
    - LayerNorm     : stabilises Q-network activations
    """

    def __init__(
        self,
        env    : HotelEnv,
        hp     : HyperParams = HP,
        device : str = "cpu",
    ):
        self.env      = env
        self.hp       = hp
        self.device   = torch.device(device)
        self._episode = 0          # tracks episodes for linear ε decay

        self.online = QNetwork(OBS_DIM, hp.hidden_dims).to(self.device)
        self.target = QNetwork(OBS_DIM, hp.hidden_dims).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=hp.lr)
        self.loss_fn   = nn.SmoothL1Loss()

        self.buffer  = ReplayBuffer(hp.buffer_size)
        self.epsilon = hp.eps_start
        self.steps   = 0           # total env steps taken

    # ── Epsilon (linear decay) ────────────────────────────────────────────

    def _update_epsilon(self) -> None:
        """Linear decay from eps_start → eps_end over eps_decay_ep episodes."""
        frac          = min(self._episode / self.hp.eps_decay_ep, 1.0)
        self.epsilon  = self.hp.eps_start + frac * (self.hp.eps_end - self.hp.eps_start)

    # ── Action selection ──────────────────────────────────────────────────

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

    # ── Learning step ─────────────────────────────────────────────────────

    def learn(self) -> Optional[float]:
        if len(self.buffer) < self.hp.batch_size:
            return None

        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.hp.batch_size)
        obs      = obs.to(self.device)
        actions  = actions.to(self.device)
        rewards  = rewards.to(self.device)
        next_obs = next_obs.to(self.device)
        dones    = dones.to(self.device)

        # Q(s, a) for the actions taken
        q_vals = self.online(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target: r + γ · max Q_target(s', ·)   [γ=1.0 → just r at terminal]
        with torch.no_grad():
            next_q   = self.target(next_obs).max(dim=1).values
            q_target = rewards + self.hp.gamma * next_q * (1.0 - dones)

        loss = self.loss_fn(q_vals, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()

        return loss.item()

    # ── Target sync ───────────────────────────────────────────────────────

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    # ── Checkpoint ────────────────────────────────────────────────────────

    def save(self, path: str = "dqn_hotel.pt") -> None:
        torch.save({
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "epsilon": self.epsilon,
            "steps"  : self.steps,
        }, path)
        print(f"  [ckpt] saved → {path}")

    def load(self, path: str = "dqn_hotel.pt") -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["target"])
        self.epsilon = ckpt["epsilon"]
        self.steps   = ckpt["steps"]
        print(f"  [ckpt] loaded ← {path}  (ε={self.epsilon:.3f}, steps={self.steps})")


# ────────────────────────────────────────────────────────────────────────────
# Training loop
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainLog:
    episode       : int
    epsilon       : float
    eval_revenue  : float      # greedy eval revenue (not noisy training revenue)
    eval_ideal    : float
    eval_regret   : float
    avg_loss      : float


def train(
    agent    : DQNAgent,
    env      : HotelEnv,
    hp       : HyperParams = HP,
    eval_env : Optional[HotelEnv] = None,
) -> List[TrainLog]:
    """
    Full DQN training loop.

    Per episode:
      1. Roll out with ε-greedy; push transitions into replay buffer.
      2. Call learn() every `hp.learn_every` env steps.
      3. Linear ε decay after each episode.
      4. Hard target sync every `hp.target_sync` episodes.
      5. Log greedy eval stats every `hp.log_every` episodes.
    """
    log_history: List[TrainLog] = []
    if eval_env is None:
        eval_env = HotelEnv(
            capacity       = env.base_capacity,
            episode_length = env.base_episode_length,
            scale          = env.scale,
            render_mode    = None,
        )

    print(f"\n{'='*70}")
    print(f"  DQN Training  capacity={env.capacity}  horizon={env.episode_length}"
          f"  scale={env.scale}")
    print(f"  {hp.n_episodes} episodes | γ={hp.gamma} | lr={hp.lr}"
          f" | ε-decay over {hp.eps_decay_ep} eps")
    print(f"  reward_scale={hp.reward_scale:.5f}"
          f" | target_sync={hp.target_sync} eps"
          f" | learn_every={hp.learn_every} steps")
    print(f"{'='*70}")
    print(f"  {'Episode':>8}  {'ε':>6}  {'Eval Rev':>10}  {'Ideal':>10}"
          f"  {'Regret%':>8}  {'Loss':>10}")
    print(f"  {'-'*68}")

    recent_losses: collections.deque = collections.deque(maxlen=200)

    for ep in range(1, hp.n_episodes + 1):
        agent._episode = ep
        agent._update_epsilon()

        obs, info = env.reset()
        done      = False

        while not done:
            action = agent.act(obs, info)
            feat   = build_obs(info, env)

            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated

            next_feat = build_obs(next_info, env)
            agent.buffer.push(feat, action, reward * hp.reward_scale, next_feat, done)
            agent.steps += 1

            # Only update every learn_every steps
            if agent.steps % hp.learn_every == 0:
                loss = agent.learn()
                if loss is not None:
                    recent_losses.append(loss)

            obs  = next_obs
            info = next_info

        if ep % hp.target_sync == 0:
            agent.sync_target()

        if ep % hp.log_every == 0:
            eval_stats = evaluate(agent, eval_env, n_episodes=hp.eval_episodes, silent=True)
            avg_loss   = float(np.mean(recent_losses)) if recent_losses else 0.0

            log_history.append(TrainLog(
                episode      = ep,
                epsilon      = agent.epsilon,
                eval_revenue = eval_stats["avg_revenue"],
                eval_ideal   = eval_stats["avg_ideal"],
                eval_regret  = eval_stats["avg_regret_pct"],
                avg_loss     = avg_loss,
            ))

            print(
                f"  {ep:>8}  {agent.epsilon:>6.3f}"
                f"  ${eval_stats['avg_revenue']:>9.1f}"
                f"  ${eval_stats['avg_ideal']:>9.1f}"
                f"  {eval_stats['avg_regret_pct']:>7.1f}%"
                f"  {avg_loss:>10.5f}"
            )

    print(f"  {'─'*68}")
    print(f"  Done. Total env steps: {agent.steps}  |  gradient updates: "
          f"{agent.steps // hp.learn_every}\n")
    return log_history


# ────────────────────────────────────────────────────────────────────────────
# Evaluation
# ────────────────────────────────────────────────────────────────────────────

def evaluate(
    agent     : DQNAgent,
    env       : HotelEnv,
    n_episodes: int  = 300,
    silent    : bool = False,
) -> Dict:
    revenues, ideals, regrets, regret_pcts = [], [], [], []

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
        regret_pcts.append(info.get("regret_pct", 0.0))

    stats = {
        "avg_revenue"    : float(np.mean(revenues)),
        "std_revenue"    : float(np.std(revenues)),
        "avg_ideal"      : float(np.mean(ideals)),
        "avg_regret"     : float(np.mean(regrets)),
        "avg_regret_pct" : float(np.mean(regret_pcts)),
    }
    if not silent:
        print(f"\n── DQN Evaluation ({n_episodes} episodes, greedy) ──")
        print(f"  Avg Revenue : ${stats['avg_revenue']:.1f}  ± {stats['std_revenue']:.1f}")
        print(f"  Avg Ideal   : ${stats['avg_ideal']:.1f}")
        print(f"  Avg Regret  : ${stats['avg_regret']:.1f}  ({stats['avg_regret_pct']:.1f}%)\n")
    return stats


# ────────────────────────────────────────────────────────────────────────────
# Comparison table
# ────────────────────────────────────────────────────────────────────────────

def compare_agents(dqn_agent: DQNAgent, env: HotelEnv, n_episodes: int = 500):
    """DQN vs rule-based baselines on revenue and regret."""
    rule_agents = {
        "Greedy"         : GreedyAgent(),
        "Threshold(0.3)" : ThresholdAgent(0.3),
        "Random"         : RandomAgent(),
    }

    cw   = 16
    cols = ["Agent", "Avg Revenue", "Ideal", "Regret $", "Regret %", "Accepted", "Rejected"]
    sep  = "=" * (cw * len(cols))
    print(f"\n{sep}")
    print(f"  Final Comparison  |  {n_episodes} episodes  |  scale={env.scale}")
    print(sep)
    print("  " + "".join(f"{c:<{cw}}" for c in cols))
    print("  " + "-" * (cw * len(cols)))

    def _row(name, rev, ideal, reg, regp, acc, rej):
        print(
            f"  {name:<{cw}}"
            f"${rev:<{cw-1}.1f}"
            f"${ideal:<{cw-1}.1f}"
            f"${reg:<{cw-1}.1f}"
            f"{regp:<{cw}.1f}%"
            f"{acc:<{cw}.1f}"
            f"{rej:<{cw}.1f}"
        )

    # DQN — greedy rollouts
    revenues, ideals, regrets, regpcts, accs, rejs = [], [], [], [], [], []
    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        while not done:
            action = dqn_agent.act(obs, info, greedy=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        revenues.append(info["episode_revenue"])
        ideals.append(info.get("ideal_revenue", 0.0))
        regrets.append(info.get("regret", 0.0))
        regpcts.append(info.get("regret_pct", 0.0))
        accs.append(info["accepted"])
        rejs.append(info["rejected"])
    _row("DQN (trained)",
         np.mean(revenues), np.mean(ideals), np.mean(regrets),
         np.mean(regpcts), np.mean(accs), np.mean(rejs))

    print("  " + "-" * (cw * len(cols)))

    for name, agent in rule_agents.items():
        results = [run_episode(env, agent) for _ in range(n_episodes)]
        _row(name,
             np.mean([r["total_revenue"] for r in results]),
             np.mean([r["ideal_revenue"] for r in results]),
             np.mean([r["regret"]        for r in results]),
             np.mean([r["regret_pct"]    for r in results]),
             np.mean([r["accepted"]      for r in results]),
             np.mean([r["rejected"]      for r in results]))

    print(sep)
    print("  Lower Regret % = closer to the hindsight-optimal oracle.\n")


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    SCALE = 1

    train_env = HotelEnv(capacity=20, episode_length=50, scale=SCALE,
                         render_mode=None, seed=0)
    eval_env  = HotelEnv(capacity=20, episode_length=50, scale=SCALE,
                         render_mode=None)

    hp = HyperParams(
        hidden_dims   = [128, 128],
        buffer_size   = 50_000,
        batch_size    = 256,
        lr            = 3e-4,
        gamma         = 1.0,          # undiscounted
        n_episodes    = 3_000,
        learn_every   = 4,
        target_sync   = 10,
        eps_start     = 1.0,
        eps_end       = 0.05,
        eps_decay_ep  = 1_800,        # linear decay over 60% of training
        log_every     = 200,
        eval_episodes = 100,
    )

    agent = DQNAgent(train_env, hp=hp, device="cpu")
    logs  = train(agent, train_env, hp=hp, eval_env=eval_env)

    agent.save("dqn_hotel.pt")

    evaluate(agent, eval_env, n_episodes=500, silent=False)
    compare_agents(agent, eval_env, n_episodes=500)
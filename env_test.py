"""
Hotel Revenue Management Environment
======================================
A single-product (rooms) Gymnasium environment for Revenue Management.

Observation space:
    - rooms_occupied  : int  [0, capacity]
    - time_step       : int  [0, episode_length]
    - customer_type   : int  [0, num_customer_types - 1]  <- current arriving customer
    - requested_rooms : int  [1, max_request]

Action space:
    - 0 : Reject the customer
    - 1 : Accept the customer

Reward:
    Revenue paid by the accepted customer (0 if rejected or no capacity).

Customer types (configurable):
    Budget   – requests 1 room,  pays $80
    Standard – requests 1–2 rooms, pays $120
    Premium  – requests 1–3 rooms, pays $200
    Group    – requests 3–6 rooms, pays $90/room

Run:
    python hotel_env.py
"""

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any


# ---------------------------------------------------------------------------
# Customer type definition
# ---------------------------------------------------------------------------

@dataclass
class CustomerType:
    name: str
    min_rooms: int
    max_rooms: int
    reward_per_room: float          # $ per room requested
    arrival_prob: float             # relative weight (normalised internally)


DEFAULT_CUSTOMER_TYPES: List[CustomerType] = [
    CustomerType("Budget",   min_rooms=1, max_rooms=1, reward_per_room=80.0,  arrival_prob=0.40),
    CustomerType("Standard", min_rooms=1, max_rooms=2, reward_per_room=120.0, arrival_prob=0.35),
    CustomerType("Premium",  min_rooms=1, max_rooms=3, reward_per_room=200.0, arrival_prob=0.15),
    CustomerType("Group",    min_rooms=3, max_rooms=6, reward_per_room=90.0,  arrival_prob=0.10),
]


# ---------------------------------------------------------------------------
# Gymnasium Environment
# ---------------------------------------------------------------------------

class HotelEnv(gym.Env):
    """
    Hotel single-product (rooms) Revenue Management environment.

    At each time-step a customer arrives with a type and a room request.
    The decision-maker (agent) chooses to Accept (1) or Reject (0).

    Parameters
    ----------
    capacity        : total rooms available
    episode_length  : number of time-steps per episode (e.g. booking horizon)
    customer_types  : list of CustomerType objects
    seed            : random seed
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        capacity: int = 20,
        episode_length: int = 50,
        customer_types: Optional[List[CustomerType]] = None,
        render_mode: Optional[str] = "human",
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.capacity = capacity
        self.episode_length = episode_length
        self.customer_types = customer_types or DEFAULT_CUSTOMER_TYPES
        self.render_mode = render_mode
        self.num_types = len(self.customer_types)

        # Normalise arrival probabilities
        total = sum(c.arrival_prob for c in self.customer_types)
        self._arrival_probs = np.array(
            [c.arrival_prob / total for c in self.customer_types], dtype=np.float32
        )

        self.max_request = max(c.max_rooms for c in self.customer_types)

        # ---------------------------------------------------------------
        # Gymnasium spaces
        # ---------------------------------------------------------------
        # Observation: [rooms_occupied, time_step, customer_type_idx, requested_rooms]
        self.observation_space = spaces.Box(
            low  = np.array([0, 0, 0, 1],                                          dtype=np.int32),
            high = np.array([capacity, episode_length, self.num_types - 1, self.max_request], dtype=np.int32),
            dtype=np.int32,
        )

        # Action: 0 = Reject, 1 = Accept
        self.action_space = spaces.Discrete(2)

        # Internal state
        self._rooms_occupied: int = 0
        self._time_step: int = 0
        self._current_customer_type: int = 0
        self._current_request: int = 0

        # Episode stats
        self._episode_revenue: float = 0.0
        self._accepted: int = 0
        self._rejected: int = 0

        # RNG
        self.np_random, _ = gym.utils.seeding.np_random(seed)

    # ------------------------------------------------------------------
    # Core Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        self._rooms_occupied = 0
        self._time_step = 0
        self._episode_revenue = 0.0
        self._accepted = 0
        self._rejected = 0

        self._sample_customer()

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        assert self.action_space.contains(action), f"Invalid action: {action}"

        ctype = self.customer_types[self._current_customer_type]
        reward = 0.0

        if action == 1:                              # Accept
            rooms_needed = self._current_request
            if self._rooms_occupied + rooms_needed <= self.capacity:
                reward = ctype.reward_per_room * rooms_needed
                self._rooms_occupied += rooms_needed
                self._episode_revenue += reward
                self._accepted += 1
            else:
                # Tried to accept but no capacity – forced reject, no reward
                self._rejected += 1
        else:                                        # Reject
            self._rejected += 1

        self._time_step += 1
        terminated = self._time_step >= self.episode_length
        truncated  = False

        if not terminated:
            self._sample_customer()

        obs  = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human" and not terminated:
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        ctype = self.customer_types[self._current_customer_type]
        bar_full  = int(self._rooms_occupied / self.capacity * 20)
        bar_empty = 20 - bar_full
        bar = "[" + "█" * bar_full + "░" * bar_empty + "]"
        print(
            f"t={self._time_step:>3} | Rooms {bar} {self._rooms_occupied:>2}/{self.capacity} "
            f"| Next: {ctype.name:<8} req={self._current_request} "
            f"(${ctype.reward_per_room:.0f}/room) "
            f"| Revenue=${self._episode_revenue:>8.1f}"
        )

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sample_customer(self):
        """Draw a new arriving customer from the type distribution."""
        self._current_customer_type = int(
            self.np_random.choice(self.num_types, p=self._arrival_probs)
        )
        ctype = self.customer_types[self._current_customer_type]
        self._current_request = int(
            self.np_random.integers(ctype.min_rooms, ctype.max_rooms + 1)
        )

    def _get_obs(self) -> np.ndarray:
        return np.array(
            [
                self._rooms_occupied,
                self._time_step,
                self._current_customer_type,
                self._current_request,
            ],
            dtype=np.int32,
        )

    def _get_info(self) -> Dict[str, Any]:
        ctype = self.customer_types[self._current_customer_type]
        return {
            "rooms_occupied"    : self._rooms_occupied,
            "rooms_available"   : self.capacity - self._rooms_occupied,
            "time_step"         : self._time_step,
            "episode_revenue"   : self._episode_revenue,
            "accepted"          : self._accepted,
            "rejected"          : self._rejected,
            "current_customer"  : ctype.name,
            "requested_rooms"   : self._current_request,
            "reward_per_room"   : ctype.reward_per_room,
        }

    # ------------------------------------------------------------------
    # Torch helper: obs → tensor
    # ------------------------------------------------------------------

    def obs_to_tensor(self, obs: np.ndarray, device: str = "cpu") -> torch.Tensor:
        """Convert a numpy observation to a normalised float32 tensor."""
        high = self.observation_space.high.astype(np.float32)
        return torch.tensor(obs / high, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Simple rule-based agents (baselines)
# ---------------------------------------------------------------------------

class GreedyAgent:
    """Always accept if capacity allows."""
    def __call__(self, obs: np.ndarray, info: Dict) -> int:
        return 1 if info["rooms_available"] >= info["requested_rooms"] else 0


class ThresholdAgent:
    """
    Accept only when remaining capacity > threshold fraction.
    Useful as a conservative baseline.
    """
    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold

    def __call__(self, obs: np.ndarray, info: Dict) -> int:
        env_capacity = obs[0] + info["rooms_available"]   # total
        frac_available = info["rooms_available"] / env_capacity
        if frac_available > self.threshold and info["rooms_available"] >= info["requested_rooms"]:
            return 1
        return 0


class RandomAgent:
    """Accepts randomly with 50 % probability."""
    def __call__(self, obs: np.ndarray, info: Dict) -> int:
        return int(np.random.rand() > 0.5)


# ---------------------------------------------------------------------------
# Run demo
# ---------------------------------------------------------------------------

def run_episode(env: HotelEnv, agent, verbose: bool = False) -> Dict[str, float]:
    obs, info = env.reset()
    total_reward = 0.0
    done = False

    while not done:
        action = agent(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    return {
        "total_revenue": info["episode_revenue"],
        "accepted"     : info["accepted"],
        "rejected"     : info["rejected"],
    }


def benchmark(n_episodes: int = 200):
    env = HotelEnv(capacity=20, episode_length=50, render_mode=None)

    agents = {
        "Greedy"         : GreedyAgent(),
        "Threshold(0.3)" : ThresholdAgent(0.3),
        "Random"         : RandomAgent(),
    }

    print(f"\n{'='*60}")
    print(f"  Hotel RM Environment — {n_episodes}-episode benchmark")
    print(f"  Capacity: {env.capacity} rooms | Horizon: {env.episode_length} steps")
    print(f"{'='*60}")
    print(f"  {'Agent':<20} {'Avg Revenue':>12} {'Avg Accept':>12} {'Avg Reject':>12}")
    print(f"  {'-'*56}")

    for name, agent in agents.items():
        results = [run_episode(env, agent) for _ in range(n_episodes)]
        avg_rev = np.mean([r["total_revenue"] for r in results])
        avg_acc = np.mean([r["accepted"]      for r in results])
        avg_rej = np.mean([r["rejected"]      for r in results])
        print(f"  {name:<20} ${avg_rev:>10.1f} {avg_acc:>12.1f} {avg_rej:>12.1f}")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    print("\n--- Single episode (Greedy agent, render_mode='human') ---\n")
    env = HotelEnv(capacity=10, episode_length=20, render_mode="human", seed=42)
    agent = GreedyAgent()
    stats = run_episode(env, agent)
    print(f"\nEpisode done → Revenue: ${stats['total_revenue']:.1f} | "
          f"Accepted: {stats['accepted']} | Rejected: {stats['rejected']}")

    benchmark(n_episodes=500)

    print("--- Torch tensor demo ---")
    env2 = HotelEnv(seed=0)
    obs, _ = env2.reset()
    tensor = env2.obs_to_tensor(obs)
    print(f"Raw obs  : {obs}")
    print(f"Tensor   : {tensor}  (normalised float32, ready for a neural net)\n")
"""
Continuous-Time Markov Chain Environment for Xie et al. (2024) Network 1.

This implements the reusable resource allocation problem as a Gymnasium environment.
Unlike hotel_env.py (single resource, finite horizon), this models:
  - 2 resources (a, b) with capacities q_a, q_b
  - 3 customer types with Poisson arrivals and exponential service
  - Customers DEPART after random service times (resources are reusable)
  - Infinite horizon (episodic approximation with long horizon)

Network 1 Parameters (Xie, Gurvich, Kucukyavuz 2024):
  Type 1: lambda=3, mu=0.5, r=5, uses resource a only
  Type 2: lambda=2, mu=1.0, r=1, uses resources a AND b
  Type 3: lambda=5, mu=1/3, r=2, uses resource b only
  A = [[1,1,0], [0,1,1]]  (adjacency matrix)

Usage:
  env = CTMCEnv(N=1)  # N is the scaling parameter
  obs, info = env.reset()
  for _ in range(10000):
      action = policy(obs, info)
      obs, reward, done, truncated, info = env.step(action)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class CustomerType:
    name: str
    arrival_rate: float    # lambda_i
    service_rate: float    # mu_i (1/mean_service_time)
    reward: float          # r_i (collected on acceptance)
    resources: List[int]   # indices of resources used


# Network 1 from Xie et al. (2024)
NETWORK1_TYPES = [
    CustomerType("VIP",    arrival_rate=3.0, service_rate=0.5,   reward=5.0, resources=[0]),
    CustomerType("Suite",  arrival_rate=2.0, service_rate=1.0,   reward=1.0, resources=[0, 1]),
    CustomerType("Budget", arrival_rate=5.0, service_rate=1/3,   reward=2.0, resources=[1]),
]

NETWORK1_CAPACITIES = [7, 6]  # q_a=7, q_b=6
NETWORK1_A = np.array([[1, 1, 0],
                        [0, 1, 1]])  # adjacency matrix


class CTMCEnv(gym.Env):
    """
    Gymnasium environment for the reusable resource allocation CTMC.

    At each step, one event occurs (arrival or departure), determined
    by the rates. On arrivals, the agent chooses accept/reject.
    Departures happen automatically.

    State: (x_1, x_2, x_3) = number of each type currently in service
    Action: 0 = reject, 1 = accept (only meaningful on arrivals)
    Reward: r_i on acceptance of type i, 0 otherwise
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, N=1, customer_types=None, capacities=None, A=None,
                 max_events=10000, render_mode=None):
        super().__init__()

        self.N = N
        self.customer_types = customer_types or NETWORK1_TYPES
        self.base_capacities = np.array(capacities or NETWORK1_CAPACITIES)
        self.capacities = self.base_capacities * N  # scaled capacities
        self.A = np.array(A) if A is not None else NETWORK1_A.copy()
        self.n_types = len(self.customer_types)
        self.n_resources = len(self.capacities)
        self.max_events = max_events
        self.render_mode = render_mode

        # Scale arrival rates by N
        self.arrival_rates = np.array([ct.arrival_rate * N for ct in self.customer_types])
        self.service_rates = np.array([ct.service_rate for ct in self.customer_types])
        self.rewards = np.array([ct.reward for ct in self.customer_types])

        # Compute LP bound
        self._compute_lp_bound()

        # Action space: 0=reject, 1=accept
        self.action_space = spaces.Discrete(2)

        # Observation space: customer counts + current event info
        max_customers = int(max(self.capacities)) + 1
        self.observation_space = spaces.Box(
            low=0,
            high=max_customers,
            shape=(self.n_types + 2,),  # x_1..x_n, event_type, is_arrival
            dtype=np.float32
        )

        self.state = None
        self.time = 0.0
        self.event_count = 0
        self.total_reward = 0.0
        self.current_event = None  # (type_idx, is_arrival)

    def _compute_lp_bound(self):
        """Compute the LP relaxation upper bound."""
        try:
            from scipy.optimize import linprog
            # max r_mu' y  s.t.  Ay <= q, 0 <= y <= lambda/mu
            r_mu = self.rewards * self.service_rates  # reward rate
            offered_load = self.arrival_rates / self.service_rates

            # linprog minimizes, so negate
            res = linprog(
                c=-r_mu,
                A_ub=self.A,
                b_ub=self.capacities.astype(float),
                bounds=[(0, ol) for ol in offered_load],
                method='highs'
            )
            self.lp_bound = -res.fun if res.success else None
            self.lp_solution = res.x if res.success else None
        except ImportError:
            self.lp_bound = None
            self.lp_solution = None

    def _resource_usage(self, state):
        """Compute resource usage: A @ x."""
        return self.A @ state

    def _can_accept(self, state, type_idx):
        """Check if type_idx can be accepted in the current state."""
        test_state = state.copy()
        test_state[type_idx] += 1
        return np.all(self.A @ test_state <= self.capacities)

    def _sample_event(self, state):
        """Sample the next event (arrival or departure)."""
        # Arrival rates (always positive)
        arr_rates = self.arrival_rates.copy()

        # Departure rates (proportional to current customers)
        dep_rates = self.service_rates * state

        all_rates = np.concatenate([arr_rates, dep_rates])
        total_rate = all_rates.sum()

        # Time until next event
        dt = np.random.exponential(1.0 / total_rate)

        # Which event
        probs = all_rates / total_rate
        event_idx = np.random.choice(len(all_rates), p=probs)

        if event_idx < self.n_types:
            # Arrival
            return dt, event_idx, True
        else:
            # Departure
            return dt, event_idx - self.n_types, False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.state = np.zeros(self.n_types, dtype=int)
        self.time = 0.0
        self.event_count = 0
        self.total_reward = 0.0
        self.n_accepted = np.zeros(self.n_types, dtype=int)
        self.n_rejected = np.zeros(self.n_types, dtype=int)
        self.n_departed = np.zeros(self.n_types, dtype=int)
        self.n_blocked = np.zeros(self.n_types, dtype=int)  # rejected because full

        # Sample first event
        self._advance_to_next_arrival()

        return self._get_obs(), self._get_info()

    def _advance_to_next_arrival(self):
        """Keep processing departures until an arrival occurs."""
        while self.event_count < self.max_events:
            dt, type_idx, is_arrival = self._sample_event(self.state)
            self.time += dt
            self.event_count += 1

            if is_arrival:
                self.current_event = (type_idx, True)
                return
            else:
                # Process departure automatically
                if self.state[type_idx] > 0:
                    self.state[type_idx] -= 1
                    self.n_departed[type_idx] += 1

        # Max events reached
        self.current_event = None

    def step(self, action):
        """
        Process the current arrival event.
        action: 0 = reject, 1 = accept
        """
        if self.current_event is None:
            return self._get_obs(), 0.0, True, True, self._get_info()

        type_idx, is_arrival = self.current_event
        assert is_arrival, "step() should only be called on arrivals"

        reward = 0.0
        can_accept = self._can_accept(self.state, type_idx)

        if action == 1 and can_accept:
            # Accept
            self.state[type_idx] += 1
            reward = self.rewards[type_idx]
            self.total_reward += reward
            self.n_accepted[type_idx] += 1
        elif action == 1 and not can_accept:
            # Tried to accept but no room
            self.n_blocked[type_idx] += 1
        else:
            # Reject
            self.n_rejected[type_idx] += 1

        # Advance to next arrival
        self._advance_to_next_arrival()

        done = self.current_event is None or self.event_count >= self.max_events
        truncated = self.event_count >= self.max_events

        return self._get_obs(), reward, done, truncated, self._get_info()

    def _get_obs(self):
        """Build observation vector."""
        if self.current_event is None:
            event_type = 0
            is_arr = 0
        else:
            event_type = self.current_event[0]
            is_arr = 1 if self.current_event[1] else 0

        obs = np.zeros(self.n_types + 2, dtype=np.float32)
        obs[:self.n_types] = self.state.astype(np.float32)
        obs[self.n_types] = float(event_type)
        obs[self.n_types + 1] = float(is_arr)
        return obs

    def _get_info(self):
        """Return info dict."""
        usage = self._resource_usage(self.state)
        can_accept = self._can_accept(self.state, self.current_event[0]) if self.current_event else False

        info = {
            "state": self.state.copy(),
            "time": self.time,
            "event_count": self.event_count,
            "total_reward": self.total_reward,
            "reward_rate": self.total_reward / max(self.time, 1e-10),
            "resource_usage": usage,
            "resource_available": self.capacities - usage,
            "can_accept": can_accept,
            "n_accepted": self.n_accepted.copy(),
            "n_rejected": self.n_rejected.copy(),
            "n_departed": self.n_departed.copy(),
            "lp_bound": self.lp_bound,
            "N": self.N,
        }

        if self.current_event:
            info["current_type"] = self.current_event[0]
            info["current_type_name"] = self.customer_types[self.current_event[0]].name

        return info

    def render(self):
        if self.render_mode != "human":
            return
        usage = self._resource_usage(self.state)
        print(f"State: {self.state}, Resource usage: {usage}/{self.capacities}, "
              f"Time: {self.time:.2f}, Rate: {self.total_reward/max(self.time,1e-10):.2f}")


# ============================================================
# Built-in policies for comparison
# ============================================================

class GreedyPolicy:
    """Accept all feasible arrivals."""
    name = "Greedy (Accept All)"

    def __call__(self, obs, info):
        return 1  # always try to accept


class RejectAllPolicy:
    """Reject everything."""
    name = "Reject All"

    def __call__(self, obs, info):
        return 0


class VIPOnlyPolicy:
    """Only accept Type 1 (VIP) customers."""
    name = "VIP Only"

    def __call__(self, obs, info):
        return 1 if info.get("current_type", -1) == 0 else 0


def run_policy(env, policy, n_events=100000, seed=42):
    """Run a policy and return average reward rate."""
    obs, info = env.reset(seed=seed)
    total_reward = 0.0

    while not (info.get("event_count", 0) >= n_events):
        action = policy(obs, info)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        if done:
            break

    rate = total_reward / max(info["time"], 1e-10)
    return rate, info


if __name__ == "__main__":
    print("=" * 60)
    print("  CTMC Environment for Xie et al. Network 1")
    print("=" * 60)

    for N in [1, 2]:
        print(f"\n--- N = {N} ---")
        env = CTMCEnv(N=N, max_events=200000)

        policies = [GreedyPolicy(), VIPOnlyPolicy()]

        for policy in policies:
            rate, info = run_policy(env, policy, n_events=200000)
            lp = info["lp_bound"]
            regret = lp - rate if lp else "N/A"
            print(f"  {policy.name:20s}: Rate={rate:.3f}, LP={lp:.3f}, Regret={regret:.3f}")

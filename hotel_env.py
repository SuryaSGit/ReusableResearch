"""
Hotel Revenue Management Environment
======================================
A single-product (rooms) Gymnasium environment for Revenue Management.

New features vs v1:
  - Backtracking  : env records full episode history; call env.backtrack(k)
                    to rewind k steps and replay from that point.
  - Ideal reward  : after every episode the hindsight-optimal revenue is
                    computed (oracle that replays the same arrival sequence
                    and always accepts in descending $/room order while
                    capacity allows).  Regret = ideal - actual is reported
                    in info and returned by run_episode().
  - N-scaling     : pass scale=N to __init__ (or use make_scaled_env(N)).
                    Capacity, all customer room requests, and the episode
                    length all scale by N so the environment is a faithful
                    N-times larger hotel.  Rewards also scale (more rooms
                    sold), so for fair cross-scale comparison use
                    info["revenue_per_room_sold"] or the normalised regret
                    info["regret_pct"].

Observation space  (all int32):
    [rooms_occupied, time_step, customer_type_idx, requested_rooms]

Action space:
    0 = Reject   1 = Accept

Run:
    python hotel_env.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Customer type
# ---------------------------------------------------------------------------

@dataclass
class CustomerType:
    name: str
    min_rooms: int          # base (un-scaled) min rooms requested
    max_rooms: int          # base (un-scaled) max rooms requested
    reward_per_room: float  # $ per room
    arrival_prob: float     # relative arrival weight


DEFAULT_CUSTOMER_TYPES: List[CustomerType] = [
    CustomerType("Budget",   min_rooms=1, max_rooms=1, reward_per_room=80.0,  arrival_prob=0.40),
    CustomerType("Standard", min_rooms=1, max_rooms=2, reward_per_room=120.0, arrival_prob=0.35),
    CustomerType("Premium",  min_rooms=1, max_rooms=3, reward_per_room=200.0, arrival_prob=0.15),
    CustomerType("Group",    min_rooms=3, max_rooms=6, reward_per_room=90.0,  arrival_prob=0.10),
]

# Tighter environment: capacity pressure forces real accept/reject tradeoffs.
# Expected rooms demanded per step ≈ 2.5; with capacity=20 and horizon=50,
# ~125 rooms are demanded vs 20 available — agent MUST reject low-value guests.
TIGHT_CUSTOMER_TYPES: List[CustomerType] = [
    CustomerType("Budget",   min_rooms=1, max_rooms=2, reward_per_room=80.0,  arrival_prob=0.40),
    CustomerType("Standard", min_rooms=1, max_rooms=3, reward_per_room=120.0, arrival_prob=0.35),
    CustomerType("Premium",  min_rooms=2, max_rooms=4, reward_per_room=200.0, arrival_prob=0.15),
    CustomerType("Group",    min_rooms=4, max_rooms=8, reward_per_room=90.0,  arrival_prob=0.10),
]


# ---------------------------------------------------------------------------
# Step record – used by the backtracking buffer
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    """A snapshot of state *before* the step, plus what happened during it."""
    # State before action
    rooms_occupied_before : int
    episode_revenue_before: float
    accepted_before       : int
    rejected_before       : int
    # The arriving customer
    customer_type         : int    # index into customer_types
    request               : int    # rooms requested (scaled)
    time_step             : int    # t at which this customer arrived
    # What the agent did
    action                : int
    reward                : float


# ---------------------------------------------------------------------------
# Hindsight oracle  (ideal revenue upper bound)
# ---------------------------------------------------------------------------

def _ideal_revenue(
    history       : List[StepRecord],
    capacity      : int,
    customer_types: List[CustomerType],
) -> float:
    """
    Compute the true sequential-optimal revenue on the realised arrival sequence
    using backward induction (DP).

    Why not the knapsack oracle?
    The knapsack oracle re-orders arrivals by $/room and fills greedily — but
    that is *not* achievable online.  A real agent must decide at t=1 without
    knowing who arrives at t=2..T.  The knapsack bound is therefore unachievably
    optimistic and produces inflated regret numbers.

    The DP oracle IS achievable: it knows the full sequence in hindsight and
    makes the optimal accept/reject at each time-step in order.  This is the
    tightest valid upper bound for any online policy on this realisation.

    DP formulation
    --------------
    V[t][r] = max revenue collectible from step t onward given r rooms remaining.
    Base:     V[T][r] = 0 for all r.
    Recurse:  V[t][r] = max(
                  reject:  V[t+1][r],
                  accept:  reward_t + V[t+1][r - req_t]   if req_t <= r
              )
    Answer:   V[0][capacity]
    """
    # De-duplicate: if agent backtracked, keep last action per time-step
    seen: Dict[int, StepRecord] = {}
    for rec in history:
        seen[rec.time_step] = rec

    T        = len(seen)
    arrivals = [seen[t] for t in sorted(seen)]   # ordered t=0,1,...,T-1

    # V[r] = best revenue from current step onward with r rooms left
    V = np.zeros(capacity + 1, dtype=np.float64)

    for rec in reversed(arrivals):
        ctype   = customer_types[rec.customer_type]
        reward  = ctype.reward_per_room * rec.request
        req     = rec.request
        V_new   = V.copy()                        # reject branch (always available)
        for r in range(req, capacity + 1):        # accept branch
            V_new[r] = max(V_new[r], reward + V[r - req])
        V = V_new

    return float(V[capacity])


# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------

class HotelEnv(gym.Env):
    """
    Hotel single-product (rooms) Revenue Management environment.

    Parameters
    ----------
    capacity        : base number of rooms (multiplied by `scale`)
    episode_length  : base booking horizon in steps (multiplied by `scale`)
    customer_types  : list of CustomerType; defaults to DEFAULT_CUSTOMER_TYPES
    scale           : integer N – multiplies capacity, room requests, and
                      episode_length by N.  scale=1 is the un-scaled base env.
    render_mode     : "human" | None
    seed            : random seed
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        capacity      : int = 20,
        episode_length: int = 50,
        customer_types: Optional[List[CustomerType]] = None,
        scale         : int = 1,
        render_mode   : Optional[str] = "human",
        seed          : Optional[int] = None,
    ):
        super().__init__()

        if scale < 1:
            raise ValueError(f"scale must be >= 1, got {scale}")

        self.base_capacity       = capacity
        self.base_episode_length = episode_length
        self.scale               = scale

        # Scaled dimensions
        self.capacity       = capacity * scale
        self.episode_length = episode_length * scale

        self.customer_types = customer_types or DEFAULT_CUSTOMER_TYPES
        self.render_mode    = render_mode
        self.num_types      = len(self.customer_types)

        # Normalise arrival probabilities
        total = sum(c.arrival_prob for c in self.customer_types)
        self._arrival_probs = np.array(
            [c.arrival_prob / total for c in self.customer_types], dtype=np.float32
        )

        # Largest possible room request (scaled)
        self.max_request = max(c.max_rooms for c in self.customer_types) * scale

        # ── Gymnasium spaces ──────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low  = np.array([0, 0, 0, 1], dtype=np.int32),
            high = np.array(
                [self.capacity, self.episode_length, self.num_types - 1, self.max_request],
                dtype=np.int32,
            ),
            dtype=np.int32,
        )
        self.action_space = spaces.Discrete(2)   # 0=Reject  1=Accept

        # Mutable state – initialised properly in reset()
        self._rooms_occupied : int   = 0
        self._time_step      : int   = 0
        self._episode_revenue: float = 0.0
        self._accepted       : int   = 0
        self._rejected       : int   = 0
        self._customer_type  : int   = 0
        self._request        : int   = 1

        # Full episode history for backtracking & ideal-reward computation
        self._history: List[StepRecord] = []

        self.np_random, _ = gym.utils.seeding.np_random(seed)

    # ── Gymnasium API ─────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed   : Optional[int]  = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        self._rooms_occupied  = 0
        self._time_step       = 0
        self._episode_revenue = 0.0
        self._accepted        = 0
        self._rejected        = 0
        self._history         = []

        self._sample_customer()
        return self._get_obs(), self._get_info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        assert self.action_space.contains(action), f"Invalid action: {action}"

        ctype  = self.customer_types[self._customer_type]
        reward = 0.0

        # Snapshot state *before* any mutation (needed for clean backtrack)
        rec = StepRecord(
            rooms_occupied_before  = self._rooms_occupied,
            episode_revenue_before = self._episode_revenue,
            accepted_before        = self._accepted,
            rejected_before        = self._rejected,
            customer_type          = self._customer_type,
            request                = self._request,
            time_step              = self._time_step,
            action                 = action,
            reward                 = 0.0,    # filled below
        )

        if action == 1:                          # Accept
            if self._rooms_occupied + self._request <= self.capacity:
                reward                = ctype.reward_per_room * self._request
                self._rooms_occupied  += self._request
                self._episode_revenue += reward
                self._accepted        += 1
            else:
                self._rejected += 1              # over capacity → forced reject
        else:
            self._rejected += 1                  # voluntary reject

        rec.reward = reward
        self._history.append(rec)

        self._time_step += 1
        terminated = self._time_step >= self.episode_length
        truncated  = False

        if not terminated:
            self._sample_customer()

        obs  = self._get_obs()
        info = self._get_info()

        if terminated:
            ideal  = _ideal_revenue(self._history, self.capacity, self.customer_types)
            actual = self._episode_revenue
            info["ideal_revenue"] = ideal
            info["regret"]        = ideal - actual
            info["regret_pct"]    = (
                100.0 * (ideal - actual) / ideal if ideal > 0 else 0.0
            )

        if self.render_mode == "human" and not terminated:
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        ctype     = self.customer_types[self._customer_type]
        bar_full  = int(self._rooms_occupied / self.capacity * 20)
        bar_empty = 20 - bar_full
        bar = "[" + "█" * bar_full + "░" * bar_empty + "]"
        print(
            f"t={self._time_step:>4} | Rooms {bar} {self._rooms_occupied:>3}/{self.capacity}"
            f" | Next: {ctype.name:<8} req={self._request:>2}"
            f" (${ctype.reward_per_room:.0f}/room)"
            f" | Revenue=${self._episode_revenue:>9.1f}"
        )

    def close(self):
        pass

    # ── Backtracking ──────────────────────────────────────────────────────

    def backtrack(self, k: int = 1) -> Tuple[np.ndarray, Dict]:
        """
        Rewind the environment by `k` completed steps.

        The last k records are removed from history and the environment
        state is restored to exactly what it was before those k steps were
        taken.  The next call to step() will proceed from that restored
        state with a freshly sampled arriving customer.

        Parameters
        ----------
        k : number of steps to undo (must be >= 1 and <= steps taken so far)

        Returns
        -------
        obs, info  at the rewound state (same format as reset() / step())

        Raises
        ------
        ValueError  if k < 1 or k > len(history)
        """
        if k < 1:
            raise ValueError("k must be >= 1")
        if k > len(self._history):
            raise ValueError(
                f"Cannot backtrack {k} steps – only {len(self._history)} steps in history."
            )

        # Drop last k records
        self._history = self._history[:-k]

        if self._history:
            # Restore the pre-action snapshot of the record now at the tail
            prev = self._history[-1]
            # The tail record stores the state *before* that step's action,
            # so restoring from it gives us the state right after the step
            # before the tail – i.e. the correct "between-step" state.
            # We actually want the state *after* prev was applied, so we
            # use the current values stored in the record that was removed.
            # Simplest: restore from the "before" fields of the step that
            # was just REMOVED (but we already dropped it).  Instead, re-derive
            # from the last remaining record's post-action values, which are
            # encoded in its "before" fields of the next record – unavailable.
            # ──────────────────────────────────────────────────────────────
            # Cleaner: the history stores "before" state.  After k pops,
            # the state to restore is the "before" of the first popped record,
            # which is the "before" of history[-1+1] (already gone).
            # So we re-derive: walk from the start and replay accepted rewards.
            # This is O(T) but T is small (episode_length).
            self._rooms_occupied, self._episode_revenue, self._accepted, self._rejected = \
                self._replay_stats()
            self._time_step = len(self._history)
        else:
            # Rewound to episode start
            self._rooms_occupied  = 0
            self._episode_revenue = 0.0
            self._accepted        = 0
            self._rejected        = 0
            self._time_step       = 0

        self._sample_customer()
        return self._get_obs(), self._get_info()

    def _replay_stats(self) -> Tuple[int, float, int, int]:
        """Re-derive current state by replaying the kept history records."""
        rooms    = 0
        revenue  = 0.0
        accepted = 0
        rejected = 0
        for rec in self._history:
            revenue  += rec.reward
            if rec.reward > 0:
                rooms    += rec.request
                accepted += 1
            else:
                rejected += 1
        return rooms, revenue, accepted, rejected

    # ── N-scaling factory ─────────────────────────────────────────────────

    def scaled(self, n: int, **kwargs) -> "HotelEnv":
        """Return a new independent HotelEnv scaled by n."""
        return HotelEnv(
            capacity       = self.base_capacity,
            episode_length = self.base_episode_length,
            customer_types = self.customer_types,
            scale          = n,
            render_mode    = self.render_mode,
            **kwargs,
        )

    # ── Torch helper ──────────────────────────────────────────────────────

    def obs_to_tensor(self, obs: np.ndarray, device: str = "cpu") -> torch.Tensor:
        """Normalise obs to [0, 1] float32 tensor ready for a neural network."""
        high = self.observation_space.high.astype(np.float32)
        return torch.tensor(obs / high, dtype=torch.float32, device=device)

    # ── Private ───────────────────────────────────────────────────────────

    def _sample_customer(self):
        idx   = int(self.np_random.choice(self.num_types, p=self._arrival_probs))
        ctype = self.customer_types[idx]
        lo    = ctype.min_rooms * self.scale
        hi    = ctype.max_rooms * self.scale
        self._customer_type = idx
        self._request       = int(self.np_random.integers(lo, hi + 1))

    def _get_obs(self) -> np.ndarray:
        return np.array(
            [self._rooms_occupied, self._time_step, self._customer_type, self._request],
            dtype=np.int32,
        )

    def _get_info(self) -> Dict[str, Any]:
        ctype      = self.customer_types[self._customer_type]
        rooms_sold = self._rooms_occupied
        return {
            "rooms_occupied"       : self._rooms_occupied,
            "rooms_available"      : self.capacity - self._rooms_occupied,
            "utilisation_rate"     : self._rooms_occupied / self.capacity,
            "time_step"            : self._time_step,
            "episode_revenue"      : self._episode_revenue,
            "revenue_per_room_sold": (
                self._episode_revenue / rooms_sold if rooms_sold > 0 else 0.0
            ),
            "accepted"             : self._accepted,
            "rejected"             : self._rejected,
            "current_customer"     : ctype.name,
            "requested_rooms"      : self._request,
            "reward_per_room"      : ctype.reward_per_room,
            "scale"                : self.scale,
            "history_len"          : len(self._history),
            # ideal_revenue / regret / regret_pct added at episode end in step()
        }


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_scaled_env(n: int, **kwargs) -> HotelEnv:
    """make_scaled_env(3, capacity=20, episode_length=50) → scale-3 HotelEnv."""
    return HotelEnv(scale=n, **kwargs)


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

class GreedyAgent:
    """Always accept if capacity allows."""
    def __call__(self, obs: np.ndarray, info: Dict) -> int:
        return 1 if info["rooms_available"] >= info["requested_rooms"] else 0


class ThresholdAgent:
    """Accept only when available fraction > threshold."""
    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold

    def __call__(self, obs: np.ndarray, info: Dict) -> int:
        total      = obs[0] + info["rooms_available"]
        frac_avail = info["rooms_available"] / total
        if frac_avail > self.threshold and info["rooms_available"] >= info["requested_rooms"]:
            return 1
        return 0


class RandomAgent:
    def __call__(self, obs: np.ndarray, info: Dict) -> int:
        return int(np.random.rand() > 0.5)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env           : HotelEnv,
    agent,
    demo_backtrack: bool = False,
) -> Dict[str, float]:
    """
    Run one full episode and return a results dict.

    If demo_backtrack=True, the agent backtracks 3 steps once (at step 5)
    to demonstrate the mechanic.

    Result keys
    -----------
    total_revenue, ideal_revenue, regret, regret_pct, accepted, rejected
    """
    obs, info = env.reset()
    done      = False
    step_num  = 0
    backtracked = False

    while not done:
        action = agent(obs, info)

        # Backtrack demo: rewind 3 steps once at step 5
        if demo_backtrack and step_num == 5 and not backtracked:
            rewind = min(3, len(env._history))
            print(f"\n  [BACKTRACK] at t={info['time_step']} – rewinding {rewind} steps")
            obs, info = env.backtrack(rewind)
            print(
                f"  [BACKTRACK] restored → t={info['time_step']}, "
                f"revenue=${info['episode_revenue']:.1f}, "
                f"rooms={info['rooms_occupied']}/{env.capacity}\n"
            )
            backtracked = True
            continue

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_num += 1

    return {
        "total_revenue": info["episode_revenue"],
        "ideal_revenue": info.get("ideal_revenue", 0.0),
        "regret"       : info.get("regret",        0.0),
        "regret_pct"   : info.get("regret_pct",    0.0),
        "accepted"     : info["accepted"],
        "rejected"     : info["rejected"],
    }


# ---------------------------------------------------------------------------
# Cross-scale benchmark
# ---------------------------------------------------------------------------

def benchmark(
    n_episodes    : int       = 300,
    scales        : List[int] = [1, 2, 5],
    base_capacity : int       = 20,
    base_horizon  : int       = 50,
):
    """
    Run every agent × every scale and print a table.
    Regret % is scale-invariant – it should stay roughly constant across N
    for the same policy, confirming the scaling is faithful.
    """
    agents = {
        "Greedy"         : GreedyAgent(),
        "Threshold(0.3)" : ThresholdAgent(0.3),
        "Random"         : RandomAgent(),
    }

    cw  = 15
    cols = ["Agent", "Scale N", "Capacity", "Avg Revenue", "Ideal Revenue", "Regret $", "Regret %"]
    sep  = "=" * (cw * len(cols))

    print(f"\n{sep}")
    print(f"  Hotel RM Cross-Scale Benchmark  |  {n_episodes} episodes per cell")
    print(sep)
    print("  " + "".join(f"{h:<{cw}}" for h in cols))
    print("  " + "-" * (cw * len(cols)))

    for scale in scales:
        env = HotelEnv(
            capacity       = base_capacity,
            episode_length = base_horizon,
            scale          = scale,
            render_mode    = None,
        )
        for name, agent in agents.items():
            results    = [run_episode(env, agent) for _ in range(n_episodes)]
            avg_rev    = np.mean([r["total_revenue"] for r in results])
            avg_ideal  = np.mean([r["ideal_revenue"] for r in results])
            avg_regret = np.mean([r["regret"]        for r in results])
            avg_regpct = np.mean([r["regret_pct"]    for r in results])
            print(
                f"  {name:<{cw}}"
                f"{'×'+str(scale):<{cw}}"
                f"{env.capacity:<{cw}}"
                f"${avg_rev:<{cw-1}.1f}"
                f"${avg_ideal:<{cw-1}.1f}"
                f"${avg_regret:<{cw-1}.1f}"
                f"{avg_regpct:<{cw}.1f}%"
            )
        print("  " + "-" * (cw * len(cols)))

    print(sep)
    print("  NOTE: Regret % should be ~stable across scales for the same policy.\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # 1 ── Single episode with backtrack demo ─────────────────────────────
    print("\n══ 1. Single episode  (Greedy, scale=1, backtrack demo) ══\n")
    env   = HotelEnv(capacity=10, episode_length=20, scale=1,
                     render_mode="human", seed=42)
    stats = run_episode(env, GreedyAgent(), demo_backtrack=True)
    print(
        f"\n  Episode done → Revenue ${stats['total_revenue']:.1f}"
        f"  |  Ideal ${stats['ideal_revenue']:.1f}"
        f"  |  Regret ${stats['regret']:.1f} ({stats['regret_pct']:.1f}%)\n"
    )

    # 2 ── Scale ×3 quick check ────────────────────────────────────────────
    print("══ 2. Scale ×3 quick check  (Greedy, no render) ══")
    env3  = HotelEnv(capacity=10, episode_length=20, scale=3,
                     render_mode=None, seed=42)
    s3    = run_episode(env3, GreedyAgent())
    print(
        f"  Capacity={env3.capacity}  Horizon={env3.episode_length}"
        f"  Revenue=${s3['total_revenue']:.1f}"
        f"  Ideal=${s3['ideal_revenue']:.1f}"
        f"  Regret={s3['regret_pct']:.1f}%\n"
    )

    # 3 ── Cross-scale benchmark ───────────────────────────────────────────
    benchmark(n_episodes=300, scales=[1, 2, 5])

    # 4 ── Torch tensor demo ───────────────────────────────────────────────
    print("══ 4. Torch tensor (scale=2) ══")
    env2 = HotelEnv(scale=2, seed=0, render_mode=None)
    obs, _ = env2.reset()
    t = env2.obs_to_tensor(obs)
    print(f"  Raw obs : {obs}")
    print(f"  Tensor  : {t}  shape={list(t.shape)}  (normalised float32)\n")
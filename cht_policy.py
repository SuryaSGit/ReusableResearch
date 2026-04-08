"""
Corrected Head Count Threshold (CHT) Policy
============================================

Exact implementation following:
  Xie, Gurvich, Kucukyavuz (2024)
  "Dynamic Allocation of Reusable Resources: Logarithmic Regret in Overloaded Networks"
  Operations Research, 73(4).

The CHT policy achieves Theta(log N) regret in overloaded loss networks.

Key formulas (Section 4 of the paper):
  1. Target allocation: x*(zeta) = A_lp^{-1} (q^N - A_p * zeta)
     where zeta = current preferred-type customer counts
  2. Corrected head count for resource j:
     Sigma*_j = Sigma_j + sum_{i in A_lp, i != i_j} A_{ji} * (X*_i - X_i)
  3. Accept less-preferred type i at matched resource j_i if:
     q^N_{j_i} - Sigma*_{j_i} >= delta * log(N)

Usage:
  from cht_policy import CHTPolicy, compare_policies
  from ctmc_env import CTMCEnv

  env = CTMCEnv(N=2)
  cht = CHTPolicy(env, delta=3.0)
  compare_policies(N_values=[1, 2, 3, 5])
"""

import numpy as np
from ctmc_env import CTMCEnv, GreedyPolicy, VIPOnlyPolicy, run_policy


class CHTPolicy:
    """
    Corrected Head Count Threshold (CHT) policy from Xie et al. (2024).

    Implements the exact algorithm from Section 4 of the paper.
    """

    def __init__(self, env, delta=3.0):
        self.env = env
        self.delta = delta
        self.N = env.N
        self.A = env.A
        self.n_types = env.n_types
        self.n_resources = env.n_resources

        # Get LP solution
        if env.lp_solution is None:
            raise ValueError("LP solution not available. Install scipy.")

        y_star = env.lp_solution
        offered_load = env.arrival_rates / env.service_rates

        # Classify types (Section 3 of paper)
        self.preferred = []       # A_p: y*_i = lambda_i/mu_i
        self.less_preferred = []  # A_lp: 0 < y*_i < lambda_i/mu_i
        self.rejected = []        # A_0: y*_i = 0

        for i in range(self.n_types):
            if abs(y_star[i] - offered_load[i]) < 1e-6:
                self.preferred.append(i)
            elif y_star[i] > 1e-6:
                self.less_preferred.append(i)
            else:
                self.rejected.append(i)

        # Build A_p and A_lp submatrices
        # A_p: columns of A for preferred types
        # A_lp: columns of A for less-preferred types
        self.A_p = self.A[:, self.preferred] if self.preferred else np.zeros((self.n_resources, 0))
        self.A_lp = self.A[:, self.less_preferred] if self.less_preferred else np.zeros((self.n_resources, 0))

        # Invert A_lp (must be square and full rank under overload assumption)
        if len(self.less_preferred) == self.n_resources and self.A_lp.shape[0] == self.A_lp.shape[1]:
            self.A_lp_inv = np.linalg.inv(self.A_lp.astype(float))
        else:
            # Non-square: use pseudoinverse
            self.A_lp_inv = np.linalg.pinv(self.A_lp.astype(float))

        # Find the perfect matching: resource j -> less-preferred type i_j
        # Under overload, each resource j is matched to exactly one less-preferred type
        self.matching = {}  # resource j -> less-preferred type index (in A_lp list)
        self._compute_matching()

        # Threshold
        self.threshold = delta * np.log(self.N) if self.N > 1 else 0.0

        self.name = f"CHT (delta={delta})"

    def _compute_matching(self):
        """
        Find the perfect matching between resources and less-preferred types.
        Under the overload condition (Assumption 3.1), A_lp is full rank and
        there exists a perfect matching in the LP-residual graph.
        """
        # Simple greedy matching: for each resource, find a less-preferred type
        # that uses it and hasn't been matched yet
        matched_types = set()
        for j in range(self.n_resources):
            for idx, i in enumerate(self.less_preferred):
                if self.A[j, i] > 0 and idx not in matched_types:
                    self.matching[j] = idx  # index into self.less_preferred
                    matched_types.add(idx)
                    break

    def _target_allocation(self, state):
        """
        Compute target allocation x*(zeta) = A_lp^{-1} (q^N - A_p * zeta)

        zeta = vector of current preferred-type customer counts.
        Returns target counts for each less-preferred type.
        """
        # zeta = preferred type counts
        zeta = np.array([state[i] for i in self.preferred], dtype=float)

        # q^N - A_p * zeta
        rhs = self.env.capacities.astype(float) - self.A_p @ zeta

        # x* = A_lp^{-1} * rhs
        x_star = self.A_lp_inv @ rhs

        return x_star

    def _corrected_head_count(self, state, resource_j):
        """
        Compute corrected head count Sigma*_j for resource j.

        Sigma*_j = Sigma_j + sum_{i in A_lp, i != i_j} A_{ji} * (X*_i - X_i)

        where:
          Sigma_j = sum_{i : A_{ji}=1} x_i  (actual resource usage)
          X*_i = target allocation for less-preferred type i
          i_j = less-preferred type matched to resource j
        """
        # Actual head count
        sigma_j = sum(state[i] for i in range(self.n_types) if self.A[resource_j, i] > 0)

        # Target allocation for less-preferred types
        x_star = self._target_allocation(state)

        # Correction: sum over less-preferred types OTHER than the matched one
        matched_lp_idx = self.matching.get(resource_j, None)
        correction = 0.0
        for idx, i in enumerate(self.less_preferred):
            if idx == matched_lp_idx:
                continue  # skip the matched type
            if self.A[resource_j, i] > 0:
                correction += self.A[resource_j, i] * (x_star[idx] - state[i])

        sigma_star_j = sigma_j + correction
        return sigma_star_j

    def __call__(self, obs, info):
        """
        Decision: accept (1) or reject (0).
        """
        type_idx = info.get("current_type", 0)
        state = info.get("state", obs[:self.n_types].astype(int))

        # Check feasibility
        if not info.get("can_accept", True):
            return 0

        # Preferred types: always accept (Algorithm step 1)
        if type_idx in self.preferred:
            return 1

        # Rejected types: always reject
        if type_idx in self.rejected:
            return 0

        # Less-preferred types: check corrected head count (Algorithm step 2)
        if type_idx in self.less_preferred:
            # Find the matched resource j_i for this type
            lp_idx = self.less_preferred.index(type_idx)
            matched_resource = None
            for j, idx in self.matching.items():
                if idx == lp_idx:
                    matched_resource = j
                    break

            if matched_resource is None:
                return 1  # no matching found, accept by default

            # Compute corrected head count
            sigma_star = self._corrected_head_count(state, matched_resource)

            # Accept if: q^N_j - Sigma*_j >= delta * log(N)
            remaining = self.env.capacities[matched_resource] - sigma_star
            if remaining >= self.threshold:
                return 1
            else:
                return 0

        # Unknown type: accept
        return 1


class SmartThresholdPolicy:
    """Simple threshold: accept VIPs always, others only if enough room."""
    name = "Smart Threshold"

    def __init__(self, env, buffer_fraction=0.3):
        self.env = env
        self.buffer = buffer_fraction

    def __call__(self, obs, info):
        type_idx = info.get("current_type", 0)
        state = info.get("state", np.zeros(3))

        if not info.get("can_accept", True):
            return 0

        if type_idx == 0:  # VIP: always
            return 1

        usage = self.env.A @ state
        for j in range(self.env.n_resources):
            if self.env.A[j, type_idx] > 0:
                if usage[j] > self.env.capacities[j] * (1 - self.buffer):
                    return 0
        return 1


def compare_policies(N_values=None, n_events=200000, seed=42):
    """Compare policies across scaling factors N."""
    if N_values is None:
        N_values = [1, 2, 3, 5]

    print("=" * 75)
    print("  Policy Comparison: Xie et al. (2024) Network 1")
    print("  CHT = Corrected Head Count Threshold")
    print("=" * 75)
    print()

    for N in N_values:
        env = CTMCEnv(N=N, max_events=n_events + 10000)
        lp_bound = env.lp_bound

        policies = [
            GreedyPolicy(),
            VIPOnlyPolicy(),
            SmartThresholdPolicy(env),
            CHTPolicy(env, delta=3.0),
            CHTPolicy(env, delta=5.0),
        ]

        print(f"--- N = {N}  |  LP bound = {lp_bound:.3f}  |  "
              f"States ~ {168*N**3}  |  threshold = {3*np.log(N) if N>1 else 0:.2f} ---")
        print(f"  {'Policy':<25s} {'Rate':>10s} {'Regret':>10s}")
        print("  " + "-" * 47)

        for policy in policies:
            rate, info = run_policy(env, policy, n_events=n_events, seed=seed)
            regret = lp_bound - rate
            name = getattr(policy, 'name', policy.__class__.__name__)
            print(f"  {name:<25s} {rate:>10.3f} {regret:>10.3f}")

        print()

    print("Notes:")
    print("  - LP bound: theoretical max (not achievable by any policy)")
    print("  - Regret = LP bound - reward rate (lower is better)")
    print("  - CHT achieves O(log N) regret asymptotically (Xie et al. Thm 4.1)")
    print("  - MDP-optimal (via Value Iteration) achieves the best possible regret")
    print()


if __name__ == "__main__":
    compare_policies(N_values=[1, 2, 3], n_events=200000)

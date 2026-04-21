"""
Plotting helpers for the ablation output. Zero hard dependency on
pandas — works off the JSON dumped by `ablation.run_ablation`.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def load_logs(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return json.load(f)


def per_arm_curves(logs: List[Dict[str, Any]]) -> Dict[str, Dict[int, List[float]]]:
    """
    Group eval-reward per episode across trials.  Returns

        { arm_name: { episode: [eval_reward_trial_0, eval_reward_trial_1, ...] } }
    """
    out: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in logs:
        out[row["arm"]][row["episode"]].append(row["eval_reward"])
    return {a: dict(sorted(eps.items())) for a, eps in out.items()}


def final_performance_table(logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Take the LAST log-point per (arm, trial), then mean/CI over trials.
    """
    per_trial: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in logs:
        key = (row["arm"], row["trial"])
        cur = per_trial.get(key)
        if cur is None or row["episode"] > cur["episode"]:
            per_trial[key] = row

    by_arm: Dict[str, List[float]] = defaultdict(list)
    for (arm, _), row in per_trial.items():
        by_arm[arm].append(row["eval_reward"])

    rows = []
    for arm, rewards in sorted(by_arm.items()):
        rewards = np.asarray(rewards, dtype=float)
        n = len(rewards)
        mean = float(rewards.mean())
        std  = float(rewards.std(ddof=1)) if n > 1 else 0.0
        ci95 = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
        rows.append({
            "arm":        arm,
            "n_trials":   n,
            "mean":       mean,
            "std":        std,
            "ci95":       ci95,
        })
    return rows


def plot_learning_curves(
    logs:      List[Dict[str, Any]],
    out_path:  str,
    title:     str = "Ablation learning curves",
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = per_arm_curves(logs)

    fig, ax = plt.subplots(figsize=(8, 5))
    for arm, by_ep in sorted(curves.items()):
        xs = np.asarray(sorted(by_ep.keys()), dtype=float)
        means = np.asarray([np.mean(by_ep[int(x)]) for x in xs])
        stds  = np.asarray([np.std(by_ep[int(x)])  for x in xs])
        ax.plot(xs, means, label=arm, linewidth=1.8)
        ax.fill_between(xs, means - stds, means + stds, alpha=0.15)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Eval reward (greedy rollouts)")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_q_magnitudes(
    logs:      List[Dict[str, Any]],
    out_path:  str,
    arm_name:  str = "D_full_residual",
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in logs if r["arm"] == arm_name]
    if not rows:
        return
    rows.sort(key=lambda r: (r["trial"], r["episode"]))

    by_trial: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_trial[r["trial"]].append(r)

    fig, ax = plt.subplots(figsize=(8, 5))
    for trial, rs in sorted(by_trial.items()):
        xs = [r["episode"] for r in rs]
        ax.plot(xs, [r["q_theta_abs"] for r in rs], label=f"|Q_theta| t{trial}", alpha=0.7)
        ax.plot(xs, [r["q_cht_abs"]   for r in rs], label=f"|Q_CHT| t{trial}",   linestyle="--", alpha=0.7)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean |Q|")
    ax.set_title(f"Q magnitudes over training — {arm_name}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

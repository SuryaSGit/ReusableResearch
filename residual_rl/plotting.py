"""
Plotting helpers for the ablation output.

- `plot_learning_curves`: per-arm eval reward vs episode, optionally with
  horizontal dashed reference lines for baselines (CHT-only, LP, hindsight).
- `plot_q_magnitudes`: evolution of |Q_theta|, |Q_CHT|, |Q_final| per arm.
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
    out: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in logs:
        out[row["arm"]][row["episode"]].append(row["eval_reward"])
    return {a: dict(sorted(eps.items())) for a, eps in out.items()}


def final_performance_table(logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
    logs:       List[Dict[str, Any]],
    out_path:   str,
    title:      str = "Ablation learning curves",
    baselines:  Optional[Dict[str, Any]] = None,
    env_name:   str = "hotel",
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = per_arm_curves(logs)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for arm, by_ep in sorted(curves.items()):
        xs = np.asarray(sorted(by_ep.keys()), dtype=float)
        means = np.asarray([np.mean(by_ep[int(x)]) for x in xs])
        stds  = np.asarray([np.std(by_ep[int(x)])  for x in xs])
        ax.plot(xs, means, label=arm, linewidth=1.8)
        ax.fill_between(xs, means - stds, means + stds, alpha=0.15)

    # Overlay baseline references
    if baselines:
        if env_name.lower() == "hotel":
            refs = [
                ("CHT-only",  baselines.get("cht_only",   {}).get("mean_reward")),
                ("Hindsight", baselines.get("hindsight",  {}).get("mean_reward")),
                ("AcceptAll", baselines.get("accept_all", {}).get("mean_reward")),
                ("RejectAll", baselines.get("reject_all", {}).get("mean_reward")),
            ]
        else:
            refs = [
                ("CHT-only",  baselines.get("cht_only",   {}).get("mean_reward")),
                ("AcceptAll", baselines.get("accept_all", {}).get("mean_reward")),
                ("RejectAll", baselines.get("reject_all", {}).get("mean_reward")),
            ]
        styles = ["--", "-.", ":", (0, (3, 1, 1, 1))]
        for i, (name, val) in enumerate(refs):
            if val is None:
                continue
            ax.axhline(val, linestyle=styles[i % len(styles)], linewidth=1.2,
                       color="gray", alpha=0.75, label=f"{name} ({val:.1f})")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Eval reward (greedy rollouts)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_q_magnitudes(
    logs:      List[Dict[str, Any]],
    out_path:  str,
    arm_name:  Optional[str] = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # If `arm_name` is None, plot a subplot grid over all arms that have logs.
    arms_in_logs = sorted({r["arm"] for r in logs})
    if arm_name is not None:
        arms_in_logs = [a for a in arms_in_logs if a == arm_name]
    if not arms_in_logs:
        return

    n = len(arms_in_logs)
    fig, axes = plt.subplots(n, 1, figsize=(8, 3.0 * n), squeeze=False)
    for idx, arm in enumerate(arms_in_logs):
        ax = axes[idx, 0]
        rows = [r for r in logs if r["arm"] == arm]
        rows.sort(key=lambda r: (r["trial"], r["episode"]))
        by_trial: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for r in rows:
            by_trial[r["trial"]].append(r)
        for trial, rs in sorted(by_trial.items()):
            xs = [r["episode"] for r in rs]
            ax.plot(xs, [r.get("q_theta_abs", r.get("q_theta", 0.0)) for r in rs],
                    label=f"|Q_theta| t{trial}", alpha=0.85)
            ax.plot(xs, [r.get("q_cht_abs",   r.get("q_cht",   0.0)) for r in rs],
                    linestyle="--", label=f"|Q_CHT| t{trial}",   alpha=0.85)
            ax.plot(xs, [r.get("q_final_abs", r.get("q_final", 0.0)) for r in rs],
                    linestyle=":",  label=f"|Q_final| t{trial}", alpha=0.85)
        ax.set_title(f"Q magnitudes — {arm}")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Mean |Q|")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

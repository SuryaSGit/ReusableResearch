"""
Load `logs.json` from a results/ dir and emit:

  * `summary.csv`         — final-performance table (arm, mean, CI95)
  * `learning_curves.png` — eval reward vs episode, one line per arm
  * `q_magnitudes.png`    — |Q_theta| vs |Q_CHT| over training for arm D
                            (only if logs contain an arm with "residual" in
                            the name; otherwise skipped)

Usage:
    python3 experiments/analyze.py results/hotel_hard_t1_ep200_20260421-120000
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from residual_rl.plotting import (
    final_performance_table,
    load_logs,
    plot_learning_curves,
    plot_q_magnitudes,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    args = ap.parse_args()

    logs_path = os.path.join(args.run_dir, "logs.json")
    if not os.path.exists(logs_path):
        print(f"ERROR: {logs_path} not found", file=sys.stderr)
        sys.exit(1)

    logs = load_logs(logs_path)
    table = final_performance_table(logs)

    # Write summary CSV
    summary_path = os.path.join(args.run_dir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    print(f"Wrote {summary_path}")

    # Print a human table
    print(f"\n{'arm':<25s} {'n':>4s} {'mean':>10s} {'std':>10s} {'ci95':>10s}")
    print("-" * 60)
    for row in table:
        print(f"{row['arm']:<25s} {row['n_trials']:>4d} {row['mean']:>10.2f} {row['std']:>10.2f} {row['ci95']:>10.2f}")

    # Plots
    curves_path = os.path.join(args.run_dir, "learning_curves.png")
    plot_learning_curves(logs, curves_path, title=f"Ablation — {os.path.basename(args.run_dir)}")
    print(f"Wrote {curves_path}")

    # Q magnitudes for the residual arm, if present
    arm_names = {row["arm"] for row in table}
    for cand in ("D_full_residual",):
        if cand in arm_names:
            qmag_path = os.path.join(args.run_dir, "q_magnitudes.png")
            plot_q_magnitudes(logs, qmag_path, arm_name=cand)
            print(f"Wrote {qmag_path}")
            break


if __name__ == "__main__":
    main()

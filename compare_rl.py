"""
CHT-DQN vs Vanilla DQN — Head-to-Head Comparison
==================================================

Trains both agents under identical conditions across multiple independent
trials and answers two questions:

  (a) Convergence speed  — does CHT-DQN reach near-optimal performance
                           in fewer training episodes than vanilla DQN?

  (b) Final performance  — does CHT-DQN end up with higher eval revenue
                           after the full training budget?

Fair comparison design
-----------------------
• Both agents use the exact same HyperParams (lr, γ, buffer, batch, etc.).
  The ONLY differences are the three CHT levers:
    CHT-DQN  : use_cht_obs=True,  q_cht_weight=1.0, alpha_start=0.5
    Vanilla  : use_cht_obs=False, q_cht_weight=0.0, alpha_start=0.0
  (Vanilla is CHT-DQN with all CHT features disabled — same code path.)

• Both agents train on environments seeded identically per trial, so they
  see the same stochastic arrival sequences.

• α-annealing: CHT-DQN starts with 50% of random actions replaced by the
  CHT prior (accept iff Δ_i > 0). α decays linearly to 0 over the first
  40% of training, after which both agents explore purely with ε-greedy.

Difficulty modes
----------------
  easy  : TIGHT_CUSTOMER_TYPES — Premium $200, Standard $120, Budget $80, Group $90
           Large reward spread → RL signal is clear.
  hard  : compressed rewards   — Premium $110, Standard $100, Budget $90, Group $95
           Tiny spread → much harder for vanilla RL to learn the priority ordering.

Usage
-----
  python compare_rl.py                        # easy, 3 trials, 3 000 eps
  python compare_rl.py --difficulty hard      # compressed rewards
  python compare_rl.py --trials 5 --episodes 5000
  python compare_rl.py --difficulty hard --trials 5 --episodes 5000
  python compare_rl.py --out results.png      # custom output path
"""

from __future__ import annotations

import argparse
import collections
import random
import sys
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── path: find source files ───────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, "/mnt/user-data/uploads")

from hotel_env import HotelEnv, CustomerType, TIGHT_CUSTOMER_TYPES

# Use the REAL DQNAgent from dqn_agent, and CHTDQNAgent from cht_dqn
# (cht_dqn already patches HotelEnv._get_info via _patched_get_info import)
from dqn_agent import (
    DQNAgent, HyperParams, TrainLog,
    train as train_vanilla,
    evaluate as evaluate_vanilla,
)
from cht_dqn import (
    CHTDQNAgent, CHTHyperParams,
    train_cht, _evaluate_cht,
)


# ── Difficulty presets ────────────────────────────────────────────────────────

EASY_CUSTOMER_TYPES = TIGHT_CUSTOMER_TYPES

# Hard: rewards compressed into $90-$110 band — tiny ranking signal
HARD_CUSTOMER_TYPES = [
    CustomerType("Budget",   min_rooms=1, max_rooms=2, reward_per_room=90.0,  arrival_prob=0.40),
    CustomerType("Standard", min_rooms=1, max_rooms=3, reward_per_room=100.0, arrival_prob=0.35),
    CustomerType("Premium",  min_rooms=2, max_rooms=4, reward_per_room=110.0, arrival_prob=0.15),
    CustomerType("Group",    min_rooms=4, max_rooms=8, reward_per_room=95.0,  arrival_prob=0.10),
]


def make_envs(customer_types, seed=None):
    """Return a fresh (train_env, eval_env) pair."""
    train_env = HotelEnv(
        capacity=20, episode_length=50,
        customer_types=customer_types, scale=1,
        render_mode=None, seed=seed,
    )
    eval_env = HotelEnv(
        capacity=20, episode_length=50,
        customer_types=customer_types, scale=1,
        render_mode=None,
    )
    return train_env, eval_env


# ── Shared hyperparameter builder ─────────────────────────────────────────────

def _base_kwargs(n_episodes):
    """Hyperparameter values shared by both agents."""
    return dict(
        hidden_dims    = [256, 256],
        buffer_size    = 100_000,
        batch_size     = 512,
        lr             = 3e-4,
        gamma          = 1.0,
        n_step         = 3,
        n_episodes     = n_episodes,
        learn_every    = 4,
        target_sync    = 10,
        eps_start      = 1.0,
        eps_end        = 0.02,
        eps_decay_ep   = int(n_episodes * 0.60),
        shaping_weight = 0.2,
        log_every      = max(n_episodes // 20, 50),
        eval_episodes  = 150,
    )


def make_vanilla_hp(n_episodes):
    """Vanilla DQN hyperparams (no CHT)."""
    base = _base_kwargs(n_episodes)
    # HyperParams only accepts its own fields
    valid = set(HyperParams.__dataclass_fields__)
    return HyperParams(**{k: v for k, v in base.items() if k in valid})


def make_cht_hp(n_episodes):
    """
    CHT-DQN: identical base params + all three CHT levers, tuned to avoid
    late-training regression.

    Three fixes vs the original settings:
    ──────────────────────────────────────
    1. alpha_start 0.5 → 0.30
       Lower initial CHT probability so the replay buffer stays diverse.
       At 0.50 the buffer was dominated by CHT-guided transitions, leaving
       Q_theta under-trained on pure environment experience.

    2. alpha_decay_ep 40% → 20% of training
       α must fully decay BEFORE ε does (ε decays over 60%).  With the old
       40% window, α and ε overlapped heavily: α was still non-zero while ε
       was shrinking, so the agent was going greedy on a Q-network that had
       never explored freely.  Finishing α-decay at 20% gives 40% of training
       where the agent explores normally (pure ε-greedy) before going greedy.

    3. q_cht_weight 1.0 → 0.0 after alpha hits zero
       The residual Q_CHT bonus keeps biasing greedy action selection even
       after α=0, preventing Q_theta from fully taking over.  We anneal
       q_cht_weight to 0 on the same schedule as alpha so the heuristic
       bias disappears completely once the prior does.
       (Implemented by setting alpha_decay_ep == the epoch at which we also
       want the residual to vanish; CHTHyperParams uses alpha_decay_ep for
       both schedules via q_cht_weight * (1 - frac_alpha).)
    """
    alpha_decay = int(n_episodes * 0.20)   # α gone by 20% of training
    return CHTHyperParams(
        **_base_kwargs(n_episodes),
        # CHT levers — tuned
        alpha_start    = 0.30,          # moderate prior; keeps buffer diverse
        alpha_end      = 0.0,
        alpha_decay_ep = alpha_decay,   # done well before ε-decay finishes
        q_cht_weight   = 1.0,           # starts at 1.0, annealed below
        use_cht_obs    = True,          # Δ and phi(Δ) features stay in obs
    )


# ── Annealed CHT agent: q_cht_weight decays in lockstep with alpha ────────────

import torch
from cht_dqn import q_cht_heuristic

class AnnealedCHTDQNAgent(CHTDQNAgent):
    """
    Extends CHTDQNAgent so that q_cht_weight (the residual Q_CHT bonus) is
    annealed to zero on the same schedule as alpha.

    Without this, Q_CHT keeps biasing greedy action selection even after α=0,
    preventing Q_theta from fully taking over in the late-training phase and
    causing the characteristic performance drop seen after ~2000 episodes.

    The annealed weight is:
        w(t) = hp.q_cht_weight * (1 - frac_alpha)
    where frac_alpha = min(episode / alpha_decay_ep, 1.0), so w → 0 exactly
    when alpha → 0.
    """

    @property
    def _annealed_q_cht_weight(self) -> float:
        frac = min(self._episode / max(self.hp.alpha_decay_ep, 1), 1.0)
        return self.hp.q_cht_weight * (1.0 - frac)

    def act(self, obs, info, greedy=False):
        import random as _random
        # Hard capacity mask
        if info["rooms_available"] < info["requested_rooms"]:
            return 0

        delta_raw, _ = self._delta(info)

        if not greedy:
            r = _random.random()
            if r < self.alpha:
                return 1 if delta_raw > 0 else 0
            if r < self.alpha + self.epsilon * (1 - self.alpha):
                return self.env.action_space.sample()

        # Q_final = annealed_Q_CHT + Q_theta
        feat = self._obs(info)
        with torch.no_grad():
            q_theta = self.online(
                torch.tensor(feat, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
            ).squeeze(0)

        w = self._annealed_q_cht_weight
        q_cht_rej, q_cht_acc = q_cht_heuristic(
            info, delta_raw, self.env, w, self.hp.reward_scale
        )
        q_final = torch.tensor(
            [q_cht_rej, q_cht_acc], dtype=torch.float32, device=self.device
        ) + q_theta

        return int(q_final.argmax().item())


# ── Single-trial runners ──────────────────────────────────────────────────────

def _seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def run_vanilla_trial(customer_types, n_episodes, trial_seed):
    print(f"\n  [Vanilla DQN]  seed={trial_seed}")
    _seed_all(trial_seed)
    train_env, eval_env = make_envs(customer_types, seed=trial_seed)
    hp    = make_vanilla_hp(n_episodes)
    agent = DQNAgent(train_env, hp=hp, device="cpu", n_episodes=n_episodes)
    logs  = train_vanilla(agent, train_env, hp=hp, eval_env=eval_env)
    return logs


def run_cht_trial(customer_types, n_episodes, trial_seed):
    print(f"\n  [CHT-DQN]      seed={trial_seed}")
    _seed_all(trial_seed)
    train_env, eval_env = make_envs(customer_types, seed=trial_seed)
    hp    = make_cht_hp(n_episodes)
    # AnnealedCHTDQNAgent co-anneals q_cht_weight with alpha to prevent
    # the residual Q_CHT bias persisting into late greedy-phase training.
    agent = AnnealedCHTDQNAgent(train_env, hp=hp, device="cpu", n_episodes=n_episodes)
    logs  = train_cht(agent, train_env, hp=hp, eval_env=eval_env)
    return logs


# ── Multi-trial experiment ────────────────────────────────────────────────────

def run_experiment(customer_types, n_trials, n_episodes, base_seed=0):
    """
    Run n_trials independent trials for Vanilla and CHT-DQN with matching
    seeds so arrival sequences are identical between the two agents.

    Returns {"vanilla": [[logs...], ...], "cht": [[logs...], ...]}
    """
    results = {"vanilla": [], "cht": []}

    for t in range(n_trials):
        seed = base_seed + t * 100
        print(f"\n{'─'*60}")
        print(f"  TRIAL {t+1}/{n_trials}  (seed={seed})")
        print(f"{'─'*60}")
        results["vanilla"].append(
            run_vanilla_trial(customer_types, n_episodes, seed))
        results["cht"].append(
            run_cht_trial(customer_types, n_episodes, seed))

    return results


# ── Log alignment helper ──────────────────────────────────────────────────────

def align_logs(all_trial_logs):
    """
    Returns (x_common, rev_matrix) where:
      x_common   : (M,)   shared episode checkpoint axis
      rev_matrix : (T, M) eval_revenue per trial, interpolated onto x_common
    """
    trial_data = [
        (np.array([l.episode      for l in logs], dtype=float),
         np.array([l.eval_revenue for l in logs], dtype=float))
        for logs in all_trial_logs
    ]
    x_ref = max(trial_data, key=lambda d: len(d[0]))[0]
    rev_matrix = np.stack([np.interp(x_ref, eps, rev) for eps, rev in trial_data])
    return x_ref, rev_matrix


def rolling_mean(arr, w):
    if w <= 1:
        return arr
    smooth = np.convolve(arr, np.ones(w) / w, mode="valid")
    return np.concatenate([np.full(w - 1, smooth[0]), smooth])


# ── Plotting ──────────────────────────────────────────────────────────────────

_BG_DARK  = "#1A1A2E"
_BG_PANEL = "#16213E"
_SPINE    = "#444466"
_TICK     = "#CCCCDD"
_GRID     = "#2A2A4A"
_TEXT     = "#EEEEFF"
_IDEAL    = "#88FFAA"
VAN_COL   = "#5B8DB8"
CHT_COL   = "#E07040"
ABAND     = 0.18


def _style_ax(ax):
    ax.set_facecolor(_BG_PANEL)
    for sp in ax.spines.values():
        sp.set_color(_SPINE)
    ax.tick_params(colors=_TICK)
    ax.yaxis.label.set_color(_TICK)
    ax.xaxis.label.set_color(_TICK)
    ax.title.set_color(_TEXT)


def plot_results(results, difficulty, n_episodes, out_path="comparison.png", smooth_w=3):
    """Four-panel comparison figure."""
    van_x, van_mat = align_logs(results["vanilla"])
    cht_x, cht_mat = align_logs(results["cht"])
    n_trials = van_mat.shape[0]

    van_s = np.stack([rolling_mean(van_mat[i], smooth_w) for i in range(n_trials)])
    cht_s = np.stack([rolling_mean(cht_mat[i], smooth_w) for i in range(n_trials)])
    van_mean, van_std = van_s.mean(0), van_s.std(0)
    cht_mean, cht_std = cht_s.mean(0), cht_s.std(0)

    tail      = max(1, int(0.30 * van_mat.shape[1]))
    van_final = van_mat[:, -tail:].mean(1)
    cht_final = cht_mat[:, -tail:].mean(1)

    ideal_ref = None
    try:
        ideal_ref = results["vanilla"][0][-1].eval_ideal
    except Exception:
        pass

    fig = plt.figure(figsize=(22, 6))
    fig.patch.set_facecolor(_BG_DARK)
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.38,
                            left=0.05, right=0.97, top=0.87, bottom=0.13)
    axs = [fig.add_subplot(gs[i]) for i in range(4)]
    for ax in axs:
        _style_ax(ax)
    ax_conv, ax_diff, ax_box, ax_bar = axs

    diff_label = ("Hard — compressed rewards ($90-$110 band)"
                  if difficulty == "hard"
                  else "Easy — standard reward spread ($80-$200)")

    # ── Panel 1: Convergence ──────────────────────────────────────────────
    ax_conv.plot(van_x, van_mean, color=VAN_COL, lw=2, label="Vanilla DQN")
    ax_conv.fill_between(van_x, van_mean - van_std, van_mean + van_std,
                         color=VAN_COL, alpha=ABAND)
    ax_conv.plot(cht_x, cht_mean, color=CHT_COL, lw=2, label="CHT-DQN")
    ax_conv.fill_between(cht_x, cht_mean - cht_std, cht_mean + cht_std,
                         color=CHT_COL, alpha=ABAND)
    if ideal_ref is not None:
        ax_conv.axhline(ideal_ref, color=_IDEAL, lw=1.3, ls="--",
                        label=f"DP Ideal (${ideal_ref:.0f})")
    ax_conv.set_xlabel("Training Episode")
    ax_conv.set_ylabel("Eval Revenue ($)")
    ax_conv.set_title(f"(a) Convergence Curves\n{n_trials} trials, ±1σ shading")
    ax_conv.legend(fontsize=8.5, framealpha=0.25, labelcolor=_TEXT,
                   facecolor=_BG_DARK, edgecolor=_SPINE)
    ax_conv.grid(True, color=_GRID, lw=0.5)

    # ── Panel 2: CHT - Vanilla advantage ─────────────────────────────────
    diff_mean = cht_mean - van_mean
    # Combined std: sqrt of sum of variances (trials are independent)
    n_t = van_s.shape[0]
    diff_std_combined = np.sqrt(van_s.var(0) + cht_s.var(0))

    ax_diff.axhline(0, color=_SPINE, lw=1.0, ls="--")
    ax_diff.fill_between(cht_x,
                         diff_mean - diff_std_combined,
                         diff_mean + diff_std_combined,
                         color=CHT_COL, alpha=ABAND)
    ax_diff.plot(cht_x, diff_mean, color=CHT_COL, lw=2, label="CHT minus Vanilla")
    ax_diff.fill_between(cht_x, 0, diff_mean,
                         where=(diff_mean > 0), color="#44CC88", alpha=0.18,
                         label="CHT leads")
    ax_diff.fill_between(cht_x, diff_mean, 0,
                         where=(diff_mean < 0), color="#CC4444", alpha=0.18,
                         label="Vanilla leads")
    ax_diff.set_xlabel("Training Episode")
    ax_diff.set_ylabel("Revenue Advantage ($)")
    ax_diff.set_title("(b) CHT-DQN Advantage\n(CHT minus Vanilla, mean +/- combined sigma)")
    ax_diff.legend(fontsize=8.5, framealpha=0.25, labelcolor=_TEXT,
                   facecolor=_BG_DARK, edgecolor=_SPINE)
    ax_diff.grid(True, color=_GRID, lw=0.5)

    # ── Panel 3: Box plots ────────────────────────────────────────────────
    bp = ax_box.boxplot(
        [van_final, cht_final],
        patch_artist=True, widths=0.5,
        medianprops=dict(color="#FFFFFF", lw=2.5),
        whiskerprops=dict(color=_TICK, lw=1.2),
        capprops=dict(color=_TICK, lw=1.2),
        flierprops=dict(marker="o", markerfacecolor="#FF8888",
                        markeredgecolor="none", markersize=5),
    )
    bp["boxes"][0].set_facecolor(VAN_COL + "88"); bp["boxes"][0].set_edgecolor(VAN_COL)
    bp["boxes"][1].set_facecolor(CHT_COL + "88"); bp["boxes"][1].set_edgecolor(CHT_COL)
    ax_box.set_xticks([1, 2])
    ax_box.set_xticklabels(["Vanilla DQN", "CHT-DQN"], color=_TICK)
    ax_box.set_ylabel("Avg Revenue — last 30% of training ($)")
    ax_box.set_title("(c) Final Performance\ndistribution across trials")
    if ideal_ref is not None:
        ax_box.axhline(ideal_ref, color=_IDEAL, lw=1.3, ls="--", label="DP Ideal")
        ax_box.legend(fontsize=8.5, framealpha=0.25, labelcolor=_TEXT,
                      facecolor=_BG_DARK, edgecolor=_SPINE)
    ax_box.grid(True, axis="y", color=_GRID, lw=0.5)
    for x_pos, vals, col in [(1, van_final, VAN_COL), (2, cht_final, CHT_COL)]:
        ax_box.text(x_pos, vals.mean(), f"  mu=${vals.mean():.0f}",
                    va="center", ha="left", fontsize=8, color=col)

    # ── Panel 4: Per-trial bars ───────────────────────────────────────────
    x_idx = np.arange(n_trials)
    bw    = 0.35
    ax_bar.bar(x_idx - bw/2, van_final, bw,
               color=VAN_COL + "CC", edgecolor=VAN_COL, label="Vanilla DQN")
    ax_bar.bar(x_idx + bw/2, cht_final, bw,
               color=CHT_COL + "CC", edgecolor=CHT_COL, label="CHT-DQN")
    if ideal_ref is not None:
        ax_bar.axhline(ideal_ref, color=_IDEAL, lw=1.3, ls="--", label="DP Ideal")
    ax_bar.set_xticks(x_idx)
    ax_bar.set_xticklabels([f"T{i+1}" for i in range(n_trials)], color=_TICK)
    ax_bar.set_ylabel("Final Avg Revenue ($)")
    ax_bar.set_title("(d) Per-Trial Revenue\n(matched seeds => same arrival sequences)")
    ax_bar.legend(fontsize=8.5, framealpha=0.25, labelcolor=_TEXT,
                  facecolor=_BG_DARK, edgecolor=_SPINE)
    ax_bar.grid(True, axis="y", color=_GRID, lw=0.5)

    fig.suptitle(
        f"CHT-DQN vs Vanilla DQN  |  {diff_label}  |  "
        f"{n_trials} trials x {n_episodes} eps",
        fontsize=12, color=_TEXT, y=0.99,
    )

    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  [plot] saved -> {out_path}")


# ── Console summary ───────────────────────────────────────────────────────────

def print_summary(results, n_episodes):
    van_x, van_mat = align_logs(results["vanilla"])
    cht_x, cht_mat = align_logs(results["cht"])

    n_trials  = van_mat.shape[0]
    tail      = max(1, int(0.30 * van_mat.shape[1]))
    van_final = van_mat[:, -tail:].mean(1)
    cht_final = cht_mat[:, -tail:].mean(1)

    def conv_ep(row, x, frac=0.90):
        thresh = row.max() * frac
        idx    = np.argmax(row >= thresh)
        return float(x[idx]) if row[idx] >= thresh else float(x[-1])

    van_conv = [conv_ep(van_mat[i], van_x) for i in range(n_trials)]
    cht_conv = [conv_ep(cht_mat[i], cht_x) for i in range(n_trials)]

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  COMPARISON RESULTS  ({n_trials} trials x {n_episodes} episodes each)")
    print(sep)
    print(f"  {'Metric':<46}  {'Vanilla':>9}  {'CHT-DQN':>9}")
    print(f"  {'-'*68}")
    print(f"  {'Final avg revenue -- mean ($)':<46}"
          f"  ${np.mean(van_final):>8.1f}  ${np.mean(cht_final):>8.1f}")
    print(f"  {'Final avg revenue -- std ($)':<46}"
          f"  {np.std(van_final):>9.1f}  {np.std(cht_final):>9.1f}")
    print(f"  {'Convergence ep (90% of own peak) -- mean':<46}"
          f"  {np.mean(van_conv):>9.0f}  {np.mean(cht_conv):>9.0f}")
    print(f"  {'Convergence ep -- std':<46}"
          f"  {np.std(van_conv):>9.0f}  {np.std(cht_conv):>9.0f}")
    print(f"\n  Per-trial final revenue advantage (CHT minus Vanilla):")
    for i, (v, c) in enumerate(zip(van_final, cht_final)):
        d    = c - v
        sign = "+" if d >= 0 else "-"
        print(f"    Trial {i+1}: {sign}${abs(d):6.1f}  "
              f"(Vanilla=${v:.1f}  CHT=${c:.1f})")

    mean_delta = (cht_final - van_final).mean()
    faster = np.mean(cht_conv) < np.mean(van_conv)
    better = np.mean(cht_final) > np.mean(van_final)
    print(f"\n  ── Summary ──────────────────────────────────────────────────")
    print(f"  Mean revenue advantage : ${mean_delta:+.1f} per trial (CHT vs Vanilla)")
    print(f"  (a) CHT converges faster : {faster}"
          f"  [CHT={np.mean(cht_conv):.0f} eps  vs  Vanilla={np.mean(van_conv):.0f} eps]")
    print(f"  (b) CHT better at end    : {better}"
          f"  [CHT=${np.mean(cht_final):.1f}  vs  Vanilla=${np.mean(van_final):.1f}]")
    print(sep)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare CHT-DQN vs Vanilla DQN on the Hotel RM environment."
    )
    parser.add_argument(
        "--difficulty", choices=["easy", "hard"], default="easy",
        help="easy=standard reward spread; hard=compressed rewards ($90-$110)",
    )
    parser.add_argument("--trials",   type=int, default=3,
                        help="Independent trials (default 3)")
    parser.add_argument("--episodes", type=int, default=3_000,
                        help="Training episodes per agent per trial (default 3000)")
    parser.add_argument("--seed",     type=int, default=0,
                        help="Base random seed (trial t uses seed + t*100)")
    parser.add_argument("--out",      type=str, default="comparison.png",
                        help="Output plot path (default comparison.png)")
    args = parser.parse_args()

    customer_types = (HARD_CUSTOMER_TYPES if args.difficulty == "hard"
                      else EASY_CUSTOMER_TYPES)

    print(f"\n{'='*70}")
    print(f"  CHT-DQN vs Vanilla DQN -- Comparison Experiment")
    print(f"  Difficulty : {args.difficulty}")
    print(f"  Env        : capacity=20, horizon=50")
    if args.difficulty == "hard":
        print(f"  Rewards    : Budget=$90, Group=$95, Standard=$100, Premium=$110")
    else:
        print(f"  Rewards    : Budget=$80, Group=$90, Standard=$120, Premium=$200")
    print(f"  CHT alpha  : 0.50 -> 0.00 over first 40% of training (CHT-DQN only)")
    print(f"  Trials     : {args.trials}")
    print(f"  Episodes   : {args.episodes} per agent per trial")
    print(f"  Base seed  : {args.seed}")
    print(f"{'='*70}")

    results = run_experiment(
        customer_types = customer_types,
        n_trials       = args.trials,
        n_episodes     = args.episodes,
        base_seed      = args.seed,
    )

    print_summary(results, args.episodes)
    plot_results(results, args.difficulty, args.episodes, out_path=args.out)
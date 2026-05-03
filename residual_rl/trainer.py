"""
Generic train/eval loop for the residual-RL agent. Env-agnostic: callers
pass env factories and an obs-builder + prior factory.

Output:
    - `TrainLog` rows (as list) — flattened diagnostic record per log-point.
    - Optional persistent logs: log.jsonl + log.csv per arm/trial when a
      `log_dir` is supplied. Baselines are passed in and interleaved into
      each row so eval_reward, gap_pct, and regret are visible side-by-side.
    - Pretty-printed stdout table with a baseline reference row before
      training and a refresher every N log rows.

Log columns differ between environments — see `HotelSchema` and
`CTMCSchema` at the bottom of this file.
"""

from __future__ import annotations

import csv
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from residual_rl.cht_prior    import ArmConfig
from residual_rl.residual_dqn import ResidualDQNAgent, ResidualHP


# ---------------------------------------------------------------------------
# Trainer config + log row schema
# ---------------------------------------------------------------------------

@dataclass
class TrainerConfig:
    n_episodes:    int    = 3000
    log_every:     int    = 50
    eval_episodes: int    = 20
    seed:          int    = 0
    device:        str    = "cpu"
    baseline_refresh_every: int = 10   # print baseline row every N log rows
    env_name:      str    = "hotel"    # "hotel" or "ctmc" — drives table format


@dataclass
class TrainLog:
    """
    Back-compat summary row. The richer per-row dict stored in jsonl/csv
    includes more columns; this dataclass is preserved so callers that
    previously consumed `List[TrainLog]` still work.
    """
    arm:              str
    trial:            int
    episode:          int
    epsilon:          float
    alpha:            float
    train_reward:     float
    eval_reward:      float
    eval_reward_std:  float
    avg_loss:         float
    q_theta_abs:      float
    q_cht_abs:        float
    q_final_abs:      float = 0.0
    q_theta_over_final: float = 0.0
    lr:               float = 0.0
    wallclock_s:      float = 0.0
    gap_pct:          float = 0.0
    regret:           float = 0.0
    eval_rate:        float = 0.0


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Probe-state builder
# ---------------------------------------------------------------------------

def build_probe_set(
    env_factory:         Callable[[int], Any],
    prior_factory:       Callable[[Any, ArmConfig], Any],
    obs_builder_factory: Callable[[Any], Any],
    arm:                 ArmConfig,     # only used to know delta-feature shape
    episode_runner:      Callable,
    n_probe:             int = 64,
    seed:                int = 0,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Sample a fixed set of `(obs, info)` pairs by running a short random
    rollout on a dedicated probe env. Kept identical across arms and
    log-points.

    Observation dim is derived with the arm's `use_delta_features` flag
    so probe_obs aligns with the agent.build_feat() output.
    """
    probe_env = env_factory(7_777_777 + seed)
    prior     = prior_factory(probe_env, arm)
    obs_build = obs_builder_factory(probe_env)

    # Ensure arrival stream is deterministic for the probe.
    np.random.seed(7_777_777 + seed)
    random.seed(7_777_777 + seed)

    _, info = probe_env.reset(seed=7_777_777 + seed) \
              if hasattr(probe_env, "reset") else (None, {})
    prior.reset_allocation() if hasattr(prior, "reset_allocation") else None

    probe_infos: List[Dict[str, Any]] = []
    probe_feats: List[np.ndarray]     = []
    done = False
    steps_taken = 0
    while len(probe_infos) < n_probe and not done and steps_taken < 5 * n_probe:
        info_snap = {k: (np.asarray(v).copy() if isinstance(v, np.ndarray) else v)
                     for k, v in info.items()}
        # Build feat matching arm config
        base = obs_build(info_snap)
        if arm.use_delta_features:
            feat = np.concatenate([base, prior.delta_features(info_snap)])
        else:
            feat = base
        probe_feats.append(feat)
        probe_infos.append(info_snap)

        # Random action to walk the env
        a = random.randint(0, prior.n_actions - 1)
        if hasattr(prior, "update_allocation"):
            prior.update_allocation(info_snap, a)
        _, _, term, trunc, info = probe_env.step(a)
        done = term or trunc or (info.get("current_type") is None and "state" in info)
        steps_taken += 1
        if done:
            # reset to keep sampling if we still need states
            _, info = probe_env.reset()
            if hasattr(prior, "reset_allocation"):
                prior.reset_allocation()
            done = False

    if len(probe_feats) == 0:
        return np.zeros((0, 1), dtype=np.float32), []
    return np.stack(probe_feats).astype(np.float32), probe_infos


# ---------------------------------------------------------------------------
# Pretty-printed table
# ---------------------------------------------------------------------------

# Column definitions: (key, header, width, formatter)
_HOTEL_COLS: List[Tuple[str, str, int, Callable[[Any], str]]] = [
    ("episode",     "Ep",        6,  lambda v: f"{int(v):>6d}"),
    ("arm",         "Arm",       18, lambda v: f"{str(v):<18s}"),
    ("train_reward","TrainR",    9,  lambda v: f"{float(v):>9.1f}"),
    ("eval_reward", "EvalR",     9,  lambda v: f"{float(v):>9.1f}"),
    ("cht_ref",     "CHT_ref",   9,  lambda v: "—".rjust(9) if v is None else f"{float(v):>9.1f}"),
    ("hindsight",   "Hindsight", 10, lambda v: "—".rjust(10) if v is None else f"{float(v):>10.1f}"),
    ("gap_pct",     "Gap%",      7,  lambda v: "—".rjust(7)  if v is None else f"{float(v):>7.2f}"),
    ("q_theta",     "|Qθ|",      8,  lambda v: f"{float(v):>8.3f}"),
    ("q_cht",       "|Q_CHT|",   9,  lambda v: f"{float(v):>9.3f}"),
    ("q_final",     "|Q_f|",     8,  lambda v: f"{float(v):>8.3f}"),
    ("q_theta_over_final", "θ/f", 6, lambda v: f"{float(v):>6.2f}"),
    ("lr",          "lr",        9,  lambda v: f"{float(v):>9.2e}"),
    ("epsilon",     "ε",         6,  lambda v: f"{float(v):>6.3f}"),
    ("alpha",       "α",         6,  lambda v: "—".rjust(6)  if v is None else f"{float(v):>6.3f}"),
]

_CTMC_COLS: List[Tuple[str, str, int, Callable[[Any], str]]] = [
    ("episode",     "Ep",        6,  lambda v: f"{int(v):>6d}"),
    ("arm",         "Arm",       18, lambda v: f"{str(v):<18s}"),
    ("train_reward","TrainR",    9,  lambda v: f"{float(v):>9.2f}"),
    ("eval_reward", "EvalR",     9,  lambda v: f"{float(v):>9.2f}"),
    ("cht_ref",     "CHT_ref",   9,  lambda v: "—".rjust(9) if v is None else f"{float(v):>9.2f}"),
    ("lp_ub",       "LP_UB",     8,  lambda v: "—".rjust(8) if v is None else f"{float(v):>8.2f}"),
    ("regret",      "Regret",    8,  lambda v: "—".rjust(8) if v is None else f"{float(v):>8.2f}"),
    ("gap_pct",     "Gap%",      7,  lambda v: "—".rjust(7) if v is None else f"{float(v):>7.2f}"),
    ("q_theta",     "|Qθ|",      8,  lambda v: f"{float(v):>8.3f}"),
    ("q_cht",       "|Q_CHT|",   9,  lambda v: f"{float(v):>9.3f}"),
    ("q_final",     "|Q_f|",     8,  lambda v: f"{float(v):>8.3f}"),
    ("q_theta_over_final", "θ/f", 6, lambda v: f"{float(v):>6.2f}"),
    ("lr",          "lr",        9,  lambda v: f"{float(v):>9.2e}"),
    ("epsilon",     "ε",         6,  lambda v: f"{float(v):>6.3f}"),
    ("alpha",       "α",         6,  lambda v: "—".rjust(6) if v is None else f"{float(v):>6.3f}"),
]


def _cols_for(env_name: str):
    return _HOTEL_COLS if env_name.lower() == "hotel" else _CTMC_COLS


def format_header(env_name: str) -> str:
    cols = _cols_for(env_name)
    headers = [f"{h:>{w}s}" if i > 1 else (f"{h:>{w}s}" if i == 0 else f"{h:<{w}s}")
               for i, (_, h, w, _) in enumerate(cols)]
    sep_line = " ".join("-" * w for _, _, w, _ in cols)
    return " ".join(headers) + "\n" + sep_line


def format_row(row: Dict[str, Any], env_name: str) -> str:
    cols = _cols_for(env_name)
    parts = []
    for key, _, w, fmt in cols:
        v = row.get(key, None)
        try:
            parts.append(fmt(v))
        except (TypeError, ValueError):
            parts.append("—".rjust(w))
    return " ".join(parts)


def format_baseline_row(
    env_name:   str,
    baselines:  Dict[str, Any],
    label:      str = "[baseline]",
) -> str:
    """One-line compact summary of the reference baselines."""
    if env_name.lower() == "hotel":
        cht  = baselines.get("cht_only",   {}).get("mean_reward", None)
        hind = baselines.get("hindsight",  {}).get("mean_reward", None)
        aa   = baselines.get("accept_all", {}).get("mean_reward", None)
        rr   = baselines.get("reject_all", {}).get("mean_reward", None)
        return (
            f"{label:<20s} "
            f"CHT-only={cht:.1f}  Hindsight={hind:.1f}  "
            f"AcceptAll={aa:.1f}  RejectAll={rr:.1f}"
        )
    else:
        cht  = baselines.get("cht_only",   {}).get("mean_reward", None)
        lp   = baselines.get("lp_bound",   {}).get("mean_reward", None)
        aa   = baselines.get("accept_all", {}).get("mean_reward", None)
        rr   = baselines.get("reject_all", {}).get("mean_reward", None)
        return (
            f"{label:<20s} "
            f"CHT-only={cht:.2f}  LP_UB={lp:.2f}/time  "
            f"AcceptAll={aa:.2f}  RejectAll={rr:.2f}"
        )


# ---------------------------------------------------------------------------
# Per-row construction
# ---------------------------------------------------------------------------

def _build_row_hotel(
    arm: ArmConfig, trial: int, episode: int,
    train_reward: float,
    eval_mean: float, eval_std: float,
    qmag: Dict[str, float],
    epsilon: float, alpha: Optional[float], lr: float, avg_loss: float,
    wall: float,
    baselines: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    cht_ref   = baselines["cht_only"]["mean_reward"]  if baselines else None
    hindsight = baselines["hindsight"]["mean_reward"] if baselines else None
    gap_pct = None
    if hindsight is not None and hindsight > 1e-8:
        gap_pct = 100.0 * (hindsight - eval_mean) / hindsight
    return {
        "arm":             arm.name,
        "trial":           trial,
        "episode":         episode,
        "train_reward":    float(train_reward),
        "eval_reward":     float(eval_mean),
        "eval_reward_std": float(eval_std),
        "cht_ref":         cht_ref,
        "hindsight":       hindsight,
        "gap_pct":         gap_pct,
        "q_theta":         qmag["q_theta"],
        "q_cht":           qmag["q_cht"],
        "q_final":         qmag["q_final"],
        "q_theta_over_final": qmag["q_theta_over_final"],
        "epsilon":         float(epsilon),
        "alpha":           (float(alpha) if alpha is not None else None),
        "lr":              float(lr),
        "avg_loss":        float(avg_loss),
        "wallclock_s":     float(wall),
    }


def _build_row_ctmc(
    arm: ArmConfig, trial: int, episode: int,
    train_reward: float,
    eval_mean: float, eval_std: float,
    eval_rate: float,
    qmag: Dict[str, float],
    epsilon: float, alpha: Optional[float], lr: float, avg_loss: float,
    wall: float,
    baselines: Optional[Dict[str, Any]],
    episode_time: float,
) -> Dict[str, Any]:
    cht_ref = baselines["cht_only"]["mean_reward"] if baselines else None
    lp_rate = baselines["lp_bound"]["mean_reward"] if baselines else None
    # LP bound is a reward-rate; scale to per-episode via the empirical
    # episode time (approximate, but consistent within an episode length).
    lp_total = lp_rate * episode_time if (lp_rate is not None and episode_time > 0) else None
    regret  = (lp_total - eval_mean) if lp_total is not None else None
    gap_pct = None
    if lp_total is not None and lp_total > 1e-8:
        gap_pct = 100.0 * regret / lp_total
    return {
        "arm":             arm.name,
        "trial":           trial,
        "episode":         episode,
        "train_reward":    float(train_reward),
        "eval_reward":     float(eval_mean),
        "eval_reward_std": float(eval_std),
        "eval_rate":       float(eval_rate),
        "cht_ref":         cht_ref,
        "lp_ub":           lp_total,       # LP per-episode expected reward
        "lp_ub_rate":      lp_rate,
        "regret":          regret,
        "gap_pct":         gap_pct,
        "q_theta":         qmag["q_theta"],
        "q_cht":           qmag["q_cht"],
        "q_final":         qmag["q_final"],
        "q_theta_over_final": qmag["q_theta_over_final"],
        "epsilon":         float(epsilon),
        "alpha":           (float(alpha) if alpha is not None else None),
        "lr":              float(lr),
        "avg_loss":        float(avg_loss),
        "wallclock_s":     float(wall),
        "episode_time":    float(episode_time),
    }


def _row_to_trainlog(row: Dict[str, Any]) -> TrainLog:
    return TrainLog(
        arm             = row["arm"],
        trial           = row["trial"],
        episode         = row["episode"],
        epsilon         = row["epsilon"],
        alpha           = (row["alpha"] if row.get("alpha") is not None else 0.0),
        train_reward    = row["train_reward"],
        eval_reward     = row["eval_reward"],
        eval_reward_std = row["eval_reward_std"],
        avg_loss        = row["avg_loss"],
        q_theta_abs     = row["q_theta"],
        q_cht_abs       = row["q_cht"],
        q_final_abs     = row.get("q_final", 0.0),
        q_theta_over_final = row.get("q_theta_over_final", 0.0),
        lr              = row.get("lr", 0.0),
        wallclock_s     = row["wallclock_s"],
        gap_pct         = (row["gap_pct"] if row.get("gap_pct") is not None else 0.0),
        regret          = (row.get("regret") if row.get("regret") is not None else 0.0),
        eval_rate       = row.get("eval_rate", 0.0),
    )


# ---------------------------------------------------------------------------
# Persistent log writer
# ---------------------------------------------------------------------------

class _LogWriter:
    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self.jsonl_path = os.path.join(log_dir, "log.jsonl")
        self.csv_path   = os.path.join(log_dir, "log.csv")
        self._jsonl_fp: Any = open(self.jsonl_path, "w")
        self._csv_writer = None
        self._csv_fp: Any = None
        self._keys: Optional[List[str]] = None

    def write(self, row: Dict[str, Any]) -> None:
        # jsonl
        self._jsonl_fp.write(json.dumps(row, default=float) + "\n")
        self._jsonl_fp.flush()
        # csv
        if self._csv_writer is None:
            self._keys = list(row.keys())
            self._csv_fp = open(self.csv_path, "w", newline="")
            self._csv_writer = csv.DictWriter(
                self._csv_fp, fieldnames=self._keys, extrasaction="ignore"
            )
            self._csv_writer.writeheader()
        self._csv_writer.writerow(row)
        self._csv_fp.flush()

    def close(self) -> None:
        try:
            self._jsonl_fp.close()
        finally:
            if self._csv_fp is not None:
                self._csv_fp.close()


# ---------------------------------------------------------------------------
# One training run (one arm, one trial)
# ---------------------------------------------------------------------------

def train_one_arm(
    arm:             ArmConfig,
    env_factory:     Callable[[int], Any],
    prior_factory:   Callable[[Any, ArmConfig], Any],
    obs_builder_factory: Callable[[Any], Any],
    episode_runner:  Callable[[Any, Any, Any, bool, Optional[int]], Tuple[float, Dict]],
    trainer_cfg:     TrainerConfig,
    trial:           int = 0,
    hp:              Optional[ResidualHP] = None,
    verbose:         bool = False,
    baselines:       Optional[Dict[str, Any]] = None,
    probe_feats:     Optional[np.ndarray] = None,
    probe_infos:     Optional[List[Dict[str, Any]]] = None,
    log_dir:         Optional[str] = None,
    print_header:    bool = True,
) -> Tuple[List[TrainLog], ResidualDQNAgent, List[Dict[str, Any]]]:
    """
    Train one (arm, trial) pair. Deterministic given `trainer_cfg.seed + trial`.

    Returns
    -------
    logs:       back-compat List[TrainLog]
    agent:      final agent
    log_rows:   List[Dict] — richer per-row records for external consumption
    """
    root_seed = trainer_cfg.seed + 100_003 * trial
    set_seeds(root_seed)

    env_seed  = 7 * (trainer_cfg.seed + trial)        # shared across arms
    eval_seed = 99991 + 13 * (trainer_cfg.seed + trial)

    env       = env_factory(env_seed)
    eval_env  = env_factory(eval_seed)
    prior     = prior_factory(env, arm)
    obs_build = obs_builder_factory(env)
    eval_prior= prior_factory(eval_env, arm)
    eval_obs  = obs_builder_factory(eval_env)

    agent = ResidualDQNAgent(
        env          = env,
        obs_builder  = obs_build,
        prior        = prior,
        arm          = arm,
        hp           = hp or ResidualHP(),
        n_episodes   = trainer_cfg.n_episodes,
        device       = trainer_cfg.device,
    )

    # Per-arm writer
    writer: Optional[_LogWriter] = _LogWriter(log_dir) if log_dir else None

    logs:     List[TrainLog]      = []
    rows_all: List[Dict[str, Any]] = []

    if verbose and print_header:
        print(format_header(trainer_cfg.env_name))

    t0 = time.time()
    log_point_counter = 0

    for ep in range(trainer_cfg.n_episodes):
        train_reward, _ = episode_runner(agent, env, prior, False, None)
        agent.end_of_episode()

        if (ep + 1) % trainer_cfg.log_every == 0 or ep == 0:
            # Eval rollouts
            eval_rewards: List[float] = []
            eval_rates:   List[float] = []
            eval_times:   List[float] = []
            for k in range(trainer_cfg.eval_episodes):
                rollout_seed = eval_seed + 13 * k
                r, info_last = episode_runner(agent, eval_env, eval_prior, True, rollout_seed)
                eval_rewards.append(r)
                t_end = float(info_last.get("time", 0.0)) if isinstance(info_last, dict) else 0.0
                eval_times.append(t_end)
                eval_rates.append(r / t_end if t_end > 0 else 0.0)
            eval_mean = float(np.mean(eval_rewards))
            eval_std  = float(np.std(eval_rewards))
            eval_rate = float(np.mean(eval_rates)) if eval_rates else 0.0
            ep_time   = float(np.mean(eval_times)) if eval_times else 0.0

            # Q magnitudes on the shared probe set.
            # `probe_infos` is shared across arms; feats are rebuilt per-arm
            # using the agent's own obs builder so obs_dim always matches.
            if probe_infos is not None and len(probe_infos) > 0:
                rebuilt = np.stack([agent.build_feat(i) for i in probe_infos]).astype(np.float32)
                qmag = agent.q_magnitudes(rebuilt, probe_infos)
            elif probe_feats is not None and probe_feats.shape[0] > 0:
                qmag = agent.q_magnitudes(probe_feats, probe_infos or [])
            else:
                qmag = {"q_theta": 0.0, "q_cht": 0.0, "q_final": 0.0,
                        "q_theta_over_final": 0.0}
                if len(agent.buffer) >= 32:
                    obs, _, _, _, _, infos, _ = agent.buffer.sample(32)
                    qmag = agent.q_magnitudes(obs.cpu().numpy(), infos)

            # LR + schedule read
            cur_lr = float(agent.optimizer.param_groups[0]["lr"])
            avg_loss = (
                float(np.mean(agent._recent_losses[-50:]))
                if agent._recent_losses else 0.0
            )
            alpha_for_row = float(agent.alpha) if arm.use_warm_start else None

            if trainer_cfg.env_name.lower() == "hotel":
                row = _build_row_hotel(
                    arm, trial, ep + 1,
                    train_reward, eval_mean, eval_std,
                    qmag, float(agent.epsilon), alpha_for_row, cur_lr, avg_loss,
                    time.time() - t0, baselines,
                )
            else:
                row = _build_row_ctmc(
                    arm, trial, ep + 1,
                    train_reward, eval_mean, eval_std, eval_rate,
                    qmag, float(agent.epsilon), alpha_for_row, cur_lr, avg_loss,
                    time.time() - t0, baselines, ep_time,
                )

            rows_all.append(row)
            logs.append(_row_to_trainlog(row))
            if writer is not None:
                writer.write(row)

            if verbose:
                print(format_row(row, trainer_cfg.env_name))
                log_point_counter += 1
                if baselines and (log_point_counter % trainer_cfg.baseline_refresh_every == 0):
                    print(format_baseline_row(trainer_cfg.env_name, baselines))

    if writer is not None:
        # Final baseline footer
        writer.close()

    return logs, agent, rows_all


# Kept for back-compat; newer callers use q_magnitudes via the probe set.
def _q_magnitudes(agent, env, prior, obs_builder, n: int = 32) -> Tuple[float, float]:
    if len(agent.buffer) < n:
        return 0.0, 0.0
    obs, actions, _, _, _, infos, _ = agent.buffer.sample(n)
    with torch.no_grad():
        q_theta = agent.online(obs.to(agent.device))
        q_theta_mag = float(q_theta.abs().mean().item())
    q_cht = prior.q_cht_batch(infos)
    q_cht_mag = float(q_cht.abs().mean().item())
    return q_theta_mag, q_cht_mag

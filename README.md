# surya_reusable_research

Research repo exploring reusable-resource allocation with RL. This fork
adds a `residual_rl/` package with a corrected residual-Q-learning
implementation and a clean ablation suite.

## Directory layout

```
.
|-- hotel_env.py        # single-resource finite-horizon hotel RM env
|-- ctmc_env.py         # multi-resource CTMC loss network (Xie et al. 2024, Network 1)
|-- cht_policy.py       # corrected head-count threshold heuristic
|-- cht_dqn.py          # (original, untouched) CHT-DQN reference implementation
|-- dqn_agent.py        # (original, untouched) vanilla DQN reference implementation
|-- compare_rl.py       # (original, untouched) original comparison runner
|-- residual_rl/        # FORK: clean residual-RL package (see below)
|-- experiments/        # FORK: ablation runners and analysis script
|-- tests/              # FORK: pytest suite
|-- results/            # (git-ignored) training logs and plots
```

## Fork notes — `residual_rl/` package

A clean reimplementation of CHT-augmented DQN with three goals:

1. **Correct residual-Q TD target.** The original `cht_dqn.py` uses
   `Q_CHT + Q_theta` in the argmax of `act()` but the plain `Q_theta`
   TD target, which is not residual Q-learning. The new agent uses
   `Q_theta + Q_CHT` *consistently* in both the target argmax and the
   target value, with `Q_theta` regressing toward
   `r + gamma^n * [Q_theta(s', a*) + Q_CHT(s', a*)] - Q_CHT(s, a)`.
   When `q_cht_weight = 0`, this reduces bit-for-bit to Double-DQN.

2. **Single-source ablation config.** `residual_rl.cht_prior.ArmConfig`
   defines four canonical arms using three orthogonal lever flags:

   | Arm | `use_delta_features` | `use_warm_start` | `use_residual_q` |
   |---- |--------------------- |----------------- |----------------- |
   | A — Vanilla DQN                 | False | False | False |
   | B — State-augmented             | True  | False | False |
   | C — + alpha warm-start          | True  | True  | False |
   | D — Full residual RL            | True  | True  | True  |

3. **Both envs under one agent.** `residual_rl.cht_prior` exposes
   `HotelCHTPrior` and `CTMCCHTPrior` with the same interface, so the
   same `ResidualDQNAgent` runs on both `hotel_env.HotelEnv` and
   `ctmc_env.CTMCEnv` without code changes.

### Running the ablation

```bash
# Hotel RM, 4 arms, 3 trials, 3000 eps each
python3 experiments/run_hotel.py --trials 3 --episodes 3000 --arms A,B,C,D

# CTMC env, 4 arms, 3 trials, 500 eps each at N=1
python3 experiments/run_ctmc.py  --trials 3 --episodes 500  --N 1 --arms A,B,C,D

# Analyze
python3 experiments/analyze.py results/<run-dir>
```

A **fast smoke test** to verify the pipeline works end-to-end:

```bash
python3 experiments/run_hotel.py --trials 1 --episodes 200 --arms A,D
python3 experiments/run_ctmc.py  --trials 1 --episodes 200 --N 1 --arms A,D
```

### Running the tests

```bash
python3 -m pytest tests/ -v
```

### Original files preserved

`cht_dqn.py`, `dqn_agent.py`, and `compare_rl.py` are left untouched on
this branch so the differences between the original and the corrected
implementation are readable in git diff. They should be considered the
"before" state: useful as evidence of the bugs, but not used in the
ablation suite.

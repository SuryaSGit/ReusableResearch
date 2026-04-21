"""
residual_rl
===========

Clean reimplementation of the CHT-augmented DQN used in this repo's
`cht_dqn.py`, with the following corrections over the original:

  * Residual Q-learning TD target matches the argmax in `act()`
    (original mismatched; see audit bug B1).
  * Reward-shaping round-trip bug removed (B2).
  * Single `ArmConfig` dataclass drives all four ablation arms:

        A — Vanilla DQN          (no Delta, no warm-start, no residual-Q)
        B — State-augmented      (Delta features only)
        C — + warm-start         (alpha-annealed CHT exploration)
        D — Full residual RL     (Q_CHT in both target *and* argmax)

  * Public training entry point: `residual_rl.trainer.train_one_arm(...)`.
  * Public ablation entry point: `residual_rl.ablation.run_ablation(...)`.

Surya's original files (`cht_dqn.py`, `compare_rl.py`, `dqn_agent.py`)
are left untouched — this package is a fresh, self-contained reimpl.
"""

from residual_rl.cht_prior import (
    ArmConfig,
    ARM_A,
    ARM_B,
    ARM_C,
    ARM_D,
    HotelCHTPrior,
    CTMCCHTPrior,
)
from residual_rl.residual_dqn import ResidualDQNAgent
from residual_rl.trainer import train_one_arm, TrainerConfig, TrainLog
from residual_rl.ablation import run_ablation

__all__ = [
    "ArmConfig",
    "ARM_A",
    "ARM_B",
    "ARM_C",
    "ARM_D",
    "HotelCHTPrior",
    "CTMCCHTPrior",
    "ResidualDQNAgent",
    "train_one_arm",
    "TrainerConfig",
    "TrainLog",
    "run_ablation",
]

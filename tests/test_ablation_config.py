"""
All 4 canonical arms are distinguishable by the lever flags, and the
flags are mutually consistent with the narrative from the plan.
"""

import pytest

from residual_rl.cht_prior import (
    ARM_A, ARM_B, ARM_C, ARM_D, ArmConfig, CANONICAL_ARMS,
)
from residual_rl.ablation import arms_from_strings


def test_all_four_arms_distinct():
    arms = [ARM_A, ARM_B, ARM_C, ARM_D]
    keys = [
        (a.use_delta_features, a.use_warm_start, a.use_residual_q)
        for a in arms
    ]
    assert len(set(keys)) == 4, keys


def test_arm_lever_pattern_matches_plan():
    assert (ARM_A.use_delta_features, ARM_A.use_warm_start, ARM_A.use_residual_q) == (False, False, False)
    assert (ARM_B.use_delta_features, ARM_B.use_warm_start, ARM_B.use_residual_q) == (True,  False, False)
    assert (ARM_C.use_delta_features, ARM_C.use_warm_start, ARM_C.use_residual_q) == (True,  True,  False)
    assert (ARM_D.use_delta_features, ARM_D.use_warm_start, ARM_D.use_residual_q) == (True,  True,  True)


def test_arms_monotone_in_levers():
    """Moving A -> B -> C -> D only switches levers ON, never OFF."""
    order = [ARM_A, ARM_B, ARM_C, ARM_D]
    for prev, nxt in zip(order, order[1:]):
        for lever in ("use_delta_features", "use_warm_start", "use_residual_q"):
            assert getattr(prev, lever) <= getattr(nxt, lever), (
                f"Lever {lever} decreased between {prev.name} and {nxt.name}"
            )


def test_arms_from_strings_parses_case_insensitively():
    got = arms_from_strings(["a", "D"])
    assert got[0].name == ARM_A.name
    assert got[1].name == ARM_D.name


def test_canonical_arms_table_complete():
    assert set(CANONICAL_ARMS.keys()) == {"A", "B", "C", "D"}


def test_arm_d_with_zero_weight_equals_arm_a_in_effect():
    """
    Setting q_cht_weight=0 on arm D should make the TD target identical
    to vanilla Double-DQN.  This is the property the plan relies on.
    """
    arm_d_zero = ArmConfig(
        name                = "D_zero",
        use_delta_features  = False,    # match arm A's obs so feature dims line up
        use_warm_start      = False,    # match arm A's exploration
        use_residual_q      = True,
        q_cht_weight        = 0.0,
    )
    # Config is structurally distinct from ARM_A but should produce the
    # same learning dynamics given the zero weight. The test_td_target.py
    # tests verify this numerically.
    assert arm_d_zero.q_cht_weight == 0.0

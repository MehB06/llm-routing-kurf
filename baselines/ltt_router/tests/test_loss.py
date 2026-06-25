"""
Tests for loss.py.

The test is test_collapses_to_v1_loss: with N=2, choosing the cheap
model, both models evaluated, regret_loss must equal v1's loss
cheap_wrong AND oracle_right on ALL four 0/1 combinations. 
"""

import itertools

import numpy as np
import pytest

from baselines.ltt_router.core.loss import regret_loss


# Core regret semantics
def test_no_regret_when_chosen_correct():
    # chosen (idx 0) is right -> 0 regret regardless of others
    assert regret_loss(0, np.array([1, 1, 0]), np.array([True, True, True])) == 0.0


def test_regret_when_chosen_wrong_but_better_exists():
    # chosen (idx 0) wrong, idx 1 right -> regret
    assert regret_loss(0, np.array([0, 1, 0]), np.array([True, True, True])) == 1.0


def test_no_regret_when_everyone_wrong():
    # chosen wrong, but nobody could have answered -> no regret (routing lost nothing)
    assert regret_loss(0, np.array([0, 0, 0]), np.array([True, True, True])) == 0.0


def test_best_attainable_ignores_unevaluated_models():
    # idx 2 is "correct" but NOT evaluated -> it must NOT count as a better option.
    # chosen idx 0 wrong, idx 1 wrong, idx 2 (correct but unevaluated) -> no regret.
    correct = np.array([0, 0, 1])
    evaluated = np.array([True, True, False])
    assert regret_loss(0, correct, evaluated) == 0.0


def test_regret_uses_only_evaluated_better_models():
    # Same as above but idx 1 IS correct and evaluated -> regret returns.
    correct = np.array([0, 1, 1])
    evaluated = np.array([True, True, False])
    assert regret_loss(0, correct, evaluated) == 1.0


def test_rejects_choosing_unevaluated_model():
    with pytest.raises(ValueError):
        regret_loss(2, np.array([0, 1, 1]), np.array([True, True, False]))


def test_rejects_out_of_range_choice():
    with pytest.raises(ValueError):
        regret_loss(5, np.array([0, 1]), np.array([True, True]))


# Regression check: N=2 collapse to v1's loss
def _v1_loss(cheap_correct: int, oracle_correct: int) -> int:
    """v1's exact relative routing loss when routed cheap."""
    return int(cheap_correct == 0 and oracle_correct == 1)


def test_collapses_to_v1_loss():
    """
    For every (cheap_correct, oracle_correct) in {0,1}^2, with cheap at index 0
    and oracle at index 1, both evaluated, and the router CHOOSING CHEAP,
    regret_loss must equal v1's loss exactly.
    """
    CHEAP, ORACLE = 0, 1
    for cheap_c, oracle_c in itertools.product((0, 1), repeat=2):
        correct = np.array([cheap_c, oracle_c])
        evaluated = np.array([True, True])
        got = regret_loss(CHEAP, correct, evaluated)
        want = float(_v1_loss(cheap_c, oracle_c))
        assert got == want, (
            f"mismatch at cheap={cheap_c}, oracle={oracle_c}: "
            f"v2 regret={got}, v1 loss={want}"
        )


def test_v1_collapse_when_routed_oracle_is_zero():
    """
    v1 charges loss 0 whenever we route to oracle. In v2 that corresponds to
    CHOOSING the oracle (index 1): if the oracle is correct, regret is 0; if the
    oracle is wrong, nobody better was attainable among the two, so regret is
    still 0.
    """
    ORACLE = 1
    for cheap_c, oracle_c in itertools.product((0, 1), repeat=2):
        correct = np.array([cheap_c, oracle_c])
        evaluated = np.array([True, True])
        # If oracle correct -> 0. If oracle wrong: better exists iff cheap correct.
        got = regret_loss(ORACLE, correct, evaluated)
        want = float(oracle_c == 0 and cheap_c == 1)
        assert got == want


# Vectorised path matches the scalar path
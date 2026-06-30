"""
Tests for the LTT calibration.

Three layers of verification:

  1. STATISTICAL PRIMITIVES: the binomial p-value, HB p-value, and FST.

  2. FST DIRECTION is correct; FST must reject a contiguous prefix from the safe 
    high-λ end and stop at the first failure low→high. We construct a sequence 
    with a known answer and assert it.

  3. END-TO-END GUARANTEE on synthetic data with a CONTROLLED true risk. We build
     queries whose score perfectly separates regret from no-regret, so we know
     the ground-truth safe threshold, then check that the certified λ̂ keeps
     realized risk ≤ α, and an impossible α certifies nothing.
"""

import numpy as np
import pytest

from baselines.ltt_router.protocols import QueryRecord
from baselines.ltt_router.core.calibration import (
    binomial_pvalue,
    fixed_sequence_test,
    cheapest_safe_decision_factory,
    build_loss_table,
    calibrate_threshold,
)


# 1. Statistical primitives
def test_binomial_pvalue_basic():
    # 0 failures out of 100 under null risk 0.2 -> very small p-value.
    p = binomial_pvalue(risk_hat=0.0, n=100, alpha=0.2)
    assert 0.0 <= p < 1e-6


def test_binomial_pvalue_at_null_is_large():
    # observed risk == alpha -> not significant -> p-value near (but below) 0.5+.
    p = binomial_pvalue(risk_hat=0.2, n=100, alpha=0.2)
    assert p > 0.4  # CDF at the mean of a binomial is ~0.5


def test_binomial_pvalue_monotone_in_observed_risk():
    # Higher observed risk -> weaker evidence against "unsafe" -> larger p-value.
    alpha, n = 0.2, 200
    ps = [binomial_pvalue(r, n, alpha) for r in (0.0, 0.05, 0.10, 0.15, 0.20)]
    assert all(ps[i] <= ps[i + 1] for i in range(len(ps) - 1))




def test_fst_certifies_contiguous_prefix_from_safe_end():
    # lambdas high->low; p-values: safe (high λ) pass, then fail.
    # ordered high→low
    lambdas = [0.9, 0.8, 0.7, 0.6, 0.5]
    pvals = [0.01, 0.02, 0.30, 0.001, 0.001]
    # Must stop at 0.7 and return the last success = 0.8
    lam_hat = fixed_sequence_test(pvals, lambdas, delta=0.10)
    assert lam_hat == 0.8


def test_fst_returns_none_if_safest_fails():
    lambdas = [0.9, 0.8, 0.7]
    pvals = [0.5, 0.001, 0.001]  # even the safest (high-λ) fails
    assert fixed_sequence_test(pvals, lambdas, delta=0.10) is None


def test_fst_most_permissive_when_all_pass():
    lambdas = [0.9, 0.7, 0.5, 0.3]
    pvals = [0.001, 0.001, 0.001, 0.001]
    # all certified -> return the most permissive (lowest) λ
    assert fixed_sequence_test(pvals, lambdas, delta=0.10) == 0.3


# Synthetic-data helpers for the end-to-end tests
def _make_two_model_queries(n, true_risk, seed=0):
    """
    Build N=2 (cheap=idx0, oracle=idx1) queries where the cheap model's score is
    informative: a `true_risk` fraction of cheap-routed queries are regretful
    (cheap wrong, oracle right), the rest safe. All scores are high (0.95) so the
    cheapest-safe rule routes them all to cheap at low λ.
    """
    rng = np.random.default_rng(seed)
    queries = []
    for _ in range(n):
        regret = rng.random() < true_risk
        if regret:
            cheap_c, oracle_c = 0, 1     # cheap wrong, oracle right -> regret
        else:
            # safe: either cheap right, or both wrong
            if rng.random() < 0.5:
                cheap_c, oracle_c = 1, 1
            else:
                cheap_c, oracle_c = 0, 0
        queries.append(QueryRecord(
            scores=np.array([0.95, 0.0]),       # cheap scores high, oracle never routed-to by score
            correct=np.array([cheap_c, oracle_c]),
            cost=np.array([0.1, 2.0]),
            evaluated=np.array([True, True]),
            dataset_id="synth",
            prompt=f"q{_}",
        ))
    return queries


# build_loss_table + decision rule
def test_decision_factory_picks_cheapest_clearing_lambda():
    decide = cheapest_safe_decision_factory(cost_order=np.array([0, 1]), fallback_idx=1)
    q = QueryRecord(
        scores=np.array([0.4, 0.9]),
        correct=np.array([1, 1]),
        cost=np.array([0.1, 2.0]),
        evaluated=np.array([True, True]),
    )
    # λ=0.5: cheap (0.4) fails, oracle (0.9) clears -> picks oracle (idx1, the fallback too)
    assert decide(q, 0.5) == 1
    # λ=0.3: cheap (0.4) clears -> picks cheap (idx0)
    assert decide(q, 0.3) == 0


def test_decision_factory_skips_unevaluated():
    decide = cheapest_safe_decision_factory(cost_order=np.array([0, 1]), fallback_idx=1)
    q = QueryRecord(
        scores=np.array([0.9, 0.9]),
        correct=np.array([0, 1]),
        cost=np.array([0.1, 2.0]),
        evaluated=np.array([False, True]),   # cheap not evaluated
    )
    # cheap clears λ but wasn't evaluated -> skip to oracle
    assert decide(q, 0.5) == 1


# 3. End-to-end guarantee
def test_calibration_certifies_when_risk_below_alpha():
    # True risk 0.08, target alpha 0.15 -> should certify a λ.
    queries = _make_two_model_queries(n=500, true_risk=0.08, seed=1)
    decide = cheapest_safe_decision_factory(cost_order=np.array([0]), fallback_idx=1)
    res = calibrate_threshold(
        queries, alpha=0.15, decision_fn=decide, fallback_idx=1,
        delta=0.10, min_routed=30,
    )
    assert res.certified
    assert res.lambda_hat is not None
    # realized risk at λ̂ should be ≤ alpha 
    idx = int(np.argmin(np.abs(res.lambdas - res.lambda_hat)))
    assert res.risks[idx] <= res.alpha + 1e-9


def test_calibration_certifies_nothing_for_impossible_alpha():
    # True risk ~0.30 but we demand alpha 0.05 -> nothing should certify.
    queries = _make_two_model_queries(n=500, true_risk=0.30, seed=2)
    decide = cheapest_safe_decision_factory(cost_order=np.array([0]), fallback_idx=1)
    res = calibrate_threshold(
        queries, alpha=0.05, decision_fn=decide, fallback_idx=1,
        delta=0.10, min_routed=30,
    )
    assert not res.certified
    assert res.lambda_hat is None


def test_calibration_result_diagnostics_shapes():
    queries = _make_two_model_queries(n=200, true_risk=0.10, seed=3)
    decide = cheapest_safe_decision_factory(cost_order=np.array([0]), fallback_idx=1)
    res = calibrate_threshold(
        queries, alpha=0.15, decision_fn=decide, fallback_idx=1, n_lambdas=100,
    )
    assert res.lambdas.shape == (100,)
    assert res.risks.shape == (100,)
    assert res.ns.shape == (100,)
    assert res.pvalues.shape == (100,)

# 4. Regression: 
def _graded_score_queries(n, cheap_acc, oracle_acc, seed=0):
    # N=2 queries with a GRADED cheap score (correlated with correctness + noise),
    # so routed count varies smoothly with λ, the real-data shape. At high λ only
    # a few route cheap; at low λ almost all do. Reproduces the bug
    # where a noisy small-n p-value at high λ halts FST and certifies a useless λ̂.
    rng = np.random.default_rng(seed)
    qs = []
    for i in range(n):
        cheap_c = int(rng.random() < cheap_acc)
        oracle_c = int(rng.random() < oracle_acc)
        s_cheap = float(np.clip(rng.normal(0.65 if cheap_c else 0.40, 0.18), 0, 1))
        qs.append(QueryRecord(
            scores=np.array([s_cheap, 0.0]),
            correct=np.array([cheap_c, oracle_c]),
            cost=np.array([0.1, 2.0]),
            evaluated=np.array([True, True]),
            prompt=f"q{i}",
        ))
    return qs


def test_power_floor_prevents_starved_lambda_hat():
    # With the default floor, calibration must NOT certify a λ̂ that
    # routes only a tiny fraction; it should route a large majority while
    # keeping realized risk ≤ α, the behaviour the real benchmark run needs.
    queries = _graded_score_queries(n=2000, cheap_acc=0.78, oracle_acc=0.96, seed=0)
    decide = cheapest_safe_decision_factory(cost_order=np.array([0]), fallback_idx=1)
    res = calibrate_threshold(queries, alpha=0.15, decision_fn=decide, fallback_idx=1)

    assert res.certified, "should certify a usable threshold with adequate power"
    idx = int(np.argmin(np.abs(res.lambdas - res.lambda_hat)))
    routed_frac = res.ns[idx] / len(queries)
    assert routed_frac > 0.5, f"λ̂ routes only {routed_frac:.1%} -- power-starved"
    assert res.risks[idx] <= res.alpha + 1e-9


def test_low_floor_is_unreliable_across_seeds():
    # The OLD low floor (30) is unreliable across seeds it sometimes
    # certifies a low λ̂ (or nothing) because a noisy small-n p-value at high λ
    # halts FST. The default floor is reliably useful on EVERY seed; the low floor
    # is no better at its weakest seed (and usually worse).
    decide = cheapest_safe_decision_factory(cost_order=np.array([0]), fallback_idx=1)

    def routed_frac_at(res, n):
        if not res.certified:
            return 0.0
        idx = int(np.argmin(np.abs(res.lambdas - res.lambda_hat)))
        return res.ns[idx] / n

    default_fracs, low_fracs = [], []
    for seed in range(6):
        qs = _graded_score_queries(n=2000, cheap_acc=0.78, oracle_acc=0.96, seed=seed)
        default_fracs.append(routed_frac_at(
            calibrate_threshold(qs, alpha=0.15, decision_fn=decide, fallback_idx=1), 2000))
        low_fracs.append(routed_frac_at(
            calibrate_threshold(qs, alpha=0.15, decision_fn=decide, fallback_idx=1,
                                min_routed=30), 2000))

    assert min(default_fracs) > 0.5, f"default floor unreliable: {default_fracs}"
    assert min(low_fracs) <= min(default_fracs)
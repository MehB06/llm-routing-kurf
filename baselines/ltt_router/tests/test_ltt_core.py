"""
Tests for the LTT calibration.

Four layers of verification:

  1. STATISTICAL PRIMITIVES: the binomial p-value and FST.

  2. FST DIRECTION is correct; FST must reject a contiguous prefix from the safe
    high-λ end and stop at the first failure low→high. We construct a sequence
    with a known answer and assert it.

  3. END-TO-END GUARANTEE on synthetic data with a CONTROLLED true risk. We build
     queries whose score perfectly separates regret from no-regret, so we know
     the ground-truth safe threshold, then check that the certified λ̂ keeps
     realized risk ≤ α, and an impossible α certifies nothing.

  4. THE PROMISE ITSELF, Monte-Carlo: on a synthetic model where the population
     risk R(λ) has a CLOSED FORM, the fraction of calibration draws whose λ̂ is
     truly unsafe must not exceed δ (up to binomial tolerance). This is the only
     test that checks the statistical guarantee rather than the mechanics, and it
     exercises the min_routed eligibility filter under the same conditions the
     FWER argument covers.
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
    p = binomial_pvalue(n_fail=0, n=100, alpha=0.2)
    assert 0.0 <= p < 1e-6


def test_binomial_pvalue_at_null_is_large():
    # observed failures at the null rate -> not significant -> p-value near 0.5.
    p = binomial_pvalue(n_fail=20, n=100, alpha=0.2)
    assert p > 0.4  # CDF at the mean of a binomial is ~0.5


def test_binomial_pvalue_monotone_in_observed_failures():
    # More observed failures -> weaker evidence against "unsafe" -> larger p-value.
    alpha, n = 0.2, 200
    ps = [binomial_pvalue(k, n, alpha) for k in (0, 10, 20, 30, 40)]
    assert all(ps[i] <= ps[i + 1] for i in range(len(ps) - 1))


def test_binomial_pvalue_empty_denominator_is_uninformative():
    # n = 0 routed queries carries no evidence; p-value must be 1 (never rejects).
    assert binomial_pvalue(n_fail=0, n=0, alpha=0.2) == 1.0




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


def test_build_loss_table_counts_match_risks():
    # n_fails must be the exact integer numerator of each risk entry.
    queries = _make_two_model_queries(n=300, true_risk=0.2, seed=7)
    decide = cheapest_safe_decision_factory(cost_order=np.array([0]), fallback_idx=1)
    lambdas = np.linspace(0.0, 1.0, 11)
    risks, ns, n_fails = build_loss_table(queries, lambdas, decide, fallback_idx=1)
    for i in range(len(lambdas)):
        if ns[i] > 0:
            assert risks[i] == pytest.approx(n_fails[i] / ns[i])
        else:
            assert n_fails[i] == 0 and risks[i] == 0.0


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


# 5. The promise itself: empirical FWER over calibration draws
def _closed_form_risk_queries(n, rng):
    """
    N=2 queries where the population risk has a CLOSED FORM.

    Cheap model (idx 0): score s ~ Uniform(0, 1); correct with probability
    exactly s (a perfectly calibrated scorer). Fallback (idx 1): always correct.
    Under the cheapest-safe rule with cost_order=[0], a query is actively routed
    iff s ≥ λ, and regret = cheap wrong (the fallback is always right), so

        R(λ) = E[1 − s | s ≥ λ] = (1 − λ) / 2.

    Hence λ̂ is TRULY safe iff λ̂ ≥ 1 − 2α — checkable without any test set.
    """
    qs = []
    for i in range(n):
        s = rng.random()
        cheap_c = int(rng.random() < s)
        qs.append(QueryRecord(
            scores=np.array([s, 0.0]),
            correct=np.array([cheap_c, 1]),
            cost=np.array([0.1, 2.0]),
            evaluated=np.array([True, True]),
            prompt=f"q{i}",
        ))
    return qs


def test_fwer_guarantee_holds_empirically():
    """
    Monte-Carlo check of the LTT promise: over repeated calibration draws,
    P(true risk of λ̂ > α) ≤ δ. Uses the closed-form model above so each trial's
    λ̂ can be judged truly-safe/truly-unsafe exactly, with no test-set noise.
    Abstention (λ̂ = None) is never a violation. Runs with the min_routed filter
    active, so the FWER argument for the eligibility filter is exercised too.

    Deliberately the slowest test in the suite (~10–20s): it is the only one
    that checks the statistical guarantee rather than the mechanics.
    """
    alpha, delta = 0.15, 0.10
    lam_safe = 1.0 - 2.0 * alpha          # λ̂ ≥ 0.7 is truly safe
    n_trials, n_calib = 120, 600
    decide = cheapest_safe_decision_factory(cost_order=np.array([0]), fallback_idx=1)
    rng = np.random.default_rng(0)

    violations, certified = 0, 0
    for _ in range(n_trials):
        qs = _closed_form_risk_queries(n_calib, rng)
        res = calibrate_threshold(
            qs, alpha=alpha, decision_fn=decide, fallback_idx=1,
            delta=delta, n_lambdas=21, min_routed=60,
        )
        if res.certified:
            certified += 1
            if res.lambda_hat < lam_safe - 1e-12:
                violations += 1

    # Non-vacuous: the test must not pass by constant abstention.
    assert certified / n_trials > 0.5, f"only {certified}/{n_trials} trials certified"

    # The guarantee, with 3σ binomial tolerance on the Monte-Carlo estimate.
    tol = 3.0 * (delta * (1.0 - delta) / n_trials) ** 0.5
    rate = violations / n_trials
    assert rate <= delta + tol, (
        f"empirical FWER {rate:.3f} exceeds δ={delta} + tolerance {tol:.3f}"
    )
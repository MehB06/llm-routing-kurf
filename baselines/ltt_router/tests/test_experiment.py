"""
Regression tests for the repeated-trials guarantee validation.

  1. pooled_true_risk recovers the true risk (low variance over many trials).
  2. corrected_violation_rate (CI-corrected, denominator-aware) stays ≤ δ.
  3. raw_violation_rate is NOT a valid check (it can be ~50% even when safe) —
     documented here so nobody mistakes it for the guarantee again.
"""

import numpy as np

from baselines.ltt_router.core.calibration import binomial_pvalue
from baselines.ltt_router.experiment import (
    TrialOutcome,
    pooled_true_risk,
    corrected_violation_rate,
    raw_violation_rate,
)


def _simulate_trials(true_risk, n_routed, n_trials, alpha, delta, seed=0):
    """
    One synthetic trial = certify a threshold on a calib draw, then measure a
    fresh test draw. Only CERTIFIED trials are kept (mirrors the real harness,
    which only plots trials whose λ̂ certified). Returns a list of TrialOutcome
    built from the TEST draw.
    """
    rng = np.random.default_rng(seed)
    outcomes = []
    for _ in range(n_trials):
        # calibration draw -> certify via the exact binomial test
        fails_cal = rng.binomial(n_routed, true_risk)
        certified = binomial_pvalue(fails_cal / n_routed, n_routed, alpha) <= delta
        if not certified:
            continue
        # independent test draw at the SAME true risk
        fails_test = int(rng.binomial(n_routed, true_risk))
        outcomes.append(TrialOutcome(
            realized_risk=fails_test / n_routed,
            routed_fraction=1.0,
            certified=True,
            cost_saved=0.0,
            n_routed=n_routed,
            n_routed_fail=fails_test,
        ))
    return outcomes


def test_pooled_true_risk_recovers_population_risk():
    true_risk, alpha, delta = 0.13, 0.15, 0.10
    outcomes = _simulate_trials(true_risk, n_routed=300, n_trials=2000,
                                alpha=alpha, delta=delta, seed=7)
    assert len(outcomes) > 50, "expected many certified trials at risk below alpha"
    # Pooled estimate should be close to the true risk (large pooled n).
    assert abs(pooled_true_risk(outcomes) - true_risk) < 0.01


def test_corrected_violation_stays_within_delta():
    # The hardest honest case: true risk sitting right AT alpha. The guarantee
    # bounds true risk ≤ alpha; the CI-corrected violation rate must stay ≤ δ.
    alpha, delta = 0.15, 0.10
    outcomes = _simulate_trials(true_risk=alpha, n_routed=300, n_trials=3000,
                                alpha=alpha, delta=delta, seed=11)
    assert len(outcomes) > 30
    cv = corrected_violation_rate(outcomes, alpha, delta)
    # allow a small Monte-Carlo cushion above delta
    assert cv <= delta + 0.03, f"corrected violation {cv:.3f} exceeds delta={delta}"


def test_guarantee_holds_against_true_risk_when_below_alpha():
    # When true risk is safely below alpha, true-risk violations must be ~0.
    alpha, delta = 0.15, 0.10
    outcomes = _simulate_trials(true_risk=0.10, n_routed=300, n_trials=2000,
                                alpha=alpha, delta=delta, seed=3)
    assert pooled_true_risk(outcomes) <= alpha
    assert corrected_violation_rate(outcomes, alpha, delta) <= delta + 0.02


def test_raw_rate_is_not_the_guarantee():
    # Documentation test: the RAW finite-sample rate can be far above delta even
    # though the rule is safe — this is exactly the artifact the P0 fix removes.
    # The corrected (significance-based) rate stays controlled at the boundary.
    alpha, delta = 0.15, 0.10
    outcomes = _simulate_trials(true_risk=alpha, n_routed=300, n_trials=3000,
                                alpha=alpha, delta=delta, seed=5)
    raw = raw_violation_rate(outcomes, alpha)
    corrected = corrected_violation_rate(outcomes, alpha, delta)
    # raw is inflated near the boundary; corrected is controlled near delta.
    assert raw > corrected
    assert corrected <= delta + 0.03
"""
LTT calibration.

Training-free: nothing here is fitted by gradient descent. It runs a hypothesis
test on held-out calibration data and certifies a threshold whose POPULATION risk
is ≤ α with probability ≥ 1 − δ over the calibration draw. (The test set is only
a diagnostic; a finite test split may sit above α by sampling noise without
violating the guarantee.)

We calibrate a SINGLE SCALAR threshold λ, applied uniformly: a query may route to
model i iff that model's score clears λ (no per-model threshold vector).

Fixed Sequence Testing (FST) walks λ in a fixed order from the SAFE end (high λ)
toward the PERMISSIVE end (low λ), rejecting "λ is unsafe" while p ≤ δ and
stopping at the first failure; λ̂ is the last (most permissive) λ it rejected.
FST controls the family-wise error over this sequence REGARDLESS of whether risk
is monotone in λ — which need not hold for a conditional (selective-routing) risk
whose denominator changes with λ.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np
from scipy import stats

from baselines.ltt_router.protocols import QueryRecord
from baselines.ltt_router.core.loss import regret_loss


# Minimum routed-query count for a λ to be eligible for FST. Below this the
# binomial p-value is too noisy: a small-n fluke at the SAFE (high-λ) end can end
# the chain early and certify nothing. It is data-independent (n depends on
# scores, not outcomes), so filtering on it preserves FWER ≤ δ.
MIN_ROUTED_DEFAULT = 200


def binomial_pvalue(risk_hat: float, n: int, alpha: float) -> float:
    """
    Exact binomial p-value for the null H: true risk ≥ alpha.

    With a 0/1 loss, the number of failures among n routed queries is
    Binomial(n, true_risk). Under the null (true_risk = alpha), the probability
    of seeing this few failures or fewer is the binomial CDF at the observed
    failure count.
    """
    n_fail = int(round(risk_hat * n))
    return float(stats.binom.cdf(n_fail, n, alpha))



# Fixed Sequence Testing
def fixed_sequence_test(
    pvalues: Sequence[float],
    lambdas: Sequence[float],
    delta: float,
) -> Optional[float]:
    """
    Fixed Sequence Testing (FST) for family-wise error control.

    The caller passes (pvalues, lambdas) and this walks them ordered from the
    SAFE end (high λ, low risk) toward the PERMISSIVE end (low λ, high risk). We
    reject the null "λ is unsafe" while p ≤ delta and STOP at the first λ we fail
    to reject. λ̂ is the LAST λ we successfully rejected; the most permissive
    (most cost-saving) certified threshold.

    Returns λ̂, or None if even the safest hypothesis fails to reject.
    """
    lam_hat: Optional[float] = None
    for p, lam in sorted(zip(pvalues, lambdas), key=lambda t: -t[1]):  # high → low
        if p <= delta:
            lam_hat = lam
        else:
            break
    return lam_hat


# Decision function: how a candidate λ turns one query into a routing choice
DecisionFn = Callable[[QueryRecord, float], int]


def cheapest_safe_decision_factory(
    cost_order: np.ndarray,
    fallback_idx: int,
) -> DecisionFn:
    """
    Build the cost-ordered cheapest-safe decision rule used for calibration.

    cost_order:
        int[K] model indices sorted cheapest → most-expensive. 
        These are the models the rule is allowed to pick.
    fallback_idx:
        Index of the most capable survivor, used when no candidate clears λ.

    Returns
    DecisionFn: (query, lambda) -> chosen_idx. The first model in cost_order that
        was evaluated and whose score ≥ λ; else the fallback.

    Sparse-data fallback: benchmark data is not dense, so the designated fallback
    may have no row for a query (its outcome is unknown, regret unscorable). We
    then take the most-capable model that WAS evaluated, walking cost_order from
    the expensive end. QueryRecord guarantees ≥1 evaluated model, so this always
    resolves.
    """
    cost_order = np.asarray(cost_order, dtype=int)

    def decide(q: QueryRecord, lam: float) -> int:
        for idx in cost_order:
            if q.evaluated[idx] and q.scores[idx] >= lam:
                return int(idx)
        # No candidate cleared λ. Prefer the designated fallback if it was
        # evaluated; otherwise the most-capable evaluated survivor (most expensive -> cheap),
        # and finally any evaluated model at all.
        if q.evaluated[fallback_idx]:
            return int(fallback_idx)
        for idx in cost_order[::-1]:
            if q.evaluated[idx]:
                return int(idx)
        return int(np.flatnonzero(q.evaluated)[0])

    return decide


def is_active_route(q: QueryRecord, lam: float, cost_order: np.ndarray) -> bool:
    """
    True iff some model actually CLEARED λ for this query (an active cheap route),
    vs a deferral to the capable fallback. Used for the risk denominator: the
    guarantee bounds regret AMONG actively-routed queries. This is robust to
    sparse data.
    """
    cost_order = np.asarray(cost_order, dtype=int)
    for idx in cost_order:
        if q.evaluated[idx] and q.scores[idx] >= lam:
            return True
    return False


# Loss table over the λ grid
def build_loss_table(
    queries: List[QueryRecord],
    lambdas: np.ndarray,
    decision_fn: DecisionFn,
    fallback_idx: int,
    cost_order: Optional[np.ndarray] = None,
):
    """
    Build per-λ empirical risk and the routed count, for the rule.

    For each λ and each query we apply decision_fn(query, λ) to get the chosen
    model, then score it with the regret loss. "Routed" means the rule actively
    chose a cheaper model (something CLEARED λ) rather than deferring to the
    capable fallback. Risk is the mean regret AMONG those actively routed queries,
    which is the quantity the guarantee bounds.

    cost_order is used to classify active-route vs deferral robustly (via
    is_active_route). If omitted we fall back to the chosen==fallback_idx test
    (valid only on dense data where the fallback is always evaluated).

    Returns
    risks : float[L]   mean regret among actively-routed queries, per λ
    ns    : int[L]     number of actively-routed queries, per λ
    """
    L = len(lambdas)
    risks = np.zeros(L)
    ns = np.zeros(L, dtype=int)

    for i, lam in enumerate(lambdas):
        losses = []
        for q in queries:
            if cost_order is not None:
                active = is_active_route(q, lam, cost_order)
            else:
                active = decision_fn(q, lam) != fallback_idx
            if not active:
                # Deferred to the capable fallback: loss 0 by construction, and it
                # does NOT count toward the certified risk denominator.
                continue
            chosen = decision_fn(q, lam)
            losses.append(regret_loss(chosen, q.correct, q.evaluated))
        ns[i] = len(losses)
        risks[i] = float(np.mean(losses)) if losses else 0.0

    return risks, ns


# Top-level calibration
@dataclass
class CalibrationResult:
    """Outcome of one LTT calibration run, plus diagnostics for plotting."""
    lambda_hat: Optional[float]      # certified threshold, or None if nothing certifies
    alpha: float
    delta: float
    lambdas: np.ndarray
    risks: np.ndarray
    ns: np.ndarray
    pvalues: np.ndarray
    min_routed: int
    certified: bool = field(init=False)

    def __post_init__(self) -> None:
        self.certified = self.lambda_hat is not None


def calibrate_threshold(
    queries: List[QueryRecord],
    alpha: float,
    decision_fn: DecisionFn,
    fallback_idx: int,
    delta: float = 0.10,
    n_lambdas: int = 100,
    min_routed: int = MIN_ROUTED_DEFAULT,
    cost_order: Optional[np.ndarray] = None,
) -> CalibrationResult:
    """
    Run LTT calibration: find the most permissive λ̂ whose true regret under the
    routing rule is provably ≤ alpha with probability ≥ 1 − delta.

    queries:
        Calibration QueryRecord's. MUST be disjoint from the scorer's training
        data and from the test set, or the guarantee breaks (the three-way split
        in the adaptor enforces this).
    alpha:
        Risk target. regret ≤ alpha.
    decision_fn:
        The routing rule to certify.
    fallback_idx:
        The safe-default model index (queries routed here are excluded from the
        certified risk denominator).
    delta:
        Failure probability of the guarantee.
    n_lambdas:
        Resolution of the λ grid over [0, 1].
    min_routed:
        Minimum actively-routed count for a λ to be ELIGIBLE for FST. Ineligible
        λ are FILTERED OUT of the sequence.

    Returns
    CalibrationResult
    """
    lambdas = np.linspace(0.0, 1.0, n_lambdas)
    risks, ns = build_loss_table(queries, lambdas, decision_fn, fallback_idx, cost_order)

    # p-value per λ for the null (exact binomial, UMP for the 0/1 loss).
    pvals = np.ones(len(lambdas))
    for i in range(len(lambdas)):
        if ns[i] == 0:
            pvals[i] = 1.0
            continue
        pvals[i] = binomial_pvalue(risks[i], ns[i], alpha)

    # Build the fixed sequence from the SAFE end (high λ) toward permissive (low
    # λ), over ELIGIBLE thresholds only (n ≥ min_routed).
    order = np.argsort(lambdas)[::-1]  # high → low
    seq_p: List[float] = []
    seq_lam: List[float] = []
    for i in order:
        if ns[i] < min_routed:
            continue
        seq_p.append(float(pvals[i]))
        seq_lam.append(float(lambdas[i]))

    lam_hat = fixed_sequence_test(seq_p, seq_lam, delta)

    return CalibrationResult(
        lambda_hat=lam_hat,
        alpha=alpha,
        delta=delta,
        lambdas=lambdas,
        risks=risks,
        ns=ns,
        pvalues=pvals,
        min_routed=min_routed,
    )
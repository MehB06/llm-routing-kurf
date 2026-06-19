"""
LTT core: the statistical machinery that turns calibration data into a
provably-safe routing threshold λ̂.

For a candidate threshold λ, we route a calibration query to cheap iff
score ≥ λ. Each routed-cheap query contributes loss 1 if (cheap wrong AND oracle
right), else 0. The empirical risk at λ is the mean of that loss over routed
queries. LTT asks: is the TRUE risk provably ≤ α? It answers with a p-value, and
FST rejects "λ is unsafe" for as many λ as it safely can. λ̂ = the most permissive
(lowest, i.e. most queries routed cheap) threshold that survives.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from scipy import stats


def binomial_pvalue(risk_hat: float, n: int, alpha: float) -> float:
    """
    Exact binomial p-value for the null H: true risk >= alpha.

    With a 0/1 loss, the number of failures among n routed queries is
    Binomial(n, true_risk). Under the null (true_risk = alpha), the probability
    of seeing this few failures or fewer is the binomial CDF at the observed
    failure count..

    This is the uniformly most powerful test for a Bernoulli loss, which is why
    it's preferred over HB when the loss is exactly 0/1.
    """
    n_fail = int(round(risk_hat * n))
    # P(X <= n_fail) where X ~ Binomial(n, alpha)
    return stats.binom.cdf(n_fail, n, alpha)


def hoeffding_bentkus_pvalue(risk_hat: float, n: int, alpha: float) -> float:
    """
    Hoeffding-Bentkus p-value for H: true risk >= alpha. Distribution-free. 
    We keep it as a cross-check against the binomial p-value.

    Returns min(Hoeffding bound, Bentkus bound).
    """
    if risk_hat >= alpha:
        return 1.0
    # Hoeffding bound
    h = np.exp(-n * _kl_bernoulli(risk_hat, alpha))
    # Bentkus bound
    b = np.e * stats.binom.cdf(int(np.ceil(n * risk_hat)), n, alpha)
    return float(min(h, b, 1.0))


def _kl_bernoulli(p: float, q: float) -> float:
    """KL divergence between Bernoulli(p) and Bernoulli(q), used by Hoeffding."""
    eps = 1e-12
    p = min(max(p, eps), 1 - eps)
    q = min(max(q, eps), 1 - eps)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def fixed_sequence_test(
    pvalues: List[float],
    lambdas: List[float],
    delta: float,
) -> Optional[float]:
    """
    Fixed Sequence Testing (FST) for family-wise error control.

    The caller passes (pvalues, lambdas) ALREADY ORDERED from the safe end
    (high λ, low risk) toward the permissive end (low λ, high risk). We walk
    that fixed order and reject the null "λ is unsafe" while p-value <= delta,
    STOPPING at the first λ we fail to reject. λ̂ is the LAST λ we successfully
    rejected, the most permissive (most cost-saving) certified threshold.

    Why this ordering and stopping rule give a valid guarantee: empirical RISK
    is monotone decreasing in λ (a higher bar for cheap routing means only
    confident queries go cheap), so the certified region is a contiguous PREFIX
    of this safe-to-permissive sequence. FST only tests hypothesis k+1 if it
    rejected 1..k; that prefix structure controls family-wise error with no
    multiplicity penalty across the ~100 candidate λ.

    Returns λ̂ (most permissive certified threshold), or None if even the first
    (safest) hypothesis fails to reject.
    """
    # Sequence is pre-ordered safe -> permissive; reject while p <= delta,
    # break at the first failure; lam_hat is the last successful rejection.
    lam_hat = None
    for p, lam in sorted(zip(pvalues, lambdas), key=lambda t: -t[1]):
        if p <= delta:
            lam_hat = lam      
        else:
            break            
    return lam_hat


@dataclass
class CalibQuery:
    """One calibration query, aligned across the cheap/oracle pair."""
    score: float        # scorer output: P(cheap succeeds)
    cheap_correct: int  # did cheap actually succeed? 0/1
    oracle_correct: int # did oracle actually succeed? 0/1


def routing_loss(q: CalibQuery, routed_cheap: bool) -> int:
    """
    RELATIVE routing loss .

    Failure iff we routed to cheap AND cheap was wrong where oracle was right.
    If routed to oracle: loss 0 (we deferred to the strong model).
    If both were wrong anyway: loss 0 (routing cheap lost us nothing).
    """
    if not routed_cheap:
        return 0
    return int(q.cheap_correct == 0 and q.oracle_correct == 1)


def build_loss_table(queries: List[CalibQuery], lambdas: np.ndarray) -> np.ndarray:
    """
    Build the (n_lambdas,) array of EMPIRICAL RISK per candidate threshold.

    For each λ: route query to cheap iff score >= λ. Risk = mean routing loss
    over the queries that were routed to cheap. (Queries routed to oracle carry
    loss 0 by definition and don't contribute failures, but they DO matter for
    how many queries we're averaging over)

    NOTE ON DENOMINATOR: we define risk as failures / (routed-cheap count), i.e.
    the failure rate AMONG cheap-routed queries. This is the conditional risk
    "given we sent it cheap, how often did we regret it." Returns risk per λ and
    the routed-cheap count per λ (n for the p-value).
    """
    scores = np.array([q.score for q in queries])
    risks = np.empty(len(lambdas))
    ns = np.empty(len(lambdas), dtype=int)
    for i, lam in enumerate(lambdas):
        routed = scores >= lam
        n_routed = int(routed.sum())
        ns[i] = n_routed
        if n_routed == 0:
            risks[i] = 0.0
            continue
        losses = np.array([
            routing_loss(q, True) for q, r in zip(queries, routed) if r
        ])
        risks[i] = losses.mean()
    return risks, ns


def calibrate_threshold(
    queries: List[CalibQuery],
    alpha: float,
    delta: float = 0.1,
    n_lambdas: int = 100,
    pvalue: str = "binomial",
) -> dict:
    """
    Run LTT calibration: find the most permissive λ̂ whose true relative risk is
    provably <= alpha with probability >= 1 - delta.

    We evaluate risk and a p-value on a grid of λ in [0, 1]. Risk falls as λ
    rises (a higher bar routes only confident queries cheap), so safety is
    MONOTONE in λ. We hand FST the eligible λ (n >= MIN_ROUTED) ordered from the
    safe end (high λ) toward the permissive end (low λ); it rejects "λ unsafe"
    along that prefix and stops at the first failure. λ̂ is the lowest (most
    cost-saving) λ in the certified prefix.

    Returns dict with lambda_hat, and diagnostics.
    """
    lambdas = np.linspace(0.0, 1.0, n_lambdas)
    risks, ns = build_loss_table(queries, lambdas)

    # p-value per λ for null "true risk >= alpha"
    pvals = np.ones(len(lambdas))
    for i in range(len(lambdas)):
        if ns[i] == 0:
            pvals[i] = 1.0  # no data routed cheap => can't certify
            continue
        if pvalue == "binomial":
            pvals[i] = binomial_pvalue(risks[i], ns[i], alpha)
        else:
            pvals[i] = hoeffding_bentkus_pvalue(risks[i], ns[i], alpha)


    MIN_ROUTED = 30  

    # Build the fixed sequence FROM THE SAFE END (high lambda) toward permissive
    # (low lambda), over ELIGIBLE thresholds only (n >= MIN_ROUTED). Filtering
    # first (rather than poisoning ineligible lambdas with p=1.0) is essential:
    # the high-lambda end has tiny n, so a poisoned p there would break the FST
    # chain at its very first step and certify nothing.
    order = np.argsort(lambdas)[::-1]  # high -> low (safe -> permissive)
    seq_p, seq_lam = [], []
    for i in order:
        if ns[i] < MIN_ROUTED:
            continue  # ineligible: not part of the tested family at all
        seq_p.append(pvals[i])
        seq_lam.append(lambdas[i])

    lam_hat = fixed_sequence_test(seq_p, seq_lam, delta)

    return {
        "lambda_hat": lam_hat,
        "alpha": alpha,
        "delta": delta,
        "lambdas": lambdas,
        "risks": risks,
        "ns": ns,
        "pvalues": pvals,
        "pvalue_method": pvalue,
    }
"""
The routing rule 

1. PARETO PRE-FILTER (offline, train/calib only). We drop any model that is dominated 
   some other model is both cheaper AND at least as accurate across the data.
   A dominated model can never be the right routing choice, so removing it shrinks
   the action set without losing anything. 

2. COST ORDERING. Sort the Pareto survivors cheapest to most expensive. 

3. CHEAPEST-SAFE ROUTING. For an incoming query, walk cheapest-first and
   take the first whose score clears the calibrated λ̂ (and which is evaluated);
   if none clear, fall back to the most capable model (the safe default).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from baselines.ltt_router.protocols import QueryRecord, RoutingFunction, ModelSpec
from baselines.ltt_router.calibration import (
    cheapest_safe_decision_factory,
    calibrate_threshold,
    CalibrationResult,
    DecisionFn,
)

# 1. Pareto pre-filter
def model_accuracies(queries: List[QueryRecord], n_models: int) -> np.ndarray:
    """
    Mean accuracy of each model over the queries on which it was EVALUATED.
    """
    sums = np.zeros(n_models)
    counts = np.zeros(n_models)
    for q in queries:
        ev = q.evaluated
        sums[ev] += q.correct[ev]
        counts[ev] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        acc = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    return acc


def pareto_survivors(
    costs: np.ndarray,
    accuracies: np.ndarray,
    tol: float = 1e-9,
) -> np.ndarray:
    """
    Indices of models on the cost/accuracy Pareto frontier.

    Model j DOMINATES model i iff j is no more expensive AND at least as accurate,
    with at least one of those strict (so j is genuinely a better-or-equal deal).
    A model is a survivor iff nobody dominates it.

    costs, accuracies:
        float[N] aligned by model index.
    tol:
        Numerical tolerance for the "at least as accurate / no more expensive"
        comparisons.

    Returns
        Sorted int[K] indices of surviving (non-dominated) models.
    """
    n = costs.shape[0]
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            cheaper_or_equal = costs[j] <= costs[i] + tol
            better_or_equal = accuracies[j] >= accuracies[i] - tol
            strictly_better = (costs[j] < costs[i] - tol) or (accuracies[j] > accuracies[i] + tol)
            if cheaper_or_equal and better_or_equal and strictly_better:
                dominated[i] = True
                break
    return np.flatnonzero(~dominated)


def cost_ordered(survivors: np.ndarray, costs: np.ndarray) -> np.ndarray:
    """Sort survivor indices cheapest to most expensive (ties broken by index)."""
    survivors = np.asarray(survivors, dtype=int)
    if survivors.size == 0:
        return survivors
    # Primary key: cost; tie-break: the index itself (both ascending).
    order = np.lexsort((survivors, costs[survivors]))
    return survivors[order]


def most_capable(survivors: np.ndarray, accuracies: np.ndarray) -> int:
    """
    The safe-default fallback: the most accurate surviving model. Used when no
    cheaper model clears the threshold.
    """
    survivors = np.asarray(survivors, dtype=int)
    return int(survivors[int(np.argmax(accuracies[survivors]))])


# 2. The Router object
@dataclass
class RouterPlan:
    """
    The fitted, calibrated routing policy: everything needed to route a new
    query, plus the diagnostics behind it.

    This is produced by Router.fit and consumed by Router.route. It is the
    serialisable summary of "what did calibration decide".
    """
    models: List[ModelSpec]
    survivors: np.ndarray           # Pareto-surviving model indices
    cost_order: np.ndarray          # survivors sorted cheapest to most expensive
    fallback_idx: int               # most-capable survivor
    calibration: CalibrationResult
    accuracies: np.ndarray          # per-model accuracy on calib (diagnostic)

    @property
    def lambda_hat(self) -> Optional[float]:
        return self.calibration.lambda_hat

    @property
    def certified(self) -> bool:
        return self.calibration.certified


class Router:
    """
    The public N-model, risk-controlled router.

    Wraps an injected RoutingFunction with the Pareto filter, cost-ordering, and LTT calibration. 
    The scorer is the ONLY trained component — the rest is deterministic and risk-controlled.

        router = Router(scorer)                      # scorer: RoutingFunction
        plan = router.fit(calib_queries, alpha=0.15) # Pareto + cost-order + LTT
        choice = router.route(new_query)             # cheapest-safe decision

    Scorer must already be trained on a DISJOINT split before calibration
    (the adaptor enforces train / calib / test disjointness).
    """

    def __init__(self, scorer: RoutingFunction):
        self.scorer = scorer
        self.plan: Optional[RouterPlan] = None

    def fit(
        self,
        calib_queries: List[QueryRecord],
        alpha: float,
        delta: float = 0.10,
        apply_pareto: bool = True,
        n_lambdas: int = 100,
        pvalue: str = "binomial",
        min_routed: int = 30,
    ) -> RouterPlan:
        """
        Pareto-filter (on the calibration data), cost-order, then run LTT to
        certify λ̂ for the cheapest-safe rule over the survivors.

        apply_pareto=False gives the budget-only ablation (cost-order over ALL
        models, no domination filter)
        """
        if not calib_queries:
            raise ValueError("no calibration queries provided")
        n_models = calib_queries[0].n_models

        accuracies = model_accuracies(calib_queries, n_models)
        costs = self._model_costs(n_models)

        if apply_pareto:
            survivors = pareto_survivors(costs, accuracies)
        else:
            survivors = np.arange(n_models)

        order = cost_ordered(survivors, costs)
        fallback_idx = most_capable(survivors, accuracies)

        decision_fn = cheapest_safe_decision_factory(order, fallback_idx)
        calib = calibrate_threshold(
            calib_queries,
            alpha=alpha,
            decision_fn=decision_fn,
            fallback_idx=fallback_idx,
            delta=delta,
            n_lambdas=n_lambdas,
            pvalue=pvalue,
            min_routed=min_routed,
        )

        self.plan = RouterPlan(
            models=list(self.scorer.models),
            survivors=survivors,
            cost_order=order,
            fallback_idx=fallback_idx,
            calibration=calib,
            accuracies=accuracies,
        )
        return self.plan

    def route(self, query: QueryRecord) -> int:
        """
        Route one query to a model index using the calibrated λ̂.

        If calibration certified nothing (λ̂ is None), we conservatively route to
        the fallback (most capable) model — never silently route cheap.
        """
        if self.plan is None:
            raise RuntimeError("Router.fit must be called before route")
        lam = self.plan.lambda_hat
        if lam is None:
            return self.plan.fallback_idx
        decide = cheapest_safe_decision_factory(self.plan.cost_order, self.plan.fallback_idx)
        return decide(query, lam)

    def route_batch(self, queries: List[QueryRecord]) -> np.ndarray:
        """Vectorised convenience: ``int[M]`` chosen index per query."""
        return np.array([self.route(q) for q in queries], dtype=int)

    def _model_costs(self, n_models: int) -> np.ndarray:
        costs = np.zeros(n_models)
        for m in self.scorer.models:
            costs[m.index] = m.cost
        return costs
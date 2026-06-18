"""
LTT Router: the end-to-end pipeline.

Ties together the three components built:
  scorer  ->  LTT calibration  ->  routing decisions + metrics

  - AvgAcc   : accuracy of the routed system (per-query: cheap or oracle result)
  - CostSave : fraction of cost saved vs. always-oracle
  - routed_cheap_frac : how much traffic went to the cheap model
  - relative_risk : measured regret rate (cheap wrong where oracle right) among
                    cheap-routed queries
  - PERFORMANCE-COST CURVE: sweep alpha to trace the whole tradeoff frontier

  - always-oracle : max accuracy, zero savings  (CostSave = 0)
  - always-cheap  : min cost, lowest accuracy   (CostSave = max)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional

from baselines.ltt_v1.ltt_core import CalibQuery, calibrate_threshold, routing_loss


CHEAP_DEFAULT = "qwen3-235b-a22b-2507"
ORACLE_DEFAULT = "gpt-5"


@dataclass
class RoutedQuery:
    """A query aligned across the cheap/oracle pair, with cost info."""
    score: float
    cheap_correct: int
    oracle_correct: int
    cheap_cost: float
    oracle_cost: float


def pivot_records(records, scorer, cheap_model=CHEAP_DEFAULT,
                  oracle_model=ORACLE_DEFAULT, scores=None) -> List[RoutedQuery]:
    """
    Reshape flat records into per-query RoutedQuery objects for the model pair.

    Aligns cheap and oracle rows by (dataset_id, record_index). Queries missing
    either model are skipped.
    """
    cheap, oracle = {}, {}
    for r in records:
        key = (r.dataset_id, r.record_index)
        if r.model_name == cheap_model:
            cheap[key] = r
        elif r.model_name == oracle_model:
            oracle[key] = r

    keys = sorted(set(cheap) & set(oracle))
    prompts = [cheap[k].prompt for k in keys]
    if scores is None:
        scores = scorer.predict_proba(prompts, show_progress=False)

    out = []
    for i, k in enumerate(keys):
        out.append(RoutedQuery(
            score=float(scores[i]),
            cheap_correct=int(cheap[k].score == 1.0),
            oracle_correct=int(oracle[k].score == 1.0),
            cheap_cost=float(cheap[k].cost),
            oracle_cost=float(oracle[k].cost),
        ))
    return out


def compute_routing_metrics(queries: List[RoutedQuery], lam: Optional[float]) -> Dict:
    """
    Apply threshold lam and compute the full metric set.

    Routing rule: route to cheap iff score >= lam. If lam is None (no certified
    threshold), route everything to oracle (the safe fallback).
    """
    n = len(queries)
    if lam is None:
        # No safe threshold => behave as always-oracle.
        acc = np.mean([q.oracle_correct for q in queries])
        cost = np.mean([q.oracle_cost for q in queries])
        oracle_cost = cost
        return {
            "lambda": None, "avg_acc": acc, "avg_cost": cost,
            "cost_save": 0.0, "routed_cheap_frac": 0.0,
            "relative_risk": 0.0, "n_routed_cheap": 0, "n": n,
        }

    routed_cheap = np.array([q.score >= lam for q in queries])

    # Per-query realised correctness and cost under the routing decision
    correct, cost = [], []
    regrets, n_cheap = 0, 0
    for q, rc in zip(queries, routed_cheap):
        if rc:
            correct.append(q.cheap_correct)
            cost.append(q.cheap_cost)
            n_cheap += 1
            regrets += routing_loss(
                CalibQuery(q.score, q.cheap_correct, q.oracle_correct), True)
        else:
            correct.append(q.oracle_correct)
            cost.append(q.oracle_cost)

    avg_acc = float(np.mean(correct))
    avg_cost = float(np.mean(cost))
    oracle_only_cost = float(np.mean([q.oracle_cost for q in queries]))
    cost_save = 1.0 - (avg_cost / oracle_only_cost) if oracle_only_cost > 0 else 0.0
    rel_risk = (regrets / n_cheap) if n_cheap > 0 else 0.0

    return {
        "lambda": lam,
        "avg_acc": avg_acc,
        "avg_cost": avg_cost,
        "cost_save": cost_save,
        "routed_cheap_frac": n_cheap / n,
        "relative_risk": rel_risk,
        "n_routed_cheap": n_cheap,
        "n": n,
    }


def reference_bounds(queries: List[RoutedQuery]) -> Dict:
    """The two trivial bounds every result is framed against."""
    always_oracle_acc = float(np.mean([q.oracle_correct for q in queries]))
    always_cheap_acc = float(np.mean([q.cheap_correct for q in queries]))
    oracle_cost = float(np.mean([q.oracle_cost for q in queries]))
    cheap_cost = float(np.mean([q.cheap_cost for q in queries]))
    return {
        "always_oracle": {"avg_acc": always_oracle_acc, "avg_cost": oracle_cost,
                          "cost_save": 0.0},
        "always_cheap": {"avg_acc": always_cheap_acc, "avg_cost": cheap_cost,
                         "cost_save": 1.0 - cheap_cost / oracle_cost
                                       if oracle_cost > 0 else 0.0},
    }


def run_router(calib_records, test_records, scorer,
               alpha: float, delta: float = 0.10,
               cheap_model=CHEAP_DEFAULT, oracle_model=ORACLE_DEFAULT) -> Dict:
    """
    Full pipeline for one alpha: calibrate λ̂ on calib, evaluate on test.
    """
    # Calibrate on calibration split
    calib_q = pivot_records(calib_records, scorer, cheap_model, oracle_model)
    calib_for_ltt = [CalibQuery(q.score, q.cheap_correct, q.oracle_correct)
                     for q in calib_q]
    res = calibrate_threshold(calib_for_ltt, alpha=alpha, delta=delta,
                              pvalue="binomial")
    lam_hat = res["lambda_hat"]

    # Evaluate on test split
    test_q = pivot_records(test_records, scorer, cheap_model, oracle_model)
    metrics = compute_routing_metrics(test_q, lam_hat)
    metrics["alpha"] = alpha
    metrics["delta"] = delta
    return metrics


def sweep_alpha(calib_records, test_records, scorer,
                alphas=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
                delta: float = 0.10,
                cheap_model=CHEAP_DEFAULT, oracle_model=ORACLE_DEFAULT) -> List[Dict]:
    """
    Trace the performance-cost frontier by sweeping alpha. This is the
    experiment: each alpha is one certified operating point.
    """
    results = []
    for a in alphas:
        results.append(run_router(calib_records, test_records, scorer,
                                   alpha=a, delta=delta,
                                   cheap_model=cheap_model, oracle_model=oracle_model))
    return results
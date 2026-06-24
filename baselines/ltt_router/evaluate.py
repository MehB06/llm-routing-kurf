"""
Metrics.

Two blocks, computed on the SAME held-out test QueryRecords so everything is
directly comparable:

  BENCHMARK-COMPARABLE:
    - avg_accuracy:   did the chosen model actually succeed, per query
    - avg_cost:       mean cost of the chosen model
    - cost_saved:     1 - avg_cost / all_oracle_cost   (vs always using the fallback)
    - routing_dist:   fraction of traffic sent to each model

  GUARANTEE (only our baseline reports this):
    - alpha, delta, lambda_hat, certified
    - realized_risk:  regret on the TEST split under the routed decisions
    - per_model_traffic, survivors, routed_fraction

The guarantee is the column the standard table has no slot for: same routing
decision, but with a certified bound on realized regret.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from baselines.ltt_router.protocols import QueryRecord, ModelSpec
from baselines.ltt_router.loss import regret_loss


# Reference rows (benchmark definitions, computed on OUR test set)
def oracle_accuracy(queries: List[QueryRecord]) -> float:
    # Oracle: a query is "correct" if ANY evaluated model got it right (upper bound).
    correct = [bool(np.any((q.correct == 1) & q.evaluated)) for q in queries]
    return float(np.mean(correct)) if correct else 0.0


def per_model_accuracy(queries: List[QueryRecord], n_models: int) -> np.ndarray:
    # Mean accuracy of each model over the queries where it was evaluated.
    sums = np.zeros(n_models)
    counts = np.zeros(n_models)
    for q in queries:
        ev = q.evaluated
        sums[ev] += q.correct[ev]
        counts[ev] += 1
    return np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)


def per_model_cost(queries: List[QueryRecord], n_models: int) -> np.ndarray:
    # Mean cost of each model over the queries where it was evaluated.
    sums = np.zeros(n_models)
    counts = np.zeros(n_models)
    for q in queries:
        ev = q.evaluated
        sums[ev] += q.cost[ev]
        counts[ev] += 1
    return np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)


def max_expert(queries: List[QueryRecord], n_models: int) -> Dict[str, float]:
    # Max Expert: the single best-accuracy model; report its accuracy and cost.
    acc = per_model_accuracy(queries, n_models)
    cost = per_model_cost(queries, n_models)
    best = int(np.argmax(acc))
    return {"index": best, "accuracy": float(acc[best]), "cost": float(cost[best])}


def fixed_model_row(queries: List[QueryRecord], idx: int) -> Dict[str, float]:
    # Accuracy + cost if we ALWAYS used model `idx` (e.g. all-cheap, all-oracle).
    # Only counts queries where that model was evaluated.
    accs, costs = [], []
    for q in queries:
        if q.evaluated[idx]:
            accs.append(int(q.correct[idx] == 1))
            costs.append(float(q.cost[idx]))
    return {
        "accuracy": float(np.mean(accs)) if accs else 0.0,
        "cost": float(np.mean(costs)) if costs else 0.0,
    }


def random_router_row(queries: List[QueryRecord], seed: int = 42) -> Dict[str, float]:
    # Random Router: pick a uniformly random EVALUATED model per query.
    rng = np.random.default_rng(seed)
    accs, costs = [], []
    for q in queries:
        evald = np.flatnonzero(q.evaluated)
        idx = int(rng.choice(evald))
        accs.append(int(q.correct[idx] == 1))
        costs.append(float(q.cost[idx]))
    return {
        "accuracy": float(np.mean(accs)) if accs else 0.0,
        "cost": float(np.mean(costs)) if costs else 0.0,
    }


@dataclass
class EvalResult:
    # benchmark-comparable
    avg_accuracy: float
    avg_cost: float
    cost_saved: float
    routing_dist: Dict[str, float]
    # reference rows (same test set)
    reference: Dict[str, Dict[str, float]]
    # guarantee block
    alpha: float
    delta: float
    lambda_hat: Optional[float]
    certified: bool
    realized_risk: float          # conditional: regret among actively-routed (matches λ̂)
    realized_risk_all: float      # unconditional: regret over all queries (diagnostic)
    routed_fraction: float
    per_model_traffic: Dict[str, float]
    survivors: List[str]

    def comparable_table_row(self) -> Dict[str, float]:
        # The row we hand to the benchmark's comparison table.
        return {
            "method": "LTT-Router",
            "accuracy": round(self.avg_accuracy, 4),
            "cost": round(self.avg_cost, 4),
            "cost_saved": round(self.cost_saved, 4),
            "realized_risk": round(self.realized_risk, 4),
            "alpha": self.alpha,
            "certified": self.certified,
        }


def evaluate(result, fallback_for_savings: Optional[int] = None) -> EvalResult:
    """
    Compute both metric blocks from an AdaptorResult.

    fallback_for_savings: model index that "cost saved" is measured against
        (default: the plan's fallback = the safe capable model)
    """
    queries = result.test_queries
    chosen = result.chosen_indices
    models = result.models
    n_models = len(models)
    plan = result.plan
    names = [m.name for m in models]

    # Chosen-model accuracy + cost per query, plus per-query active-route status.
    # The GUARANTEE is conditional: regret ≤ α AMONG actively-routed queries
    # So realized_risk must use the SAME denominator.
    from baselines.ltt_router.calibration import is_active_route
    lam = plan.lambda_hat if plan.lambda_hat is not None else float("inf")

    accs, costs = [], []
    routed_regrets, all_regrets = [], []
    traffic = np.zeros(n_models)
    for q, c in zip(queries, chosen):
        # chosen model is guaranteed evaluated (router only picks evaluated/fallback)
        accs.append(int(q.correct[c] == 1))
        costs.append(float(q.cost[c]))
        r = regret_loss(int(c), q.correct, q.evaluated)
        all_regrets.append(r)
        if is_active_route(q, lam, plan.cost_order):
            routed_regrets.append(r)        # only these are under the guarantee
        traffic[c] += 1

    avg_acc = float(np.mean(accs)) if accs else 0.0
    avg_cost = float(np.mean(costs)) if costs else 0.0
    # realized_risk = conditional regret among actively-routed queries (matches λ̂).
    realized_risk = float(np.mean(routed_regrets)) if routed_regrets else 0.0
    # unconditional regret over ALL queries (incl. deferrals) — diagnostic only.
    realized_risk_all = float(np.mean(all_regrets)) if all_regrets else 0.0

    fb = plan.fallback_idx if fallback_for_savings is None else fallback_for_savings
    all_fallback = fixed_model_row(queries, fb)
    cost_saved = (1.0 - avg_cost / all_fallback["cost"]) if all_fallback["cost"] > 0 else 0.0

    traffic_frac = traffic / max(1, len(queries))
    routing_dist = {names[i]: float(traffic_frac[i]) for i in range(n_models) if traffic_frac[i] > 0}
    routed_fraction = (
        float(np.mean([is_active_route(q, lam, plan.cost_order) for q in queries]))
        if queries else 0.0
    )

    # Reference rows on the same test set.
    me = max_expert(queries, n_models)
    reference = {
        "Oracle":        {"accuracy": oracle_accuracy(queries), "cost": np.nan},
        "Max Expert":    {"accuracy": me["accuracy"], "cost": me["cost"],
                          "model": names[me["index"]]},
        "All-Cheap":     fixed_model_row(queries, int(np.argmin([m.cost for m in models]))),
        "All-Fallback":  all_fallback,
        "Random Router": random_router_row(queries, seed=result.seed),
    }

    return EvalResult(
        avg_accuracy=avg_acc,
        avg_cost=avg_cost,
        cost_saved=cost_saved,
        routing_dist=routing_dist,
        reference=reference,
        alpha=result.alpha,
        delta=result.delta,
        lambda_hat=plan.lambda_hat,
        certified=plan.certified,
        realized_risk=realized_risk,
        realized_risk_all=realized_risk_all,
        routed_fraction=routed_fraction,
        per_model_traffic=routing_dist,
        survivors=[names[i] for i in plan.survivors],
    )


def format_report(ev: EvalResult) -> str:
    """Human-readable summary (printed by experiment.py / the adaptor)."""
    lines = []
    lines.append("=" * 60)
    lines.append("BENCHMARK-COMPARABLE")
    lines.append(f"  accuracy:    {ev.avg_accuracy:.4f}")
    lines.append(f"  avg cost:    {ev.avg_cost:.4f}")
    lines.append(f"  cost saved:  {ev.cost_saved:.2%} (vs All-Fallback)")
    lines.append(f"  routed:      {ev.routed_fraction:.2%} to a non-fallback model")
    lines.append("  reference rows (same test set):")
    for name, m in ev.reference.items():
        c = "  n/a" if (isinstance(m.get("cost"), float) and np.isnan(m.get("cost", np.nan))) else f"{m.get('cost', float('nan')):.4f}"
        lines.append(f"    {name:14s} acc={m['accuracy']:.4f}  cost={c}")
    lines.append("-" * 60)
    lines.append("GUARANTEE (our addition)")
    lines.append(f"  alpha={ev.alpha}  delta={ev.delta}  certified={ev.certified}")
    lines.append(f"  lambda_hat:    {ev.lambda_hat}")
    lines.append(f"  realized risk: {ev.realized_risk:.4f}  (target ≤ {ev.alpha})")
    lines.append(f"  survivors:     {ev.survivors}")
    lines.append(f"  traffic:       {', '.join(f'{k}={v:.1%}' for k, v in ev.routing_dist.items())}")
    lines.append("=" * 60)
    return "\n".join(lines)


def load_baseline_results(repo_root: str = ".") -> Dict[str, Dict[str, float]]:
    """
    Read the existing baselines' result JSONs (where present) into a uniform
    {method: {accuracy, cost}} dict for the comparison figure.
    """
    import glob
    import json
    import os

    out: Dict[str, Dict[str, float]] = {}
    rl = sorted(glob.glob(os.path.join(repo_root, "baselines/RouteLLM/results/mf_results_*.json")))
    if rl:
        accs, costs = [], []
        for path in rl:
            try:
                d = json.load(open(path))
                accs.append(d.get("sample_avg", d.get("selection_accuracy", np.nan)))
                costs.append(d.get("avg_cost", np.nan))
            except Exception:
                continue
        if accs:
            out["RouteLLM"] = {"accuracy": float(np.nanmean(accs)), "cost": float(np.nanmean(costs))}
    return out

# Markdown results table
def alpha_sweep_markdown(results: list) -> str:
    """
    Build the README's α-sweep table from a list of AdaptorResults (one per α).
    realized_risk here is the CONDITIONAL (certified) risk, so the table tests the
    guarantee like-for-like.
    """
    header = ("| α    | certified | λ̂     | routed % | realized risk | cost saved |\n"
              "|------|-----------|-------|----------|---------------|------------|")
    rows = [header]
    for r in results:
        ev = evaluate(r)
        lam = f"{ev.lambda_hat:.3f}" if ev.lambda_hat is not None else "—"
        rows.append(
            f"| {ev.alpha:<4} | {str(ev.certified):<9} | {lam:<5} | "
            f"{ev.routed_fraction*100:6.1f}  | {ev.realized_risk:6.3f}        | "
            f"{ev.cost_saved*100:6.1f}%    |"
        )
    return "\n".join(rows)
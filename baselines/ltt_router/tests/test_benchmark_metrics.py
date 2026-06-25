"""
Tests for the benchmark-comparable metric delegation 
"""

import numpy as np

from baselines.ltt_router.protocols import ModelSpec, QueryRecord
from baselines.ltt_router.benchmark_metrics import (
    emit_baseline_records, benchmark_metrics,
)


def _toy_queries(n=40, seed=0):
    rng = np.random.default_rng(seed)
    models = [ModelSpec(name=f"m{i}", cost=[0.001, 0.01, 0.05][i], index=i)
              for i in range(3)]
    qs = []
    for j in range(n):
        qs.append(QueryRecord(
            scores=rng.random(3),
            correct=rng.integers(0, 2, size=3),
            cost=np.array([0.001, 0.01, 0.05]) + rng.normal(0, 1e-4, 3),
            evaluated=np.array([True, True, True]),
            dataset_id="d",
            prompt=f"p{j}",
        ))
    chosen = np.array([int(np.argmax(q.scores)) for q in qs])
    return qs, chosen, models


def test_emit_one_record_per_query():
    qs, chosen, models = _toy_queries()
    recs = emit_baseline_records(qs, chosen, models)
    assert len(recs) == len(qs)
    # each record carries the CHOSEN model's score and cost
    for r, q, c in zip(recs, qs, chosen):
        assert r.model_name == models[int(c)].name
        assert r.score == float(q.correct[int(c)])
        assert r.cost == float(q.cost[int(c)])


def test_delegated_metrics_match_direct_means():
    qs, chosen, models = _toy_queries(seed=3)
    recs = emit_baseline_records(qs, chosen, models)
    bm = benchmark_metrics(recs)

    direct_acc = np.mean([int(q.correct[int(c)] == 1) for q, c in zip(qs, chosen)])
    direct_cost = np.mean([float(q.cost[int(c)]) for q, c in zip(qs, chosen)])

    assert np.isclose(bm["accuracy"], direct_acc)
    assert np.isclose(bm["avg_cost"], direct_cost)
    assert bm["n"] == len(qs)


def test_empty_is_safe():
    bm = benchmark_metrics([])
    assert bm == {"accuracy": 0.0, "avg_cost": 0.0, "n": 0}
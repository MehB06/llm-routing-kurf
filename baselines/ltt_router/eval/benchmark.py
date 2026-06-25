"""
Benchmark-comparable metrics, computed by the benchmark's OWN aggregator.

This module does not define a metric. It expresses the router's decisions in the
benchmark schema and hands them to baselines.aggregators.BaselineAggregator —
the same aggregation every other baseline's numbers come from. 
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from loguru import logger

from baselines.schema import BaselineRecord
from baselines.aggregators import BaselineAggregator
from baselines.ltt_router.protocols import QueryRecord, ModelSpec

# The aggregator logs an INFO line on every construction; evaluate() builds one
# per trial, so a 500-trial run would print 500+ lines. Silence just that module.
logger.disable("baselines.aggregators")


def emit_baseline_records(
    queries: List[QueryRecord],
    chosen_indices: np.ndarray,
    models: List[ModelSpec],
    split: str = "test",
) -> List[BaselineRecord]:
    """
    One BaselineRecord per routed query, holding the CHOSEN model's outcome.

    This is the router's decision expressed in the benchmark's schema: for each
    test query we record which model the router picked, that model's score
    (correctness) and its cost.
    """
    names = [m.name for m in models]
    records: List[BaselineRecord] = []
    for i, (q, c) in enumerate(zip(queries, chosen_indices)):
        c = int(c)
        records.append(BaselineRecord(
            dataset_id=q.dataset_id,
            split=split,
            model_name=names[c],
            record_index=i,
            origin_query=q.prompt,
            prompt=q.prompt,
            prediction="",
            raw_output=None,
            ground_truth="",
            score=float(q.correct[c]),     # benchmark "score": 1.0 if chosen model correct
            prompt_tokens=0,
            completion_tokens=0,
            cost=float(q.cost[c]),
        ))
    return records


def benchmark_metrics(records: List[BaselineRecord]) -> Dict[str, float]:
    """
    Accuracy and avg cost from routed BaselineRecords, via the benchmark's own
    BaselineAggregator.get_global_stats(). No metric is defined here; we read
    avg_score and avg_cost_per_record out of the benchmark's aggregation.
    """
    if not records:
        return {"accuracy": 0.0, "avg_cost": 0.0, "n": 0}
    stats = BaselineAggregator(records).get_global_stats()
    return {
        "accuracy": float(stats["avg_score"]),
        "avg_cost": float(stats["avg_cost_per_record"]),
        "n": int(stats["total_records"]),
    }
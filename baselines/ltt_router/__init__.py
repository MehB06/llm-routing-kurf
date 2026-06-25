"""
LTT Router — N-model, risk-controlled LLM routing.

Package layout (by pipeline role):
    protocols      contracts: ModelSpec, QueryRecord, RoutingFunction
    core/          the risk-controlled engine: loss, calibration, routing
    routers/       pluggable scorers: embedding_lr (real), random_router (ablation)
    eval/          measurement: metrics + the benchmark bridge
    splitting      three-way train/calib/test split
    experiment     the three figures + repeated-trials harness
"""

from baselines.ltt_router.protocols import (
    RoutingFunction,
    QueryRecord,
    ModelSpec,
)
from baselines.ltt_router.core import (
    regret_loss,
    binomial_pvalue,
    fixed_sequence_test,
    cheapest_safe_decision_factory,
    calibrate_threshold,
    is_active_route,
    CalibrationResult,
    pareto_survivors,
    cost_ordered,
    most_capable,
    model_accuracies,
    Router,
    RouterPlan,
)
from baselines.ltt_router.routers import (
    EmbeddingLRRouter,
    build_embedding_lr_router,
    RandomRouter,
)
from baselines.ltt_router.eval import (
    evaluate,
    EvalResult,
    emit_baseline_records,
    benchmark_metrics,
)

__all__ = [
    # contracts
    "RoutingFunction",
    "QueryRecord",
    "ModelSpec",
    # core: loss / calibration / routing
    "regret_loss",
    "binomial_pvalue",
    "fixed_sequence_test",
    "cheapest_safe_decision_factory",
    "calibrate_threshold",
    "is_active_route",
    "CalibrationResult",
    "pareto_survivors",
    "cost_ordered",
    "most_capable",
    "model_accuracies",
    "Router",
    "RouterPlan",
    # concrete routers
    "EmbeddingLRRouter",
    "build_embedding_lr_router",
    "RandomRouter",
    # eval
    "evaluate",
    "EvalResult",
    "emit_baseline_records",
    "benchmark_metrics",
]
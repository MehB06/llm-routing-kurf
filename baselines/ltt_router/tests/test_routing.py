"""
Tests for routing.py.

  1. PARETO domination correctness 
  2. COST ORDERING + fallback selection.
  3. The Router END-TO-END with a synthetic injected scorer
  4. The budget-only vs +Pareto switch behaves (apply_pareto).
"""

import numpy as np
import pytest

from baselines.ltt_router.protocols import QueryRecord, ModelSpec
from baselines.ltt_router.core.routing import (
    model_accuracies,
    pareto_survivors,
    cost_ordered,
    most_capable,
    Router,
)

# 1. Pareto
def test_pareto_simple_frontier():
    # model 0: cheap & weak, model 1: more expensive & strong -> both survive (trade-off)
    costs = np.array([0.1, 2.0])
    acc = np.array([0.6, 0.9])
    surv = pareto_survivors(costs, acc)
    assert set(surv.tolist()) == {0, 1}


def test_pareto_drops_dominated():
    # model 2 is more expensive than 0 AND less accurate -> dominated by 0, dropped.
    costs = np.array([0.1, 2.0, 0.5])
    acc = np.array([0.6, 0.9, 0.55])
    surv = pareto_survivors(costs, acc)
    assert set(surv.tolist()) == {0, 1}


def test_pareto_drops_equal_accuracy_more_expensive():
    # model 1 same accuracy as 0 but more expensive -> dominated.
    costs = np.array([0.1, 0.5])
    acc = np.array([0.8, 0.8])
    surv = pareto_survivors(costs, acc)
    assert surv.tolist() == [0]


def test_pareto_keeps_cheaper_equal_accuracy():
    # The cheaper of two equally-accurate models survives; the more expensive is dropped.
    costs = np.array([0.5, 0.1])
    acc = np.array([0.8, 0.8])
    surv = pareto_survivors(costs, acc)
    assert surv.tolist() == [1]


def test_pareto_all_survive_when_monotone_tradeoff():
    costs = np.array([0.1, 0.5, 1.0, 2.0])
    acc = np.array([0.5, 0.6, 0.7, 0.8])   # strictly increasing cost & acc
    surv = pareto_survivors(costs, acc)
    assert surv.tolist() == [0, 1, 2, 3]

# 2. Cost ordering + fallback
def test_cost_ordered_sorts_cheapest_first():
    costs = np.array([2.0, 0.1, 0.5])
    surv = np.array([0, 1, 2])
    assert cost_ordered(surv, costs).tolist() == [1, 2, 0]


def test_most_capable_picks_highest_accuracy():
    acc = np.array([0.6, 0.95, 0.7])
    surv = np.array([0, 1, 2])
    assert most_capable(surv, acc) == 1


def test_model_accuracies_respects_evaluated_mask():
    # model 1 evaluated once (correct), model 0 evaluated twice (1 correct).
    queries = [
        QueryRecord(np.array([0.5, 0.5]), np.array([1, 0]),
                    np.array([1.0, 1.0]), np.array([True, False])),
        QueryRecord(np.array([0.5, 0.5]), np.array([0, 1]),
                    np.array([1.0, 1.0]), np.array([True, True])),
    ]
    acc = model_accuracies(queries, n_models=2)
    assert acc[0] == pytest.approx(0.5)    # 1 of 2
    assert acc[1] == pytest.approx(1.0)    # 1 of 1 (the masked one ignored)

# 3. Router end-to-end with a synthetic scorer
class ScriptedScorer:
    """A RoutingFunction whose scores come from the query itself, 
       so we control routing exactly in tests."""

    def __init__(self, models):
        self._models = models

    @property
    def models(self):
        return self._models

    def score_batch(self, prompts):
        # Not used by these tests (queries carry their own scores), but present
        # to satisfy the protocol.
        return np.zeros((len(prompts), len(self._models)))


def _two_model_calib(n, true_risk, seed=0):
    rng = np.random.default_rng(seed)
    qs = []
    for i in range(n):
        regret = rng.random() < true_risk
        if regret:
            cc, oc = 0, 1
        else:
            cc, oc = (1, 1) if rng.random() < 0.5 else (0, 0)
        qs.append(QueryRecord(
            scores=np.array([0.95, 0.0]),
            correct=np.array([cc, oc]),
            cost=np.array([0.1, 2.0]),
            evaluated=np.array([True, True]),
            prompt=f"q{i}",
        ))
    return qs


def test_router_fit_certifies_and_routes_cheap():
    models = [ModelSpec("cheap", 0.1, 0), ModelSpec("oracle", 2.0, 1)]
    router = Router(ScriptedScorer(models))
    plan = router.fit(_two_model_calib(500, true_risk=0.08, seed=1), alpha=0.15)
    assert plan.certified
    # a high-scoring cheap query should route cheap (idx 0)
    q = QueryRecord(np.array([0.99, 0.0]), np.array([1, 1]),
                    np.array([0.1, 2.0]), np.array([True, True]))
    assert router.route(q) == 0


def test_router_falls_back_when_nothing_certified():
    models = [ModelSpec("cheap", 0.1, 0), ModelSpec("oracle", 2.0, 1)]
    router = Router(ScriptedScorer(models))
    # impossible target: true risk ~0.3, alpha 0.05 -> certifies nothing
    plan = router.fit(_two_model_calib(500, true_risk=0.30, seed=2), alpha=0.05)
    assert not plan.certified
    q = QueryRecord(np.array([0.99, 0.0]), np.array([0, 1]),
                    np.array([0.1, 2.0]), np.array([True, True]))
    # must route to fallback (oracle, idx 1)
    assert router.route(q) == plan.fallback_idx == 1


def test_router_requires_fit_before_route():
    models = [ModelSpec("cheap", 0.1, 0), ModelSpec("oracle", 2.0, 1)]
    router = Router(ScriptedScorer(models))
    q = QueryRecord(np.array([0.9, 0.0]), np.array([1, 1]),
                    np.array([0.1, 2.0]), np.array([True, True]))
    with pytest.raises(RuntimeError):
        router.route(q)


# 4. Pareto vs budget-only switch
def test_pareto_flag_changes_survivor_set():
    # Three models where the mid one is dominated; with Pareto it's dropped,
    # without Pareto all three are kept in the cost order.
    models = [
        ModelSpec("cheap", 0.1, 0),
        ModelSpec("dominated", 0.5, 1),
        ModelSpec("oracle", 2.0, 2),
    ]
    rng = np.random.default_rng(0)
    qs = []
    for i in range(300):
        # cheap acc ~0.7, dominated acc ~0.6 (worse & more expensive than cheap), oracle ~0.95
        cc = int(rng.random() < 0.70)
        dc = int(rng.random() < 0.60)
        oc = int(rng.random() < 0.95)
        qs.append(QueryRecord(
            scores=np.array([0.95, 0.95, 0.0]),
            correct=np.array([cc, dc, oc]),
            cost=np.array([0.1, 0.5, 2.0]),
            evaluated=np.array([True, True, True]),
            prompt=f"q{i}",
        ))
    router = Router(ScriptedScorer(models))
    plan_pareto = router.fit(qs, alpha=0.2, apply_pareto=True)
    plan_budget = router.fit(qs, alpha=0.2, apply_pareto=False)
    assert 1 not in plan_pareto.survivors.tolist()       # dominated dropped
    assert 1 in plan_budget.survivors.tolist()           # kept without Pareto

def test_route_handles_unevaluated_fallback():
    # 3 models; fallback is the most expensive (idx 2). On a query where idx2 has NO row,
    # and nothing clears λ, the rule must pick the most-capable EVALUATED model
    # (idx 1 here), never return the unevaluated fallback (which would be unscorable).
    from baselines.ltt_router.core.calibration import cheapest_safe_decision_factory
    decide = cheapest_safe_decision_factory(cost_order=np.array([0, 1, 2]), fallback_idx=2)
    q = QueryRecord(
        scores=np.array([0.1, 0.1, 0.1]),       # nothing clears a high λ
        correct=np.array([0, 1, 0]),
        cost=np.array([0.1, 0.5, 2.0]),
        evaluated=np.array([True, True, False]),  # fallback idx2 NOT evaluated
    )
    chosen = decide(q, lam=0.9)
    assert q.evaluated[chosen], "must choose an evaluated model"
    assert chosen == 1, "should pick the most-capable evaluated model in cost order"


def test_router_route_never_returns_unevaluated_model():
    # End-to-end: even when the plan's fallback is unevaluated on a test query,
    # Router.route returns an evaluated index (regret would be unscorable otherwise).
    from baselines.ltt_router.protocols import ModelSpec

    class ScriptedScorer:
        def __init__(self, models): self._models = models
        @property
        def models(self): return self._models
        def score_batch(self, prompts): return np.zeros((len(prompts), len(self._models)))

    models = [ModelSpec("cheap", 0.1, 0), ModelSpec("mid", 0.5, 1), ModelSpec("oracle", 2.0, 2)]
    rng = np.random.default_rng(0)
    calib = []
    for i in range(400):
        calib.append(QueryRecord(
            scores=np.array([0.9, 0.5, 0.0]),
            correct=np.array([int(rng.random() < 0.8), int(rng.random() < 0.85),
                              int(rng.random() < 0.95)]),
            cost=np.array([0.1, 0.5, 2.0]),
            evaluated=np.array([True, True, True]),
            prompt=f"q{i}",
        ))
    router = Router(ScriptedScorer(models))
    router.fit(calib, alpha=0.25)
    # a test query where the oracle (likely fallback) is NOT evaluated
    q = QueryRecord(np.array([0.1, 0.1, 0.1]), np.array([1, 1, 0]),
                    np.array([0.1, 0.5, 2.0]), np.array([True, True, False]))
    chosen = router.route(q)
    assert q.evaluated[chosen]

# 5. Active-route semantics (reviewer #4): a deferral to fallback (nothing cleared
#    λ) must NOT count as an active route; a model clearing λ IS an active route,
#    even if it is the most-capable/fallback model. This documents the intended
#    denominator and guards against the proposed (incorrect) candidate_order fix.
def test_deferral_not_counted_as_active_route():
    from baselines.ltt_router.core.calibration import is_active_route
    cost_order = np.array([0, 1])   # 0 cheap, 1 capable/fallback
    # No model's score clears a high λ -> this is a deferral, not an active route.
    q = QueryRecord(
        scores=np.array([0.2, 0.3]),
        correct=np.array([0, 1]),
        cost=np.array([0.1, 2.0]),
        evaluated=np.array([True, True]),
    )
    assert is_active_route(q, lam=0.9, cost_order=cost_order) is False


def test_model_clearing_lambda_is_active_route():
    from baselines.ltt_router.core.calibration import is_active_route
    cost_order = np.array([0, 1])
    # The cheap model clears λ -> active route.
    q = QueryRecord(
        scores=np.array([0.95, 0.0]),
        correct=np.array([1, 1]),
        cost=np.array([0.1, 2.0]),
        evaluated=np.array([True, True]),
    )
    assert is_active_route(q, lam=0.9, cost_order=cost_order) is True


def test_fit_uses_design_queries_for_action_set():
    # Design accuracies must come from design_queries, not calib. We make the two
    # splits disagree on which model looks best and check the fallback follows the
    # DESIGN split.
    models = [ModelSpec("cheap", 0.1, 0), ModelSpec("oracle", 2.0, 1)]
    router = Router(ScriptedScorer(models))

    def q(cheap_ok, oracle_ok):
        return QueryRecord(np.array([0.95, 0.0]), np.array([cheap_ok, oracle_ok]),
                           np.array([0.1, 2.0]), np.array([True, True]))

    # Design: oracle clearly most accurate -> fallback should be oracle (idx 1).
    design = [q(0, 1) for _ in range(200)]
    calib = [q(1, 1) for _ in range(300)]
    plan = router.fit(calib, alpha=0.2, design_queries=design, apply_pareto=False)
    assert plan.fallback_idx == 1
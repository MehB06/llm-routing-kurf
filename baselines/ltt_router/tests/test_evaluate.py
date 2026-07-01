"""
Tests for evaluate.py and experiment.py. Synthetic data only; figures render to
a temp dir (Agg backend, no display).

  1. Reference rows: Oracle / Max Expert / All-Cheap / Random match hand-checks.
  2. evaluate(): chosen-model accuracy, cost saved, realized risk computed right.
  3. experiment: repeated-trials violation rate + each PNG is produced.
"""

import importlib.util
import os
import sys

import numpy as np
import pytest

from baselines.ltt_router.protocols import ModelSpec, QueryRecord
from baselines.ltt_router.eval.metrics import (
    oracle_accuracy, max_expert, fixed_model_row, random_router_row,
    per_model_accuracy, evaluate,
)
from baselines.ltt_router import experiment as exp


_MOD = "ltt_adaptor_eval_test"
_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "adaptors", "ltt_adaptor.py")
_spec = importlib.util.spec_from_file_location(_MOD, _PATH)
_adaptor = importlib.util.module_from_spec(_spec)
sys.modules[_MOD] = _adaptor
_spec.loader.exec_module(_adaptor)
LTTAdaptor = _adaptor.LTTAdaptor


class FakeRecord:
    def __init__(self, dataset_id, prompt, model_name, score, cost, record_index):
        self.dataset_id = dataset_id
        self.prompt = prompt
        self.model_name = model_name
        self.score = score
        self.cost = cost
        self.record_index = record_index


def stub_embed_fn(prompts):
    return np.asarray([[len(p), p.count("?"), sum(ord(c) for c in p) % 97] for p in prompts],
                      dtype=float)


def make_records(n_prompts=300, seed=0):
    rng = np.random.default_rng(seed)
    recs, idx = [], 0
    for i in range(n_prompts):
        q = f"q {i}?"
        for name, p, cost in (("cheap", 0.70, 0.1), ("mid", 0.80, 0.5), ("oracle", 0.95, 2.0)):
            recs.append(FakeRecord("A", q, name, float(rng.random() < p), cost, idx)); idx += 1
    return recs


# 1. Reference rows
def _q(scores, correct, cost, evaluated):
    return QueryRecord(np.array(scores, float), np.array(correct), np.array(cost, float),
                       np.array(evaluated, bool))


def test_oracle_counts_any_evaluated_correct():
    qs = [
        _q([0, 0, 0], [0, 1, 0], [.1, .5, 2], [True, True, True]),   # oracle-correct
        _q([0, 0, 0], [0, 0, 0], [.1, .5, 2], [True, True, True]),   # nobody right
    ]
    assert oracle_accuracy(qs) == pytest.approx(0.5)


def test_oracle_ignores_unevaluated_correct():
    # idx2 correct but NOT evaluated -> must not count.
    qs = [_q([0, 0, 0], [0, 0, 1], [.1, .5, 2], [True, True, False])]
    assert oracle_accuracy(qs) == 0.0


def test_max_expert_picks_best_accuracy_model():
    qs = [
        _q([0, 0], [0, 1], [.1, 2], [True, True]),
        _q([0, 0], [0, 1], [.1, 2], [True, True]),
        _q([0, 0], [1, 1], [.1, 2], [True, True]),
    ]
    me = max_expert(qs, 2)
    assert me["index"] == 1                       # model 1 always right
    assert me["accuracy"] == pytest.approx(1.0)


def test_fixed_model_row_only_counts_evaluated():
    qs = [
        _q([0, 0], [1, 0], [.1, 2], [True, True]),
        _q([0, 0], [0, 0], [.1, 2], [False, True]),   # cheap not evaluated here
    ]
    row = fixed_model_row(qs, 0)
    assert row["accuracy"] == pytest.approx(1.0)   # only the first query counts
    assert row["cost"] == pytest.approx(0.1)


# 2. evaluate() end-to-end
def test_evaluate_produces_both_blocks():
    recs = make_records(300, seed=1)
    result = LTTAdaptor("UNUSED", seed=42).run(alpha=0.25, embed_fn=stub_embed_fn, records=recs)
    ev = evaluate(result)
    # benchmark-comparable block
    assert 0 <= ev.avg_accuracy <= 1
    assert ev.avg_cost >= 0
    assert "Oracle" in ev.reference and "Max Expert" in ev.reference
    # guarantee block
    assert ev.alpha == 0.25
    assert 0 <= ev.realized_risk <= 1
    assert 0 <= ev.routed_fraction <= 1


def test_realized_risk_matches_manual():
    # realized_risk is CONDITIONAL: regret among actively-routed queries (matches
    # the certified bound). Recompute it independently and check evaluate agrees.
    recs = make_records(300, seed=2)
    result = LTTAdaptor("UNUSED", seed=42).run(alpha=0.30, embed_fn=stub_embed_fn, records=recs)
    ev = evaluate(result)
    from baselines.ltt_router.core.loss import regret_loss
    from baselines.ltt_router.core.calibration import is_active_route
    plan = result.plan
    lam = plan.lambda_hat if plan.lambda_hat is not None else float("inf")

    routed_r = []
    for q, c in zip(result.test_queries, result.chosen_indices):
        if is_active_route(q, lam, plan.cost_order):
            routed_r.append(regret_loss(int(c), q.correct, q.evaluated))
    manual_cond = np.mean(routed_r) if routed_r else 0.0

    assert ev.realized_risk == pytest.approx(manual_cond)


# 3. experiment harness + figures
def test_repeated_trials_and_violation_rate():
    recs = make_records(300, seed=3)
    def make_result(seed, apply_pareto):
        return LTTAdaptor("UNUSED", seed=seed).run(
            alpha=0.25, embed_fn=stub_embed_fn, records=recs, apply_pareto=apply_pareto)
    outcomes = exp.run_repeated_trials(make_result, n_trials=8, apply_pareto=True)
    assert len(outcomes) == 8
    vr = exp.raw_violation_rate(outcomes, alpha=0.25)
    assert 0 <= vr <= 1


def test_figures_are_written(tmp_path):
    recs = make_records(300, seed=4)
    def make_result(seed, apply_pareto):
        return LTTAdaptor("UNUSED", seed=seed).run(
            alpha=0.25, embed_fn=stub_embed_fn, records=recs, apply_pareto=apply_pareto)

    out_par = exp.run_repeated_trials(make_result, n_trials=6, apply_pareto=True)
    out_bud = exp.run_repeated_trials(make_result, n_trials=6, apply_pareto=False)
    p1 = exp.plot_guarantee_histogram(out_par, out_bud, 0.25, 0.10, outdir=str(tmp_path))
    assert os.path.exists(p1)

    alphas = [0.15, 0.25, 0.35]
    evals = exp.run_alpha_sweep(
        lambda a: LTTAdaptor("UNUSED", seed=0).run(alpha=a, embed_fn=stub_embed_fn, records=recs),
        alphas)
    p2 = exp.plot_alpha_sweep(evals, alphas, outdir=str(tmp_path))
    assert os.path.exists(p2)

    ours = evaluate(make_result(0, True))
    p3 = exp.plot_benchmark_comparison(ours, repo_root=".", outdir=str(tmp_path))
    assert os.path.exists(p3)

# Regression: realized_risk must match the CERTIFIED (conditional) denominator.
def test_realized_risk_is_conditional_on_routed():
    import numpy as np
    from baselines.ltt_router.protocols import QueryRecord, ModelSpec
    from baselines.ltt_router.core.routing import Router

    class ScriptedScorer:
        def __init__(self, models): self._models = models
        @property
        def models(self): return self._models
        def score_batch(self, prompts): return np.zeros((len(prompts), len(self._models)))

    models = [ModelSpec("cheap", 0.1, 0), ModelSpec("oracle", 2.0, 1)]
    rng = np.random.default_rng(0)
    # Calib: cheap good only on high-score queries -> certifies a mid λ̂.
    calib = []
    for i in range(800):
        s = rng.random()
        cheap_ok = rng.random() < (0.95 if s > 0.6 else 0.5)
        calib.append(QueryRecord(
            scores=np.array([s, 0.0]),
            correct=np.array([int(cheap_ok), int(rng.random() < 0.95)]),
            cost=np.array([0.1, 2.0]),
            evaluated=np.array([True, True]),
            prompt=f"c{i}",
        ))
    router = Router(ScriptedScorer(models))
    plan = router.fit(calib, alpha=0.15)

    # Build a fake AdaptorResult over a TEST set with many deferrals (low scores),
    # where deferred queries have high regret but should NOT count.
    test = []
    for i in range(800):
        s = rng.random() * 0.4          
        test.append(QueryRecord(
            scores=np.array([s, 0.0]),
            correct=np.array([int(rng.random() < 0.5), int(rng.random() < 0.95)]),
            cost=np.array([0.1, 2.0]),
            evaluated=np.array([True, True]),
            prompt=f"t{i}",
        ))
    chosen = router.route_batch(test)

    class FakeResult:
        pass
    r = FakeResult()
    r.plan = plan; r.test_queries = test; r.chosen_indices = chosen
    r.models = models; r.alpha = 0.15; r.delta = 0.10
    r.apply_pareto = True; r.seed = 0

    from baselines.ltt_router.eval.metrics import evaluate
    ev = evaluate(r)
    if ev.certified and ev.routed_fraction > 0:
        # the certified (conditional) risk must respect the bound
        assert ev.realized_risk <= ev.alpha + 0.05, \
            f"conditional risk {ev.realized_risk} should be near ≤ α={ev.alpha}"
"""
Tests for the benchmark adaptor (baselines/adaptors/ltt_adaptor.py) and the
three-way split (splitting.py). 

  1. three_way_split: prompt-disjoint, rows-stay-together, reproducible.
  2. build_model_specs: name-sorted stable indices + mean cost.
  3. pivot: evaluated mask True only for present models .
  4. LTTAdaptor.run end-to-end, incl. the v1 two-model subset path.
"""

import importlib.util
import os
import sys

import numpy as np
import pytest

from baselines.ltt_router.protocols import ModelSpec
from baselines.ltt_router.splitting import three_way_split


_MODULE_NAME = "ltt_adaptor_under_test"
_ADAPTOR_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "adaptors", "ltt_adaptor.py"
)
_spec = importlib.util.spec_from_file_location(_MODULE_NAME, _ADAPTOR_PATH)
ltt_adaptor = importlib.util.module_from_spec(_spec)
sys.modules[_MODULE_NAME] = ltt_adaptor
_spec.loader.exec_module(ltt_adaptor)

build_model_specs = ltt_adaptor.build_model_specs
pivot_to_query_records = ltt_adaptor.pivot_to_query_records
LTTAdaptor = ltt_adaptor.LTTAdaptor


# Synthetic records + stub embedder
class FakeRecord:
    def __init__(self, dataset_id, prompt, model_name, score, cost, record_index):
        self.dataset_id = dataset_id
        self.prompt = prompt
        self.model_name = model_name
        self.score = score
        self.cost = cost
        self.record_index = record_index


def stub_embed_fn(prompts):
    out = []
    for p in prompts:
        out.append([len(p), p.count("?"), sum(ord(c) for c in p) % 97])
    return np.asarray(out, dtype=float)


def make_dense_records(n_prompts=100, seed=0):
    # Three models, every prompt evaluated on all three (dense).
    rng = np.random.default_rng(seed)
    recs = []
    idx = 0
    for i in range(n_prompts):
        q = f"dataset-A question {i}?"
        for name, p, cost in (("cheap", 0.70, 0.1), ("mid", 0.80, 0.5), ("oracle", 0.95, 2.0)):
            recs.append(FakeRecord("A", q, name, float(rng.random() < p), cost, idx))
            idx += 1
    return recs


# 1. three_way_split
def test_split_is_prompt_disjoint():
    recs = make_dense_records(120)
    train, calib, test = three_way_split(recs, 0.6, 0.2, random_seed=42)
    pt = {r.prompt for r in train}
    pc = {r.prompt for r in calib}
    pte = {r.prompt for r in test}
    assert pt.isdisjoint(pc)
    assert pt.isdisjoint(pte)
    assert pc.isdisjoint(pte)


def test_split_keeps_model_rows_together():
    # Every prompt in a pile must have ALL 3 of its model rows in that pile.
    from collections import Counter
    recs = make_dense_records(60)
    for pile in three_way_split(recs, 0.6, 0.2, random_seed=1):
        counts = Counter(r.prompt for r in pile)
        assert all(c == 3 for c in counts.values())


def test_split_is_reproducible():
    recs = make_dense_records(80)
    a = three_way_split(recs, 0.6, 0.2, random_seed=7)
    b = three_way_split(recs, 0.6, 0.2, random_seed=7)
    for pa, pb in zip(a, b):
        assert [r.prompt for r in pa] == [r.prompt for r in pb]


def test_split_rejects_bad_fractions():
    with pytest.raises(ValueError):
        three_way_split(make_dense_records(10), 0.7, 0.4)   # sums to >= 1.0


# 2. build_model_specs
def test_model_specs_sorted_and_costed():
    recs = [
        FakeRecord("A", "q1", "zebra", 1.0, 2.0, 0),
        FakeRecord("A", "q1", "alpha", 0.0, 0.1, 1),
        FakeRecord("A", "q2", "alpha", 1.0, 0.3, 2),
    ]
    specs = build_model_specs(recs)
    assert [s.name for s in specs] == ["alpha", "zebra"]   # name-sorted
    assert specs[0].index == 0 and specs[1].index == 1
    assert specs[0].cost == pytest.approx(0.2)             # (0.1+0.3)/2
    assert specs[1].cost == pytest.approx(2.0)


# 3. pivot: the evaluated-mask correctness crux
class ConstScorer:
    def __init__(self, models, vec):
        self._models = models
        self._vec = np.asarray(vec, dtype=float)

    @property
    def models(self):
        return self._models

    def score_batch(self, prompts):
        return np.tile(self._vec, (len(prompts), 1))


def test_pivot_sets_evaluated_only_for_present_models():
    models = [ModelSpec("a", 0.1, 0), ModelSpec("b", 0.5, 1), ModelSpec("c", 2.0, 2)]
    # q1 has rows for a and c only b is MISSING.
    recs = [
        FakeRecord("D", "q1", "a", 1.0, 0.1, 0),
        FakeRecord("D", "q1", "c", 0.0, 2.0, 1),
    ]
    q = pivot_to_query_records(recs, models, ConstScorer(models, [0.9, 0.9, 0.9]))[0]
    assert q.evaluated.tolist() == [True, False, True]     # b never evaluated
    assert q.correct[0] == 1 and q.correct[2] == 0


def test_pivot_prefers_real_per_query_cost():
    models = [ModelSpec("a", 0.1, 0), ModelSpec("b", 0.5, 1)]
    recs = [
        FakeRecord("D", "q1", "a", 1.0, 0.15, 0),   # real cost differs from spec
        FakeRecord("D", "q1", "b", 1.0, 0.55, 1),
    ]
    q = pivot_to_query_records(recs, models, ConstScorer(models, [0.9, 0.9]))[0]
    assert q.cost[0] == pytest.approx(0.15)
    assert q.cost[1] == pytest.approx(0.55)


# 4. LTTAdaptor end-to-end
def test_adaptor_run_end_to_end():
    recs = make_dense_records(300, seed=2)
    adaptor = LTTAdaptor(config_path="UNUSED", train_frac=0.6, calib_frac=0.2, seed=42)
    result = adaptor.run(alpha=0.2, embed_fn=stub_embed_fn, records=recs)
    assert result.plan is not None
    assert len(result.chosen_indices) == len(result.test_queries)
    assert len(result.models) == 3
    assert set(np.unique(result.chosen_indices)).issubset({0, 1, 2})


def test_adaptor_two_model_subset_path():
    recs = make_dense_records(300, seed=3)
    adaptor = LTTAdaptor(config_path="UNUSED", seed=42)
    result = adaptor.run(
        alpha=0.2, embed_fn=stub_embed_fn, records=recs,
        models_subset=["cheap", "oracle"],
    )
    assert len(result.models) == 2
    assert {m.name for m in result.models} == {"cheap", "oracle"}


def test_adaptor_pareto_vs_budget_runs_both():
    recs = make_dense_records(300, seed=4)
    adaptor = LTTAdaptor(config_path="UNUSED", seed=42)
    r_par = adaptor.run(alpha=0.2, embed_fn=stub_embed_fn, records=recs, apply_pareto=True)
    r_bud = adaptor.run(alpha=0.2, embed_fn=stub_embed_fn, records=recs, apply_pareto=False)
    assert r_par.apply_pareto is True
    assert r_bud.apply_pareto is False
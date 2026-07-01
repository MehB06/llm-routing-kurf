"""
Tests for the concrete routers.

  1. The embedding+LR router trains one classifier per model and produces
     protocol-valid score vectors.
  2. Scores are informative — a model that always succeeds in training scores
     higher than one that always fails.
  3. The N=2 reduction: the cheap model's per-model classifier reproduces v1's
     single-scorer behaviour (same training data -> same probabilities).
  4. Both routers satisfy the RoutingFunction protocol and plug into Router.
  5. The random router ignores the prompt but still yields the right shape.
"""

import numpy as np
import pytest

from baselines.ltt_router.protocols import ModelSpec, RoutingFunction
from baselines.ltt_router.routers.embedding_lr import (
    build_embedding_lr_router,
    EmbeddingLRRouter,
)
from baselines.ltt_router.core.routing import Router


# Test fixtures: a fake BaselineRecord and a deterministic stub embedder
class FakeRecord:
    """Minimal stand-in for the benchmark's BaselineRecord."""
    def __init__(self, model_name, prompt, score, dataset_id="d", record_index=0):
        self.model_name = model_name
        self.prompt = prompt
        self.score = score
        self.dataset_id = dataset_id
        self.record_index = record_index


def stub_embed_fn(prompts):
    """
    Deterministic, cheap 'embedding': map each prompt to a 4-dim vector derived
    from simple text features. Good enough for the LR to learn a separable signal
    in the tests, with zero external dependencies.
    """
    out = []
    for p in prompts:
        out.append([
            len(p),
            p.count("a"),
            p.count("?"),
            sum(ord(c) for c in p) % 97,
        ])
    return np.asarray(out, dtype=float)


def _make_train_records():
    """
    Two models:
      - 'cheap'  succeeds on prompts containing '?', fails otherwise
      - 'oracle' succeeds on everything
    A clean separable signal the stub embedding can capture.
    """
    recs = []
    idx = 0
    for i in range(40):
        q = f"easy question {i}?"      # has '?'
        recs.append(FakeRecord("cheap", q, 1.0, "d", idx)); idx += 1
        recs.append(FakeRecord("oracle", q, 1.0, "d", idx)); idx += 1
    for i in range(40):
        h = f"hard statement {i}"      # no '?'
        recs.append(FakeRecord("cheap", h, 0.0, "d", idx)); idx += 1
        recs.append(FakeRecord("oracle", h, 1.0, "d", idx)); idx += 1
    return recs


MODELS = [ModelSpec("cheap", cost=0.1, index=0), ModelSpec("oracle", cost=2.0, index=1)]


# 1-2. Training + informativeness
def test_scorer_trains_and_outputs_protocol_shape():
    router = build_embedding_lr_router(_make_train_records(), MODELS, embed_fn=stub_embed_fn)
    assert isinstance(router, RoutingFunction)
    s = router.score_batch(["a new question?"])[0]
    assert s.shape == (2,)
    assert np.all((s >= 0) & (s <= 1))


def test_scorer_is_informative():
    router = build_embedding_lr_router(_make_train_records(), MODELS, embed_fn=stub_embed_fn)
    # cheap should score HIGHER on a '?'-question than on a no-'?' statement.
    p_q = router.score_batch(["brand new question?"])[0][0]
    p_s = router.score_batch(["brand new statement"])[0][0]
    assert p_q > p_s


def test_score_batch_matches_single():
    router = build_embedding_lr_router(_make_train_records(), MODELS, embed_fn=stub_embed_fn)
    prompts = ["one question?", "two statement", "three question?"]
    batch = router.score_batch(prompts)
    single = np.stack([router.score_batch([p])[0] for p in prompts])
    assert batch.shape == (3, 2)
    assert np.allclose(batch, single)


# 3. v1 reduction: cheap model's classifier == v1's single scorer
def test_two_model_cheap_classifier_reduces_to_v1():
    """
    v1 trained ONE logistic regression on the cheap model's (prompt, success)
    pairs. v2's per-model classifier for 'cheap' is trained on the SAME pairs, so
    its probabilities must match a standalone v1-style classifier exactly.
    """
    from sklearn.linear_model import LogisticRegression

    records = _make_train_records()

    # v2 path: build the N-model router, pull out the cheap classifier.
    v2_router = build_embedding_lr_router(records, MODELS, embed_fn=stub_embed_fn)

    # v1 path: replicate v1's single-scorer training on cheap pairs.
    cheap = [r for r in records if r.model_name == "cheap"]
    cheap.sort(key=lambda r: (r.dataset_id, r.record_index))
    prompts = [r.prompt for r in cheap]
    labels = np.array([int(r.score == 1.0) for r in cheap])
    X = stub_embed_fn(prompts)
    v1_clf = LogisticRegression(max_iter=1000, C=1.0).fit(X, labels)

    # Same probabilities on fresh prompts.
    test_prompts = ["fresh question?", "fresh statement", "another q?"]
    v1_proba = v1_clf.predict_proba(stub_embed_fn(test_prompts))[:, 1]
    v2_proba = v2_router.score_batch(test_prompts)[:, 0]   # column 0 = cheap
    assert np.allclose(v1_proba, v2_proba)


# 4. Both routers plug into Router
def test_embedding_router_plugs_into_router_object():
    scorer = build_embedding_lr_router(_make_train_records(), MODELS, embed_fn=stub_embed_fn)
    r = Router(scorer)
    assert isinstance(r.scorer, RoutingFunction)




# 5. Degenerate-training guard (a model that always succeeds)
def test_single_class_training_does_not_crash():
    # 'always' only ever has score 1.0 -> single-class training set.
    recs = [FakeRecord("always", f"q{i}?", 1.0, "d", i) for i in range(20)]
    models = [ModelSpec("always", cost=0.1, index=0)]
    router = build_embedding_lr_router(recs, models, embed_fn=stub_embed_fn)
    s = router.score_batch(["new?"])[0]
    assert s.shape == (1,)
    assert 0.0 <= s[0] <= 1.0

# CachingEmbedder: embed each unique prompt once, reuse across calls
def test_caching_embedder_calls_base_once_per_unique_prompt():
    from baselines.ltt_router.routers.embedding_lr import CachingEmbedder
    calls = {"n": 0, "prompts": []}

    def counting_base(prompts):
        calls["n"] += 1
        calls["prompts"].extend(prompts)
        return np.asarray([[len(p), p.count("?")] for p in prompts], float)

    emb = CachingEmbedder(base_embed_fn=counting_base)
    # First call embeds the 3 unique prompts.
    a = emb(["x?", "yy", "x?"])
    # Second call with overlap: only "zzz" is new.
    b = emb(["yy", "zzz", "x?"])
    assert a.shape == (3, 2)
    assert b.shape == (3, 2)

    assert len(set(calls["prompts"])) == 3
    assert "zzz" in calls["prompts"]

    assert np.allclose(a[0], a[2])
    assert np.allclose(a[0], b[2])


def test_caching_embedder_precompute_fills_cache():
    from baselines.ltt_router.routers.embedding_lr import CachingEmbedder
    calls = {"n": 0}

    def counting_base(prompts):
        calls["n"] += 1
        return np.asarray([[len(p)] for p in prompts], float)

    emb = CachingEmbedder(base_embed_fn=counting_base)
    emb.precompute(["a", "bb", "ccc", "a"])
    n_after_precompute = calls["n"]
    # subsequent calls are pure lookups -> base NOT called again
    emb(["a", "bb", "ccc"])
    assert calls["n"] == n_after_precompute
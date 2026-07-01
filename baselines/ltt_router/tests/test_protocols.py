"""
Tests for the protocols.py.

These verify that:
  - ModelSpec / QueryRecord validate their invariants,
  - a trivial RoutingFunction implementation satisfies the runtime-checkable
    protocol (proving "any router plugs in"),
"""

import numpy as np
import pytest

from baselines.ltt_router.protocols import (
    ModelSpec,
    QueryRecord,
    RoutingFunction,
)


# ModelSpec
def test_modelspec_valid():
    m = ModelSpec(name="gpt-5", cost=1.0, index=0)
    assert m.name == "gpt-5"
    assert m.cost == 1.0
    assert m.index == 0


def test_modelspec_rejects_negative_cost():
    with pytest.raises(ValueError):
        ModelSpec(name="x", cost=-0.1, index=0)


def test_modelspec_rejects_negative_index():
    with pytest.raises(ValueError):
        ModelSpec(name="x", cost=0.1, index=-1)


# QueryRecord
def _make_record(n=3):
    return QueryRecord(
        scores=np.array([0.2, 0.5, 0.9]),
        correct=np.array([0, 1, 1]),
        cost=np.array([0.1, 0.5, 2.0]),
        evaluated=np.array([True, True, True]),
        dataset_id="toy",
        prompt="What is 2+2?",
    )


def test_queryrecord_valid():
    r = _make_record()
    assert r.n_models == 3


def test_queryrecord_rejects_misaligned_arrays():
    with pytest.raises(ValueError):
        QueryRecord(
            scores=np.array([0.1, 0.2, 0.3]),
            correct=np.array([0, 1]),            # wrong length
            cost=np.array([1.0, 1.0, 1.0]),
            evaluated=np.array([True, True, True]),
        )


def test_queryrecord_rejects_all_unevaluated():
    with pytest.raises(ValueError):
        QueryRecord(
            scores=np.array([0.1, 0.2]),
            correct=np.array([0, 0]),
            cost=np.array([1.0, 1.0]),
            evaluated=np.array([False, False]),  # nothing evaluated
        )


# RoutingFunction protocol 
class ConstantRouter:
    """Minimal RoutingFunction: always returns the same score vector."""

    def __init__(self, models, vec):
        self._models = models
        self._vec = np.asarray(vec, dtype=float)

    @property
    def models(self):
        return self._models

    def score_batch(self, prompts) -> np.ndarray:
        return np.tile(self._vec, (len(prompts), 1))


def test_constant_router_satisfies_protocol():
    models = [
        ModelSpec("cheap", cost=0.1, index=0),
        ModelSpec("mid", cost=0.5, index=1),
        ModelSpec("oracle", cost=2.0, index=2),
    ]
    r = ConstantRouter(models, [0.3, 0.6, 0.9])
    assert isinstance(r, RoutingFunction)         # runtime_checkable structural match
    out = r.score_batch(["anything"])[0]
    assert out.shape == (3,)
    assert np.allclose(out, [0.3, 0.6, 0.9])
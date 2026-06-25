"""
Tests for the three-way split.

Beyond disjointness/reproducibility (covered in test_adaptor.py), the key
property here is PROVENANCE: our TEST set must be identical to the benchmark's
own split_by_dataset_then_prompt, so our comparison against the baselines is
like-for-like. We assert that identity directly when the benchmark stack is
importable; otherwise the test skips (the benchmark loader pulls a heavy
generator/openai chain that isn't needed for the rest of the suite).
"""

import random

import pytest

from baselines.schema import BaselineRecord
from baselines.ltt_router.splitting import (
    three_way_split,
    _benchmark_train_test_split,
)


def _mk(ds, prompt, idx, model):
    return BaselineRecord(
        dataset_id=ds, split="test", model_name=model, record_index=idx,
        origin_query=prompt, prompt=prompt, prediction="", raw_output=None,
        ground_truth="", score=1.0, prompt_tokens=0, completion_tokens=0, cost=0.01,
    )


def _multi_dataset_records():
    recs, ix = [], 0
    for ds, npr in [("aime", 50), ("gpqa", 30), ("simpleqa", 40)]:
        for p in range(npr):
            for m in ("a", "b", "c"):
                recs.append(_mk(ds, f"{ds}_p{p}", ix, m))
            ix += 1
    return recs


def test_three_way_test_set_disjoint_and_reproducible():
    recs = _multi_dataset_records()
    tr, cal, te = three_way_split(recs, 0.6, 0.2, random_seed=42)
    ptr = {r.prompt for r in tr}
    pca = {r.prompt for r in cal}
    pte = {r.prompt for r in te}
    assert ptr.isdisjoint(pca) and ptr.isdisjoint(pte) and pca.isdisjoint(pte)
    # reproducible
    tr2, cal2, te2 = three_way_split(recs, 0.6, 0.2, random_seed=42)
    assert {r.prompt for r in te2} == pte
    assert {r.prompt for r in cal2} == pca


def test_test_set_matches_benchmark_splitter():
    """Our test set must equal BaselineDataLoader.split_by_dataset_then_prompt's."""
    recs = _multi_dataset_records()
    pool_ratio = 0.8  # train_frac + calib_frac for the (0.6, 0.2) default

    try:
        from baselines.data_loader import BaselineDataLoader
        _, their_test = BaselineDataLoader.split_by_dataset_then_prompt(
            None, recs, train_ratio=pool_ratio, random_seed=42
        )
    except Exception:
        pytest.skip("benchmark loader stack not importable in this environment")

    _, our_test = _benchmark_train_test_split(recs, train_ratio=pool_ratio, random_seed=42)
    assert sorted({r.prompt for r in our_test}) == sorted({r.prompt for r in their_test})

    # and three_way_split's test set must match too
    _, _, te = three_way_split(recs, 0.6, 0.2, random_seed=42)
    assert sorted({r.prompt for r in te}) == sorted({r.prompt for r in their_test})

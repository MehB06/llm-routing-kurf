"""
Three-way (train / calibration / test) prompt-level split.

LTT needs THREE disjoint sets:
  - train:       fit the scorer f̂(x) = P(model i succeeds | query)
  - calibration: run LTT's hypothesis test to certify the threshold λ̂
  - test:        held-out final evaluation

Calib MUST be disjoint from train, or LTT's risk estimate is optimistic and the
guarantee breaks. Test must be disjoint from both, or the reported numbers inflate.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Tuple


def _prompts_by_dataset(ds_records: List) -> Tuple[Dict[str, list], List[str]]:
    """
    Group one dataset's records by prompt and return (prompt->records, ordered
    prompts) where prompts are sorted by their first record_index — the stable,
    reproducible ordering both split stages rely on.
    """
    prompt_to_records = defaultdict(list)
    for r in ds_records:
        prompt_to_records[r.prompt].append(r)
    unique_prompts = sorted(
        prompt_to_records.keys(),
        key=lambda p: min(rr.record_index for rr in prompt_to_records[p]),
    )
    return prompt_to_records, unique_prompts


def _benchmark_train_test_split(
    records: List,
    train_ratio: float,
    random_seed: int,
) -> Tuple[List, List]:
    """
    Faithful reproduction of
    BaselineDataLoader.split_by_dataset_then_prompt(records, train_ratio, seed)
    with no OOD datasets. Prompt-level, stratified within each dataset.

    Kept byte-identical to the benchmark splitter (global RNG, truncating
    int(), dict dataset order) so our TEST set matches every baseline's — this is
    asserted by test_splitting.test_test_set_matches_benchmark_splitter.
    """
    # WARNING: this mutates the GLOBAL `random` module state. It is deliberate —
    # the benchmark's own splitter seeds the global RNG, and byte-identical
    # reproduction is what makes our TEST set match every baseline's. Do NOT
    # "fix" this with a local random.Random(); it would change the split. Any
    # caller relying on the global RNG must re-seed after calling this.
    random.seed(random_seed)

    dataset_groups = defaultdict(list)
    for r in records:
        dataset_groups[r.dataset_id].append(r)

    train_records, test_records = [], []
    for _dataset_id, ds_records in dataset_groups.items():
        prompt_to_records, unique_prompts = _prompts_by_dataset(ds_records)

        n_train = int(len(unique_prompts) * train_ratio)
        indices = list(range(len(unique_prompts)))
        random.shuffle(indices)
        train_idx = set(indices[:n_train])

        for i, prompt in enumerate(unique_prompts):
            recs = prompt_to_records[prompt]
            (train_records if i in train_idx else test_records).extend(recs)

    return train_records, test_records


def three_way_split(
    records: List,
    train_frac: float = 0.6,
    calib_frac: float = 0.2,
    random_seed: int = 42,
) -> Tuple[List, List, List]:
    """
    Split records into (train, calibration, test) at the prompt level.

    test_frac = 1 - train_frac - calib_frac. The TEST set is produced by the
    benchmark's split algorithm at train_ratio = train_frac + calib_frac, so it
    matches every other baseline's test set. The remaining (non-test) pool is
    then split into train + calib by prompt, stratified within each dataset.
    Same seed => identical split.
    """
    if not (0.0 < train_frac < 1.0) or not (0.0 < calib_frac < 1.0):
        raise ValueError("train_frac and calib_frac must each be in (0, 1)")
    if train_frac + calib_frac >= 1.0:
        raise ValueError("train_frac + calib_frac must leave room for a test set")

    # Step 1: benchmark-identical TEST carve-out. pool = everything not test.
    pool_ratio = train_frac + calib_frac
    train_pool, test_records = _benchmark_train_test_split(
        records, train_ratio=pool_ratio, random_seed=random_seed
    )

    # Step 2: split the pool into train + calib. calib's share OF THE POOL is
    # calib_frac / (train_frac + calib_frac). 
    calib_share_of_pool = calib_frac / pool_ratio

    by_dataset = defaultdict(list)
    for r in train_pool:
        by_dataset[r.dataset_id].append(r)

    train_records, calib_records = [], []
    for _dataset_id, ds_records in sorted(by_dataset.items()):
        prompt_to_records, unique_prompts = _prompts_by_dataset(ds_records)

        n = len(unique_prompts)
        n_calib = int(round(n * calib_share_of_pool))

        rng = random.Random(random_seed + 1)
        positions = list(range(n))
        rng.shuffle(positions)
        calib_pos = set(positions[:n_calib])

        for i, prompt in enumerate(unique_prompts):
            recs = prompt_to_records[prompt]
            (calib_records if i in calib_pos else train_records).extend(recs)

    return train_records, calib_records, test_records
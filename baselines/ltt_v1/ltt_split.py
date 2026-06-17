"""
Three-way data split for LTT router calibration.

LTT structurally needs THREE disjoint sets:
  - train: fit the scorer f̂(x) that predicts P(cheap model succeeds | query)
  - calibration: run LTT's hypothesis testing to choose the threshold λ̂
  - test: held-out final evaluation

The calibration set MUST be independent of the scorer's training data, or the
risk estimate LTT computes is optimistic and the guarantee breaks. The test set
must be independent of both, or the reported numbers are inflated.

DESIGN CHOICE: split at the PROMPT level, not the record level.
Every query has 13 rows (one per model). All 13 must move into the same pile
together. If we split rows independently, the same query could land in train
(under model A) and test (under model B), that is leakage
"""

from __future__ import annotations
import random
from collections import defaultdict
from typing import List, Tuple


def three_way_split(
    records: List,
    train_frac: float = 0.6,
    calib_frac: float = 0.2,
    random_seed: int = 42,
) -> Tuple[List, List, List]:
    """
    Split records into (train, calibration, test) at the prompt level,
    stratified within each dataset.

    Stratifying within dataset matters: datasets have very different sizes. 
    A global prompt shuffle could starve a small dataset out of one of the piles. 
    Splitting within each dataset guarantees all three piles see every dataset 
    in proportion.

    Args:
        records: all BaselineRecord objects from the loader.
        train_frac: fraction of unique prompts (per dataset) for training.
        calib_frac: fraction for calibration. test_frac = 1 - train - calib.
        random_seed: fixed for reproducibility (same seed => identical split).

    Returns:
        (train_records, calib_records, test_records)
    """
    assert train_frac + calib_frac < 1.0, "train + calib must leave room for test"

    train_records, calib_records, test_records = [], [], []

    # Group records by dataset so we can split within each one.
    by_dataset: dict = defaultdict(list)
    for r in records:
        by_dataset[r.dataset_id].append(r)

    for dataset_id, ds_records in sorted(by_dataset.items()):
        # Map each unique prompt -> all of its records (one per model).
        prompt_to_records: dict = defaultdict(list)
        for r in ds_records:
            prompt_to_records[r.prompt].append(r)

        # Deterministic ordering of prompts before shuffling, so the seed fully
        # determines the split. We order by the query's record_index.
        unique_prompts = sorted(
            prompt_to_records.keys(),
            key=lambda p: min(rr.record_index for rr in prompt_to_records[p]),
        )

        n = len(unique_prompts)
        n_train = int(n * train_frac)
        n_calib = int(n * calib_frac)
        # test gets the remainder: n - n_train - n_calib

        # Shuffle prompt positions with a per-run seeded RNG (not the global one),
        # so behaviour is independent of any other random calls in the process.
        rng = random.Random(random_seed)
        positions = list(range(n))
        rng.shuffle(positions)

        train_pos = set(positions[:n_train])
        calib_pos = set(positions[n_train:n_train + n_calib])
        # everything else -> test

        for i, prompt in enumerate(unique_prompts):
            recs = prompt_to_records[prompt]
            if i in train_pos:
                train_records.extend(recs)
            elif i in calib_pos:
                calib_records.extend(recs)
            else:
                test_records.extend(recs)

    return train_records, calib_records, test_records
"""
Three-way (train / calibration / test) prompt-level split.

LTT needs THREE disjoint sets:
  - train:       fit the scorer f̂(x) = P(model i succeeds | query)
  - calibration: run LTT's hypothesis test to certify the threshold λ̂
  - test:        held-out final evaluation

Calib MUST be disjoint from train, or LTT's risk estimate is optimistic and the
guarantee breaks. Test must be disjoint from both, or the reported numbers inflate.

We split at the PROMPT level: Stratify within each dataset so small datasets are represented in all three piles.
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
    Split records into (train, calibration, test) at the prompt level, stratified
    within each dataset. test_frac = 1 - train_frac - calib_frac.
    Same seed => identical split.
    """
    if not (0.0 < train_frac < 1.0) or not (0.0 < calib_frac < 1.0):
        raise ValueError("train_frac and calib_frac must each be in (0, 1)")
    if train_frac + calib_frac >= 1.0:
        raise ValueError("train_frac + calib_frac must leave room for a test set")

    train_records, calib_records, test_records = [], [], []

    # Group by dataset so we can split within each one.
    by_dataset = defaultdict(list)
    for r in records:
        by_dataset[r.dataset_id].append(r)

    for dataset_id, ds_records in sorted(by_dataset.items()):
        # Map each unique prompt -> all its records (one per model).
        prompt_to_records = defaultdict(list)
        for r in ds_records:
            prompt_to_records[r.prompt].append(r)

        # Deterministic prompt ordering before the seeded shuffle, so the seed
        # fully determines the split.
        unique_prompts = sorted(
            prompt_to_records.keys(),
            key=lambda p: min(rr.record_index for rr in prompt_to_records[p]),
        )

        n = len(unique_prompts)
        n_train = int(n * train_frac)
        n_calib = int(n * calib_frac)

        # Per-call seeded RNG (not the global one), so behaviour is independent of
        # any other random calls in the process.
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
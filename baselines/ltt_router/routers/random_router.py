"""
Random router (ablation)

This RoutingFunction emits uniform-random per-model scores, ignoring the
prompt entirely. The risk guarantee comes from the LTT CALIBRATION, not from the scorer.

Even with a meaningless scorer, the calibrated cheapest-safe rule must still
honour the ≤ α risk guarantee because calibration certifies the realized risk
of whatever scores it is given. A good scorer (the embedding+LR one) saves cost 
at the same guaranteed safety.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from baselines.ltt_router.protocols import ModelSpec


class RandomRouter:
    """A RoutingFunction returning seeded uniform-random scores per model."""

    def __init__(self, models: List[ModelSpec], seed: int = 0):
        self._models = models
        self._rng = np.random.default_rng(seed)

    @property
    def models(self) -> Sequence[ModelSpec]:
        return self._models

    def score(self, prompt: str, dataset_id: str = "") -> np.ndarray:
        return self._rng.random(len(self._models))

    def score_batch(self, prompts: List[str]) -> np.ndarray:
        return self._rng.random((len(prompts), len(self._models)))
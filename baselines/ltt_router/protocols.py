"""
Interfaces that decouple the pluggable pieces of the system:

    RoutingFunction:  f(x) -> scores[N]      one score per model

Nothing below the RoutingFunction boundary knows which router produced the scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class ModelSpec:
    """
    Static description of one routable model.

    name: Model identifier as it appears in the benchmark.
    cost:
        Representative scalar cost of querying this model, used to order models
        cheapest-first and to compute cost-savings metrics. Only the ordering and
        ratios matter, so any consistent positive scale is fine.
    index:
        The model's column position in the per-query score / cost arrays — the
        single source of truth that keeps the N-length arrays aligned across the
        whole pipeline.
    """

    name: str
    cost: float
    index: int

    def __post_init__(self) -> None:
        if self.cost < 0:
            raise ValueError(f"cost must be non-negative, got {self.cost} for {self.name!r}")
        if self.index < 0:
            raise ValueError(f"index must be non-negative, got {self.index} for {self.name!r}")


@dataclass(frozen=True)
class QueryRecord:
    """
    One query, aligned across all N candidate models. The per-query unit the
    calibration core and routing rule consume, produced by the adaptor from a
    group of BaselineRecords that share a prompt.

    scores:
        float[N] — the router's "route-to-me" score per model. Higher = more
        confident the model will succeed. Calibration only thresholds it.
    correct:
        int[N] in {0, 1} — did model i produce the right answer? Entries where
        evaluated[i] is False are UNDEFINED and ignored.
    cost:
        float[N] — per-model cost on this query.
    evaluated:
        bool[N] — was model i actually run on this query in the benchmark?
    dataset_id:
        Source dataset, kept for stratified splitting and per-dataset metrics.
    prompt:
        The shared prompt; the key that groups the N rows together.
    """

    scores: np.ndarray
    correct: np.ndarray      # valid only where evaluated is True
    cost: np.ndarray
    evaluated: np.ndarray
    dataset_id: str = ""
    prompt: str = ""

    def __post_init__(self) -> None:
        n = self.scores.shape[0]
        for name, arr in (
            ("correct", self.correct),
            ("cost", self.cost),
            ("evaluated", self.evaluated),
        ):
            if arr.shape[0] != n:
                raise ValueError(
                    f"QueryRecord arrays misaligned: scores has N={n} but "
                    f"{name} has N={arr.shape[0]}"
                )
        if not self.evaluated.any():
            raise ValueError(
                "QueryRecord has no evaluated models; cannot compute regret. "
                "The adaptor should drop such queries."
            )

    @property
    def n_models(self) -> int:
        return int(self.scores.shape[0])


@runtime_checkable
class RoutingFunction(Protocol):
    """
    The injected router. Any object that can score a query against the N models
    satisfies this protocol.

    A RoutingFunction is responsible ONLY for producing scores and declaring the
    model universe it scores over; it knows nothing about what lives above it.
    """

    @property
    def models(self) -> Sequence[ModelSpec]:
        """The ordered model universe this router scores over. Length N."""
        ...

    def score_batch(self, prompts: Sequence[str]) -> np.ndarray:
        """
        Return float[M, N] route-to-me scores: one row per prompt, aligned to
        self.models. This is the only scoring entry point the pipeline uses.
        """
        ...
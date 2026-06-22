"""
The general N-model routing loss: regret.

Definition:
L(chosen) = correctness_of_best_attainable - correctness_of_chosen

    L(chosen) = 1  iff  (chosen model was WRONG) AND (some evaluated model was RIGHT)
              = 0  otherwise

We only suffer loss when we picked a model that failed and a better
choice existed. If the chosen model was right, no regret. If every evaluated
model failed anyway, routing lost us nothing, so no regret either.

Note we take the baseline over the full evaluated pool, not the
Pareto-surviving subset. The loss should measure regret against what
was genuinely achievable on the data. Restricting the candidate set the router is allowed 
to pick from is a routing policy decision and lives in routing.py. 
"""

from __future__ import annotations

import numpy as np

# Re-exported for callers that want a name for "this entry is a placeholder".
MISSING = None


def regret_loss(
    chosen_idx: int,
    correct: np.ndarray,
    evaluated: np.ndarray,
) -> float:
    """
    Regret of choosing chosen_idx for one query.

    chosen_idx:
        Index of the model the router selected.
    correct:
        int[N] correctness in {0, 1}. Entries where evaluated is False
        are ignored.
    evaluated:
        bool[N] mask of which models were actually run on this query.

    Returns
        0.0 if no regret, 1.0 if the chosen model was wrong while a better
        (evaluated, correct) model existed.
    """
    correct = np.asarray(correct)
    evaluated = np.asarray(evaluated, dtype=bool)

    if not evaluated.any():
        raise ValueError("No evaluated models for this query; regret is undefined.")
    if not (0 <= chosen_idx < correct.shape[0]):
        raise ValueError(f"chosen_idx {chosen_idx} out of range for N={correct.shape[0]}")
    if not evaluated[chosen_idx]:
        raise ValueError(
            f"chosen_idx {chosen_idx} was not evaluated on this query; "
            "the routing rule must only select evaluated models (or the adaptor "
            "must guarantee the fallback model is always evaluated)."
        )

    chosen_correct = int(correct[chosen_idx]) == 1

    # was ANY evaluated model correct?
    best_attainable_correct = bool(
        np.any((correct == 1) & evaluated)
    )

    # Regret iff we were wrong AND a correct choice existed.
    return float((not chosen_correct) and best_attainable_correct)


def per_query_regret(
    chosen_indices: np.ndarray,
    correct_matrix: np.ndarray,
    evaluated_matrix: np.ndarray,
) -> np.ndarray:
    """
    Vectorised regret over a batch of queries.convenience for the calibration loss table and the metrics. 
    It is exactly equivalent to calling regret_loss per row.

    chosen_indices:
        int[M] chosen model index per query.
    correct_matrix:
        int[M, N] correctness per (query, model).
    evaluated_matrix:
        bool[M, N] evaluated mask per (query, model).

    Returns
        float[M] regret (0.0/1.0) per query.
    """
    chosen_indices = np.asarray(chosen_indices)
    correct_matrix = np.asarray(correct_matrix)
    evaluated_matrix = np.asarray(evaluated_matrix, dtype=bool)

    m, n = correct_matrix.shape
    if chosen_indices.shape[0] != m:
        raise ValueError(
            f"chosen_indices has {chosen_indices.shape[0]} rows but "
            f"correct_matrix has {m}"
        )
    if evaluated_matrix.shape != correct_matrix.shape:
        raise ValueError("correct_matrix and evaluated_matrix shapes differ")

    rows = np.arange(m)

    # Was the chosen model evaluated?
    chosen_eval = evaluated_matrix[rows, chosen_indices]
    if not chosen_eval.all():
        bad = int(np.argmin(chosen_eval))
        raise ValueError(
            f"query {bad}: chosen model {int(chosen_indices[bad])} was not evaluated"
        )

    chosen_correct = correct_matrix[rows, chosen_indices] == 1

    # any evaluated model correct?
    any_correct = np.any((correct_matrix == 1) & evaluated_matrix, axis=1)

    regret = (~chosen_correct) & any_correct
    return regret.astype(float)
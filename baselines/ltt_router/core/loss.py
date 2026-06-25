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
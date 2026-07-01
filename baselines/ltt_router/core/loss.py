"""
The general N-model routing loss: regret.

    L(chosen) = 1  iff  (chosen model was WRONG) AND (some evaluated model was RIGHT)
              = 0  otherwise

We only suffer loss when we picked a model that failed while a better choice
existed. If the chosen model was right, or every evaluated model failed anyway,
there is no regret.

The baseline is taken over the full evaluated pool, not the Pareto-surviving
subset, so the loss measures regret against what was genuinely achievable.
Restricting the candidate set the router may pick from is a routing-policy
decision and lives in routing.py.
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

    chosen_correct = correct[chosen_idx] == 1
    any_correct = bool(np.any((correct == 1) & evaluated))

    # Regret iff we were wrong AND a correct choice existed.
    return float((not chosen_correct) and any_correct)
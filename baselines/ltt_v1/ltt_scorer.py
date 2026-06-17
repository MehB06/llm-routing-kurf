"""
Scoring function for the LTT router.

The scorer estimates  P(cheap model succeeds | query)  for an incoming query.
The router later routes a query to the cheap model only when this probability
clears the calibrated threshold λ̂.

  1. Embed the prompt text with a sentence transformer
  2. Fit a logistic regression mapping that vector to the cheap model's success
     probability, trained ONLY on the train split.
"""

from __future__ import annotations
import numpy as np
from typing import List

# Module-level cache so we load the (heavy) embedding model only once per process.
_EMBED_MODEL = None
_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def _get_embed_model(model_name: str = _EMBED_MODEL_NAME):
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBED_MODEL = SentenceTransformer(model_name)
    return _EMBED_MODEL


def get_embeddings(texts: List[str], show_progress: bool = True) -> np.ndarray:
    """Embed a list of texts. Returns array of shape (n, embedding_dim)."""
    model = _get_embed_model()
    return np.asarray(
        model.encode(texts, show_progress_bar=show_progress, batch_size=64)
    )


def _cheap_examples(train_records: List, cheap_model: str):
    """Pull (prompt, success) pairs for the cheap model from the train split."""
    cheap = [r for r in train_records if r.model_name == cheap_model]
    # Deterministic order for reproducibility.
    cheap.sort(key=lambda r: (r.dataset_id, r.record_index))
    prompts = [r.prompt for r in cheap]
    labels = np.array([int(r.score == 1.0) for r in cheap])  # 1 = cheap succeeded
    return prompts, labels


def build_scorer(train_records: List, cheap_model: str) -> "LogisticRegressionScorer":
    """
    Train the scorer on the train split.

    Args:
        train_records: records from the TRAIN split only.
        cheap_model: name of the cheap model.

    Returns:
        A trained LogisticRegressionScorer.
    """
    from sklearn.linear_model import LogisticRegression

    prompts, labels = _cheap_examples(train_records, cheap_model)
    print(f"[scorer] training on {len(prompts)} cheap-model examples "
          f"(success rate {labels.mean():.1%})")

    X = get_embeddings(prompts)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X, labels)
    print(f"[scorer] train accuracy {clf.score(X, labels):.3f}")

    return LogisticRegressionScorer(clf, cheap_model)


class LogisticRegressionScorer:
    """Wraps a trained logistic regression + embedding pipeline."""

    def __init__(self, clf, cheap_model: str):
        self.clf = clf
        self.cheap_model = cheap_model

    def predict_proba(self, prompts: List[str], show_progress: bool = True) -> np.ndarray:
        """Return P(cheap model succeeds) for each prompt. Shape (n,)."""
        X = get_embeddings(prompts, show_progress=show_progress)
        return self.clf.predict_proba(X)[:, 1]  # probability of class 1 (success)
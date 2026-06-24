"""
The trained scorer .

Trains ONE logistic regression PER MODEL, all sharing the same prompt
embedding: for model i, P(model i succeeds | query). The score vector
[P(model_0 succeeds), ..., P(model_{N-1} succeeds)] is exactly the
RoutingFunction.score output the rest of the pipeline consumes. The
cheapest-safe rule then walks models cheapest-first and routes to the first whose
success-probability clears λ̂.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import numpy as np

from baselines.ltt_router.protocols import ModelSpec


EmbedFn = Callable[[List[str]], np.ndarray]


class _ConstantClassifier:
    """
    Stand-in for a LogisticRegression when the training labels were single-class. 
    Returns a constant success probability, so the model stays in the N-model pipeline
    instead of being silently dropped. 
    """

    def __init__(self, success_proba: float):
        self.success_proba = float(success_proba)
        # Two-column [P(fail), P(success)] to match LogisticRegression output.
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X) -> np.ndarray:
        n = X.shape[0]
        out = np.empty((n, 2))
        out[:, 0] = 1.0 - self.success_proba
        out[:, 1] = self.success_proba
        return out


# Default embedding backend 
_EMBED_MODEL = None
_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def default_embed_fn(prompts: List[str], show_progress: bool = False) -> np.ndarray:
    """
    Sentence-transformer embedding. Loaded once per process and cached.
    """
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBED_MODEL = SentenceTransformer(_EMBED_MODEL_NAME)
    return np.asarray(
        _EMBED_MODEL.encode(prompts, show_progress_bar=show_progress, batch_size=64)
    )


class CachingEmbedder:
    """
    Embeds each UNIQUE prompt exactly once and caches the vector, so repeated
    calls (across trials, α-values, Pareto/budget runs) are dictionary lookups
    instead of re-running the transformer.
    """

    def __init__(self, base_embed_fn=None, dim: Optional[int] = None):
        self._base = base_embed_fn or default_embed_fn
        self._cache: dict = {}
        self._dim = dim

    def precompute(self, prompts: List[str], show_progress: bool = True) -> None:
        """Embed all unseen unique prompts in one batched pass (fills the cache)."""
        unseen = list({p for p in prompts if p not in self._cache})
        if not unseen:
            return
        vecs = np.asarray(self._base(unseen))
        if self._dim is None:
            self._dim = vecs.shape[1]
        for p, v in zip(unseen, vecs):
            self._cache[p] = v

    def __call__(self, prompts: List[str]) -> np.ndarray:
        # Embed any prompts not yet cached (single batched call), then look up all.
        missing = [p for p in prompts if p not in self._cache]
        if missing:
            uniq = list(dict.fromkeys(missing))   # de-dup, preserve order
            vecs = np.asarray(self._base(uniq))
            if self._dim is None:
                self._dim = vecs.shape[1]
            for p, v in zip(uniq, vecs):
                self._cache[p] = v
        return np.asarray([self._cache[p] for p in prompts])

# Building the per-model training data
def _per_model_examples(train_records: Sequence, model_name: str):
    """
    Pull (prompt, success) pairs for one model from the train split.
    Deterministic ordering for reproducibility 
    """
    rows = [r for r in train_records if r.model_name == model_name]
    rows.sort(key=lambda r: (r.dataset_id, r.record_index))
    prompts = [r.prompt for r in rows]
    labels = np.array([int(r.score == 1.0) for r in rows])
    return prompts, labels

# The router
class EmbeddingLRRouter:
    """
    A RoutingFunction: embed the prompt once, then apply one trained logistic
    regression per model to get per-model success probabilities.
    """

    def __init__(
        self,
        models: List[ModelSpec],
        classifiers: dict,          
        embed_fn: EmbedFn,
        fallback_proba: float = 0.0,
    ):
        self._models = models
        self._clf = classifiers
        self._embed_fn = embed_fn
        self._fallback_proba = fallback_proba
        self._name_to_index = {m.name: m.index for m in models}

    @property
    def models(self) -> Sequence[ModelSpec]:
        return self._models

    def score(self, prompt: str, dataset_id: str = "") -> np.ndarray:
        return self.score_batch([prompt])[0]

    def score_batch(self, prompts: List[str]) -> np.ndarray:
        """
        Return float[M, N] success probabilities (M prompts, N models).        
        """
        X = self._embed_fn(prompts)
        n_models = len(self._models)
        out = np.full((len(prompts), n_models), self._fallback_proba, dtype=float)
        for m in self._models:
            clf = self._clf.get(m.name)
            if clf is None:
                # No classifier for this model (never seen in train) -> leave at
                # fallback probability.
                continue
            # predict_proba may be 1-column if the model only ever succeeded/failed
            # in training; guard that degenerate case.
            proba = clf.predict_proba(X)
            if proba.shape[1] == 2:
                out[:, m.index] = proba[:, 1]
            else:
                # single class seen in training: that class's constant probability
                out[:, m.index] = float(clf.classes_[0])
        return out


def build_embedding_lr_router(
    train_records: Sequence,
    models: List[ModelSpec],
    embed_fn: Optional[EmbedFn] = None,
    C: float = 1.0,
    max_iter: int = 1000,
    verbose: bool = False,
) -> EmbeddingLRRouter:
    """
    Train one logistic regression per model on the TRAIN split only.

    train_records:
        Raw benchmark records from the TRAIN split (disjoint from calib/test).
    models:
        The N-model universe (names + costs + indices). A model with no train
        rows simply gets no classifier and falls back to fallback_proba.
    embed_fn:
        Prompt embedder. Defaults to the sentence-transformer backbone; pass a
        stub in tests.
    C, max_iter:
        Logistic-regression hyperparameters.
    """
    from sklearn.linear_model import LogisticRegression

    if embed_fn is None:
        embed_fn = default_embed_fn

    classifiers = {}
    for m in models:
        prompts, labels = _per_model_examples(train_records, m.name)
        if len(prompts) == 0:
            if verbose:
                print(f"[scorer] {m.name}: no train rows, skipping")
            continue
        X = embed_fn(prompts)
        if len(np.unique(labels)) < 2:
            # Degenerate: the model always succeeded (or always failed) in training.
            # LogisticRegression cannot fit a single class, so use a constant
            # predictor at the observed success rate (0.0 or 1.0).
            classifiers[m.name] = _ConstantClassifier(float(labels[0]))
            if verbose:
                print(f"[scorer] {m.name}: single-class train "
                      f"(all={int(labels[0])}), using constant predictor")
            continue
        clf = LogisticRegression(max_iter=max_iter, C=C)
        clf.fit(X, labels)
        classifiers[m.name] = clf
        if verbose:
            print(f"[scorer] {m.name}: trained on {len(prompts)} rows, "
                  f"success rate {labels.mean():.1%}")

    return EmbeddingLRRouter(models, classifiers, embed_fn)
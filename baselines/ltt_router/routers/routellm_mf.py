"""
RouteLLM Matrix-Factorization scorer, wrapped as an LTT RoutingFunction.

We reuse RouteLLM's trained per-model quality score δ(M, q) — NOT its 2-model
routing wrapper. RouteLLM's pairwise winrate is σ(δ(a) − δ(b)); δ itself is
per model, so we read one δ per model. Raw δ has the right ORDERING but is not a
calibrated success probability, so this scorer is ALWAYS calibrated: per-model
Platt or isotonic maps δ -> P(model correct | query), fit on held-out data.

The vector [P(model_0 correct), ..., P(model_{N-1} correct)] is exactly the
RoutingFunction.score_batch output the LTT pipeline consumes. Everything below
the RoutingFunction boundary (split, Pareto, FST, metrics) is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from baselines.ltt_router.protocols import ModelSpec
from baselines.ltt_router.routers.embedding_lr import default_embed_fn, EmbedFn
from baselines.RouteLLM.routers.matrix_factorization.model import MODEL_IDS


# Calibration: map raw δ -> calibrated P(correct), per model.

class PlattCalibrator:
    """P = sigmoid(a·δ + b). Rigid 2-param S-curve; robust on little data."""
    def __init__(self, a: float, b: float):
        self.a, self.b = float(a), float(b)

    def __call__(self, delta: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(self.a * np.asarray(delta, float) + self.b)))


class IsotonicCalibrator:
    """Free monotone step function (sklearn IsotonicRegression), clipped [0,1]."""
    def __init__(self, iso):
        self._iso = iso

    def __call__(self, delta: np.ndarray) -> np.ndarray:
        return np.clip(self._iso.predict(np.asarray(delta, float)), 0.0, 1.0)


class ConstantCalibrator:
    """Constant success rate, for degenerate (single-class / sparse) models."""
    def __init__(self, p: float):
        self.p = float(p)

    def __call__(self, delta: np.ndarray) -> np.ndarray:
        return np.full(np.asarray(delta, float).shape, self.p, float)


def _fit_platt(delta: np.ndarray, y: np.ndarray) -> PlattCalibrator:
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=1000).fit(delta.reshape(-1, 1), y)
    return PlattCalibrator(lr.coef_[0, 0], lr.intercept_[0])


def _fit_isotonic(delta: np.ndarray, y: np.ndarray) -> IsotonicCalibrator:
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(delta, y)
    return IsotonicCalibrator(iso)


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Binned ECE: mean |confidence - accuracy| weighted by bin size."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    if len(p) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def _cv_ece(delta, y, fit_fn, n_splits=5, seed=0) -> float:
    """
    CROSS-VALIDATED ECE: fit on k-1 folds, score the held-out fold, average.
    This is the only honest way to compare Platt vs isotonic — isotonic's
    flexibility drives IN-SAMPLE ECE to ~0 by overfitting, which is misleading.
    """
    n = len(y)
    order = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(order, n_splits)
    eces = []
    for i in range(n_splits):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(n_splits) if j != i])
        if len(np.unique(y[tr])) < 2 or len(te) == 0:
            continue
        cal = fit_fn(delta[tr], y[tr])
        eces.append(expected_calibration_error(cal(delta[te]), y[te]))
    return float(np.mean(eces)) if eces else float("nan")


@dataclass
class ModelCalibration:
    name: str
    n: int
    pos_rate: float
    platt: Callable
    isotonic: Callable
    ece_raw: float        # ECE of sigmoid(δ), uncalibrated baseline
    ece_platt: float      # CROSS-VALIDATED
    ece_isotonic: float   # CROSS-VALIDATED
    degenerate: bool


def fit_model_calibration(delta, y, name, min_samples=50) -> ModelCalibration:
    """Fit Platt + isotonic for ONE model from held-out (δ, correct) pairs."""
    delta, y = np.asarray(delta, float), np.asarray(y, float)
    mask = ~np.isnan(delta)
    delta, y = delta[mask], y[mask]
    n = int(len(y))
    pos = float(y.mean()) if n else 0.0

    if (n < min_samples) or (len(np.unique(y)) < 2):
        const = ConstantCalibrator(pos if n else 0.5)
        return ModelCalibration(name, n, pos, const, const,
                                float("nan"), float("nan"), float("nan"), True)

    platt, iso = _fit_platt(delta, y), _fit_isotonic(delta, y)
    raw = 1.0 / (1.0 + np.exp(-delta))
    # Final calibrators fit on ALL data; reported ECEs are cross-validated.
    return ModelCalibration(
        name, n, pos, platt, iso,
        expected_calibration_error(raw, y),
        _cv_ece(delta, y, _fit_platt),
        _cv_ece(delta, y, _fit_isotonic),
        False,
    )


def choose_method(cals: Dict[str, ModelCalibration]) -> str:
    """Pick 'platt' or 'isotonic' by mean CV-ECE over non-degenerate models."""
    p = [c.ece_platt for c in cals.values() if not c.degenerate and not np.isnan(c.ece_platt)]
    i = [c.ece_isotonic for c in cals.values() if not c.degenerate and not np.isnan(c.ece_isotonic)]
    mp = float(np.mean(p)) if p else float("inf")
    mi = float(np.mean(i)) if i else float("inf")
    return "isotonic" if mi < mp else "platt"


def calibrator_map(cals: Dict[str, ModelCalibration], method: str) -> Dict[str, Callable]:
    pick = (lambda c: c.isotonic) if method == "isotonic" else (lambda c: c.platt)
    return {name: pick(c) for name, c in cals.items()}


# The scorer.

class RouteLLMMFRouter:
    """
    RoutingFunction backed by a trained MF model, ALWAYS calibrated.

    score_batch(prompts)     -> [G, N] calibrated P(correct). Unmapped models or
                                 models with no calibrator -> constant fallback.
    raw_score_batch(prompts) -> [G, N] raw δ (NaN for unmapped), used to FIT the
                                 calibrators offline.
    """

    def __init__(
        self,
        models: List[ModelSpec],
        P: np.ndarray,
        text_proj: Optional[np.ndarray],
        classifier: np.ndarray,
        embed_fn: EmbedFn,
        calibrators: Optional[Dict[str, Callable]] = None,
        fallback_score: float = 0.5,
    ):
        self._models = models
        self._P = np.asarray(P, float)
        self._proj = None if text_proj is None else np.asarray(text_proj, float)
        self._clf = np.asarray(classifier, float).reshape(-1)
        self._embed_fn = embed_fn
        self._calibrators = dict(calibrators) if calibrators else {}
        self._fallback = float(fallback_score)
        self._row = {m.index: MODEL_IDS.get(m.name) for m in models}

    @property
    def models(self) -> Sequence[ModelSpec]:
        return self._models

    def _project(self, prompts):
        X = np.asarray(self._embed_fn(prompts), float)
        return X @ self._proj.T if self._proj is not None else X

    def _delta(self, Xp, row):
        pe = self._P[row]
        pe = pe / (np.linalg.norm(pe) + 1e-12)
        return (Xp * pe) @ self._clf

    def raw_score_batch(self, prompts: List[str]) -> np.ndarray:
        Xp = self._project(prompts)
        out = np.full((len(prompts), len(self._models)), np.nan, float)
        for m in self._models:
            row = self._row.get(m.index)
            if row is not None:
                out[:, m.index] = self._delta(Xp, row)
        return out

    def score_batch(self, prompts: List[str]) -> np.ndarray:
        Xp = self._project(prompts)
        out = np.full((len(prompts), len(self._models)), self._fallback, float)
        for m in self._models:
            row = self._row.get(m.index)
            if row is None:
                continue
            delta = self._delta(Xp, row)
            cal = self._calibrators.get(m.name)
            if cal is None:
                # No calibrator: should not happen in the built-in-calibration
                # path, but stay safe with sigmoid(δ) rather than crash.
                out[:, m.index] = 1.0 / (1.0 + np.exp(-delta))
            else:
                out[:, m.index] = np.clip(cal(delta), 0.0, 1.0)
        return out

    def score(self, prompt: str, dataset_id: str = "") -> np.ndarray:
        return self.score_batch([prompt])[0]


def _load_mf_state(checkpoint_path: str) -> dict:
    import torch
    sd = torch.load(checkpoint_path, map_location="cpu")
    out = {}
    for k, v in sd.items():
        out[k] = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
    return out


def build_routellm_mf_router(
    models: List[ModelSpec],
    checkpoint_path: str,
    calibrators: Dict[str, Callable],
    embed_fn: Optional[EmbedFn] = None,
    fallback_score: float = 0.5,
    *,
    _state: Optional[dict] = None,
) -> RouteLLMMFRouter:
    """
    Load a trained MF checkpoint + per-model calibrators into a RoutingFunction.

    calibrators is REQUIRED (calibration is built-in). Fit them first with
    mf_pipeline.py fit-calibrators.
    """
    if embed_fn is None:
        embed_fn = default_embed_fn
    state = _state if _state is not None else _load_mf_state(checkpoint_path)

    P = state.get("P.weight")
    clf = state.get("classifier.weight")
    if P is None or clf is None:
        raise KeyError("Checkpoint missing 'P.weight' or 'classifier.weight'.")
    proj = state.get("text_proj.weight")  # None when use_proj=False

    return RouteLLMMFRouter(
        models=models, P=P, text_proj=proj, classifier=clf,
        embed_fn=embed_fn, calibrators=calibrators, fallback_score=fallback_score,
    )
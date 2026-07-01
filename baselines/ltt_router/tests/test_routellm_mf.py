"""
Tests for the RouteLLM MF scorer and its built-in calibration.

  1. Calibrators map δ -> [0,1] and each type behaves correctly.
  2. fit_model_calibration reports CROSS-VALIDATED ECE (isotonic is NOT a fake
     0.0 — that was the overfitting trap), and improves on the raw baseline.
  3. Degenerate (single-class / sparse) models fall back to a constant.
  4. choose_method picks the lower mean-CV-ECE method; calibrator_map extracts it.
  5. The scorer produces protocol-valid [G, N] scores in [0,1], applies
     calibrators, and falls back for models absent from RouteLLM's MODEL_IDS.
  6. raw_score_batch exposes δ (NaN for unmapped models) for fitting.

Weights are injected via build_routellm_mf_router(_state=...), so these tests
need no trained checkpoint and no downloads.
"""

import numpy as np
import pytest

from baselines.ltt_router.protocols import ModelSpec, RoutingFunction
from baselines.ltt_router.routers.routellm_mf import (
    PlattCalibrator,
    IsotonicCalibrator,
    ConstantCalibrator,
    expected_calibration_error,
    fit_model_calibration,
    choose_method,
    calibrator_map,
    build_routellm_mf_router,
)
# real names so MODEL_IDS maps them; the scorer reads δ for these.
from baselines.RouteLLM.routers.matrix_factorization.model import MODEL_NAMES


# calibrators

def test_calibrator_types_map_to_unit_interval():
    d = np.linspace(-4, 4, 50)
    assert np.all((PlattCalibrator(1.0, 0.0)(d) >= 0) & (PlattCalibrator(1.0, 0.0)(d) <= 1))
    const = ConstantCalibrator(0.3)
    assert np.allclose(const(d), 0.3)
    # Platt is monotone increasing in δ when a>0
    p = PlattCalibrator(2.0, 0.0)(d)
    assert np.all(np.diff(p) >= -1e-9)


def test_ece_zero_for_perfect_predictions():
    y = np.array([0, 0, 1, 1], float)
    perfect = y.copy()
    assert expected_calibration_error(perfect, y) == pytest.approx(0.0, abs=1e-9)


# fit_model_calibration

def _miscalibrated(n=4000, seed=0):
    """δ with the right ordering but a shifted/scaled true success curve."""
    rng = np.random.default_rng(seed)
    d = rng.normal(0, 2, n)
    true_p = 1.0 / (1.0 + np.exp(-(0.4 * d - 0.5)))
    y = (rng.uniform(size=n) < true_p).astype(float)
    return d, y


def test_calibration_improves_on_raw_and_cv_ece_is_honest():
    d, y = _miscalibrated()
    mc = fit_model_calibration(d, y, "m")
    assert not mc.degenerate
    # calibration beats the raw sigmoid(δ) baseline
    assert mc.ece_platt < mc.ece_raw
    # CV-ECE for isotonic must be a real positive number, NOT a fake in-sample 0
    assert mc.ece_isotonic > 1e-4


def test_degenerate_model_falls_back_to_constant():
    d, _ = _miscalibrated()
    mc = fit_model_calibration(d, np.ones(len(d)), "always_right")
    assert mc.degenerate
    assert mc.platt(np.array([0.0, 5.0]))[0] == pytest.approx(1.0)
    # too few samples is also degenerate
    mc2 = fit_model_calibration(np.array([0.1, 0.2]), np.array([0.0, 1.0]), "sparse")
    assert mc2.degenerate


def test_choose_method_and_calibrator_map():
    d, y = _miscalibrated()
    cals = {"m0": fit_model_calibration(d, y, "m0"),
            "m1": fit_model_calibration(*_miscalibrated(seed=1), "m1")}
    method = choose_method(cals)
    assert method in ("platt", "isotonic")
    cmap = calibrator_map(cals, method)
    assert set(cmap) == {"m0", "m1"}
    # each entry is a usable calibrator callable
    assert np.all((cmap["m0"](d) >= 0) & (cmap["m0"](d) <= 1))


# scorer

def _toy_scorer(dim=8, seed=0):
    """
    Build a RouteLLMMFRouter from injected weights. Models 0..2 use real
    MODEL_NAMES (mapped); model 3 uses a fake name (unmapped -> fallback).
    """
    rng = np.random.default_rng(seed)
    n_real = 33
    state = {
        "P.weight": rng.normal(size=(n_real, dim)),
        "classifier.weight": rng.normal(size=(1, dim)),
        "Q.weight": rng.normal(size=(10, dim)),  # training-only, must be ignored
    }
    models = [
        ModelSpec(name=MODEL_NAMES[0], cost=1.0, index=0),
        ModelSpec(name=MODEL_NAMES[1], cost=2.0, index=1),
        ModelSpec(name=MODEL_NAMES[2], cost=3.0, index=2),
        ModelSpec(name="not_a_real_model", cost=4.0, index=3),
    ]
    cals = {m.name: PlattCalibrator(1.0, 0.0) for m in models[:3]}

    def embed_fn(prompts):
        return np.array([[(hash(p + str(k)) % 97) / 97 for k in range(dim)]
                         for p in prompts], float)

    scorer = build_routellm_mf_router(
        models, "unused.pt", calibrators=cals, embed_fn=embed_fn, _state=state)
    return scorer, models


def test_scorer_is_routing_function():
    scorer, _ = _toy_scorer()
    assert isinstance(scorer, RoutingFunction)


def test_score_batch_shape_and_range():
    scorer, models = _toy_scorer()
    S = scorer.score_batch(["p1", "p2", "p3"])
    assert S.shape == (3, len(models))
    assert np.all((S >= 0.0) & (S <= 1.0))


def test_unmapped_model_gets_fallback():
    scorer, _ = _toy_scorer()
    S = scorer.score_batch(["p1", "p2"])
    # index 3 is the fake/unmapped model -> constant fallback 0.5
    assert np.allclose(S[:, 3], 0.5)
    # mapped models are not all-fallback
    assert not np.allclose(S[:, 0], 0.5)


def test_raw_score_batch_exposes_delta_with_nan_for_unmapped():
    scorer, _ = _toy_scorer()
    raw = scorer.raw_score_batch(["p1", "p2", "p3"])
    assert raw.shape == (3, 4)
    assert np.isnan(raw[:, 3]).all()          # unmapped -> NaN
    assert not np.isnan(raw[:, 0]).any()      # mapped -> real δ


def test_single_score_matches_batch_row():
    scorer, _ = _toy_scorer()
    multi = scorer.score_batch(["only", "second"])
    one = scorer.score_batch(["only"])
    assert one.shape == (1, multi.shape[1])
    assert np.allclose(one[0], multi[0])


def test_build_requires_valid_checkpoint_keys():
    rng = np.random.default_rng(0)
    bad = {"classifier.weight": rng.normal(size=(1, 8))}  # missing P.weight
    with pytest.raises(KeyError):
        build_routellm_mf_router(
            [ModelSpec(MODEL_NAMES[0], 1.0, 0)], "x", calibrators={},
            embed_fn=lambda p: np.zeros((len(p), 8)), _state=bad)
"""Analytic unit tests for the AUC tie handling in ``ModelValidation._calculate_auc``.

These tests need no R and no model inputs. They exist because the shipped BPC3 /
iCARE-Lit validation cohorts have an essentially tie-free linear predictor, so the
0.5-tie fix is a no-op on those fixtures -- only a deliberately tie-heavy example
proves the Mann-Whitney / trapezoidal tie credit (1.0 / 0.5 / 0.0) actually works.

We drive ``_calculate_auc`` directly via ``ModelValidation.__new__`` so we can build
tiny tie-heavy frames without supplying full model inputs. (The percentile binning no
longer raises on tied/small inputs -- see ``tests/test_hl_binning.py`` -- but the
``__new__`` shortcut still keeps these AUC unit tests free of model files.)
"""
import numpy as np
import pandas as pd
import pytest

from icare.model_validation import ModelValidation, ModelValidationResults


def _auc(study_data: pd.DataFrame, nested: bool) -> float:
    """Exercise the real ``_calculate_auc`` in isolation and return the AUC."""
    mv = ModelValidation.__new__(ModelValidation)
    mv.study_data = study_data
    mv.nested_case_control_study = nested
    mv.linear_predictor_variable_name = "linear_predictors"
    mv.results = ModelValidationResults()
    mv._calculate_auc()
    return mv.results.auc["auc"]


def test_auc_full_cohort_credits_ties_one_half():
    # cases [3, 2, 2] vs controls [2, 1]:
    #   3>2, 3>1, 2==2 -> .5, 2>1, 2==2 -> .5, 2>1  =>  (1+1+0.5+1+0.5+1)/6 = 5/6
    # The old strict-`>` estimator (no tie credit) would give 4/6 instead.
    study = pd.DataFrame({
        "observed_outcome": [1, 1, 1, 0, 0],
        "linear_predictors": [3.0, 2.0, 2.0, 2.0, 1.0],
    })
    assert _auc(study, nested=False) == pytest.approx(5 / 6)


def test_auc_nested_case_control_weighted_credits_ties_one_half():
    # cases scores [2, 1] with freq [2, 1]; controls [2, 1, 0] with freq [2, 4, 1].
    # Tie-aware IPW numerator = 15, denominator = 21 -> 15/21.
    # The old strict-`>` estimator would give 11/21 instead.
    # (freq = 1 / sampling_weights, so sampling_weights are the reciprocals below.)
    study = pd.DataFrame({
        "observed_outcome": [1, 1, 0, 0, 0],
        "linear_predictors": [2.0, 1.0, 2.0, 1.0, 0.0],
        "sampling_weights": [0.5, 1.0, 0.5, 0.25, 1.0],
    })
    study["frequency"] = 1.0 / study["sampling_weights"]
    assert _auc(study, nested=True) == pytest.approx(15 / 21)


def test_auc_tie_tolerance_band():
    # Floating-point jitter (1e-12, below AUC_TIE_TOLERANCE) counts as a tie -> 0.5;
    # a genuine 1e-3 gap (far above the tolerance) is a clear win -> 1.0.
    jitter = pd.DataFrame({"observed_outcome": [1, 0], "linear_predictors": [1.0 + 1e-12, 1.0]})
    gap = pd.DataFrame({"observed_outcome": [1, 0], "linear_predictors": [1.0 + 1e-3, 1.0]})
    assert _auc(jitter, nested=False) == pytest.approx(0.5)
    assert _auc(gap, nested=False) == pytest.approx(1.0)

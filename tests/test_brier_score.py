"""Unit tests for the overall Brier score in ``ModelValidation``.

These need no R and no model inputs. They drive ``ModelValidation`` internals via ``__new__``
(as in ``test_category_specific_eo_ratio.py`` / ``test_auc_ties.py``) on tiny hand-built
frames, and check the overall Brier score, its variance, and its 95% Wald confidence interval.

The Brier score is the mean squared difference between the predicted risk and the observed
outcome, ``mean((risk - outcome) ** 2)``. For nested case-control studies it is the
inverse-probability-weighted (frequency-weighted) mean, and its variance mirrors the
superpopulation + sampling-design decomposition already used for the overall E/O ratio's
observed-risk variance (with the Bernoulli ``observed * (1 - observed)`` term generalized to
the weighted sample variance of the Brier contributions). The confidence interval is a direct
(symmetric) Wald interval, consistent with the AUC CI.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from icare.model_validation import ModelValidation, ModelValidationResults

# Matches the hardcoded 95% z in ``wald_confidence_interval``.
Z = 1.96


def _unweighted_mv(study: pd.DataFrame) -> ModelValidation:
    mv = ModelValidation.__new__(ModelValidation)
    mv.nested_case_control_study = False
    mv.results = ModelValidationResults()
    mv.study_data = study
    return mv


def _weighted_mv(study: pd.DataFrame) -> ModelValidation:
    """Nested case-control ModelValidation. ``frequency = 1 / sampling_weights`` (as in
    ``_set_study_data``) is materialized here since ``__new__`` bypasses that setup."""
    mv = ModelValidation.__new__(ModelValidation)
    mv.nested_case_control_study = True
    mv.results = ModelValidationResults()
    study = study.copy()
    study["frequency"] = 1.0 / study["sampling_weights"]
    mv.study_data = study
    return mv


def test_cohort_brier_score_and_ci_match_manual_computation():
    """Cohort Brier score == mean((risk - outcome) ** 2); variance == np.var(contributions) / n
    (population variance, ddof=0); CI == the symmetric Wald interval BS +/- Z * sqrt(variance)."""
    risk = np.array([0.2, 0.8, 0.5, 0.9, 0.1])
    outcome = np.array([0, 1, 1, 0, 0])
    study = pd.DataFrame({"risk_estimates": risk, "observed_outcome": outcome})

    mv = _unweighted_mv(study)
    mv._calculate_brier_score()
    result = mv.results.brier_score

    contributions = (risk - outcome) ** 2  # [0.04, 0.04, 0.25, 0.81, 0.01]
    n = len(contributions)
    expected_bs = contributions.mean()
    expected_var = np.var(contributions) / n
    expected_stddev = np.sqrt(expected_var)

    assert expected_bs == pytest.approx(0.23)  # 1.15 / 5, ground-truth arithmetic
    assert result["brier_score"] == pytest.approx(expected_bs)
    assert result["variance"] == pytest.approx(expected_var)
    assert result["lower_ci"] == pytest.approx(expected_bs - Z * expected_stddev)
    assert result["upper_ci"] == pytest.approx(expected_bs + Z * expected_stddev)


def test_nested_brier_score_uses_sampling_weights():
    """The nested/weighted path must use the frequency-weighted mean of the Brier contributions,
    and the variance must be the superpopulation (weighted sample variance) + sampling-design
    correction decomposition."""
    study = pd.DataFrame({
        "risk_estimates": [0.6, 0.6, 0.4, 0.4],
        "observed_outcome": [1, 0, 1, 0],
        "sampling_weights": [0.25, 0.5, 0.5, 0.25],  # frequency = 4, 2, 2, 4
    })

    mv = _weighted_mv(study)
    mv._calculate_brier_score()
    result = mv.results.brier_score

    risk = study["risk_estimates"].to_numpy()
    outcome = study["observed_outcome"].to_numpy()
    sampling_weights = study["sampling_weights"].to_numpy()
    frequency = 1.0 / sampling_weights
    contributions = (risk - outcome) ** 2  # [0.16, 0.36, 0.36, 0.16]
    total_weight = frequency.sum()

    weighted_bs = np.sum(frequency * contributions) / total_weight
    # the weights genuinely change the score (else the test proves nothing)
    assert not np.isclose(weighted_bs, contributions.mean())

    centered = contributions - weighted_bs
    superpopulation = np.sum(frequency * centered ** 2) / total_weight
    design_correction = np.sum(centered ** 2 * (1 - sampling_weights) / sampling_weights ** 2) / total_weight
    expected_var = (superpopulation + design_correction) / total_weight
    expected_stddev = np.sqrt(expected_var)

    assert result["brier_score"] == pytest.approx(weighted_bs)
    assert result["variance"] == pytest.approx(expected_var)
    assert result["lower_ci"] == pytest.approx(weighted_bs - Z * expected_stddev)
    assert result["upper_ci"] == pytest.approx(weighted_bs + Z * expected_stddev)


def test_weighted_reduces_to_cohort_when_weights_uniform():
    """With uniform sampling weights (all 1.0 -> frequency 1), the design correction vanishes and
    the weighted Brier score, variance, and CI equal the cohort path exactly."""
    risk = [0.15, 0.7, 0.35, 0.9, 0.05, 0.6]
    outcome = [0, 1, 0, 1, 0, 1]

    cohort = _unweighted_mv(pd.DataFrame({"risk_estimates": risk, "observed_outcome": outcome}))
    cohort._calculate_brier_score()

    weighted = _weighted_mv(pd.DataFrame({
        "risk_estimates": risk,
        "observed_outcome": outcome,
        "sampling_weights": [1.0] * len(risk),
    }))
    weighted._calculate_brier_score()

    for key in ("brier_score", "variance", "lower_ci", "upper_ci"):
        assert weighted.results.brier_score[key] == pytest.approx(cohort.results.brier_score[key])


def test_perfect_model_gives_zero_brier_score():
    """When predicted risk equals the observed outcome for everyone, the Brier score, variance,
    and both CI bounds are exactly zero -- the direct Wald interval is well-defined at the
    boundary and emits no warnings."""
    study = pd.DataFrame({
        "risk_estimates": [0.0, 1.0, 1.0, 0.0],
        "observed_outcome": [0, 1, 1, 0],
    })

    mv = _unweighted_mv(study)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mv._calculate_brier_score()
    result = mv.results.brier_score

    assert result["brier_score"] == 0.0
    assert result["variance"] == 0.0
    assert result["lower_ci"] == 0.0
    assert result["upper_ci"] == 0.0

"""Unit tests for the bin-specific expected-by-observed (E/O) ratio in ``ModelValidation``.

These need no R and no model inputs. They drive ``ModelValidation`` internals via ``__new__``
(as in ``test_hl_binning.py`` / ``test_auc_ties.py``) on tiny hand-built frames, and check the
per-category E/O ratio and its 95% CI that are appended to ``category_specific_calibration``.

The per-bin E/O ratio is the per-category analog of the overall E/O ratio: ``predicted / observed``
per bin, with a log-Wald CI that treats the expected (predicted) risk as fixed. It reuses the same
weight-aware per-bin observed-rate variance already used for the absolute-risk calibration CI, so
the E/O CI is internally consistent with it (several tests exploit that to recover the variance from
the absolute-risk CI columns without re-implementing the weighted variance formula).
"""
import io
import json
import warnings

import numpy as np
import pandas as pd
import pytest

from icare.model_validation import (
    ModelValidation,
    ModelValidationResults,
    calculate_expected_by_observed_ratio_per_category,
)

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


def test_cohort_eo_ratio_and_ci_match_manual_computation():
    """Ground truth: per-bin ratio == predicted/observed and CI == the log-Wald interval built
    from the per-bin binomial variance observed*(1-observed)/n."""
    values = (1.0, 2.0, 3.0, 4.0)
    events = (2, 4, 6, 8)
    risks = (0.15, 0.35, 0.55, 0.85)
    n = 10
    lp, outcome, risk = [], [], []
    for v, e, r in zip(values, events, risks):
        lp += [v] * n
        outcome += [1] * e + [0] * (n - e)
        risk += [r] * n
    study = pd.DataFrame(
        {"linear_predictors": lp, "observed_outcome": outcome, "risk_estimates": risk}
    )

    mv = _unweighted_mv(study)
    mv._categorize_risk_scores([1.5, 2.5, 3.5], 4)  # explicit cutoffs -> 4 deterministic bins
    mv._calculate_calibration()
    df = mv.results.category_specific_calibration

    observed = np.array([e / n for e in events])
    predicted = np.array(risks)
    variance = observed * (1 - observed) / n
    ratio = predicted / observed
    se_log = np.sqrt(variance) / observed
    exp_lower = np.exp(np.log(ratio) - Z * se_log)
    exp_upper = np.exp(np.log(ratio) + Z * se_log)

    # rows are sorted by category ascending, i.e. the values (1, 2, 3, 4) order
    np.testing.assert_allclose(df["observed_absolute_risk"].to_numpy(), observed, atol=1e-10)
    np.testing.assert_allclose(df["predicted_absolute_risk"].to_numpy(), predicted, atol=1e-10)
    np.testing.assert_allclose(df["expected_by_observed_ratio"].to_numpy(), ratio, atol=1e-10)
    np.testing.assert_allclose(df["lower_ci_expected_by_observed_ratio"].to_numpy(), exp_lower, atol=1e-10)
    np.testing.assert_allclose(df["upper_ci_expected_by_observed_ratio"].to_numpy(), exp_upper, atol=1e-10)


def test_single_bin_eo_ratio_reduces_to_overall_cohort():
    """With one bin holding the whole cohort, the per-bin E/O ratio and CI equal the overall
    E/O ratio and CI exactly (the helper fed whole-cohort inputs == the overall formula)."""
    rng = np.random.default_rng(0)
    n = 200
    outcome = rng.integers(0, 2, size=n)
    risk = rng.uniform(0.05, 0.95, size=n)
    study = pd.DataFrame(
        {
            "linear_predictors": rng.normal(size=n),
            "observed_outcome": outcome,
            "risk_estimates": risk,
        }
    )

    mv = _unweighted_mv(study)
    mv._calculate_expected_by_observed_ratio()
    overall = mv.results.expected_by_observed_ratio

    observed_mean = outcome.mean()
    ratio, lower, upper = calculate_expected_by_observed_ratio_per_category(
        np.array([observed_mean]),
        np.array([risk.mean()]),
        np.array([observed_mean * (1 - observed_mean) / n]),
    )

    assert ratio[0] == pytest.approx(overall["ratio"])
    assert lower[0] == pytest.approx(overall["lower_ci"])
    assert upper[0] == pytest.approx(overall["upper_ci"])


def test_nested_eo_ratio_uses_sampling_weights():
    """The nested/weighted path must use frequency-weighted observed and predicted rates, and its
    E/O CI must reuse the same weight-aware per-bin variance as the absolute-risk CI."""
    study = pd.DataFrame(
        {
            "linear_predictors": [1.0, 1.0, 2.0, 2.0],
            "observed_outcome": [1, 0, 1, 0],
            "risk_estimates": [0.6, 0.6, 0.4, 0.4],
            "sampling_weights": [0.25, 0.5, 0.5, 0.25],  # frequency = 4, 2, 2, 4
        }
    )

    mv = _weighted_mv(study)
    mv._categorize_risk_scores([1.5], 2)  # deterministic split between the two scores
    mv._calculate_calibration()
    df = mv.results.category_specific_calibration
    assert len(df) == 2

    # bin A (lp=1.0): freq 4, 2 ; bin B (lp=2.0): freq 2, 4
    weighted_observed = np.array([(1 * 4 + 0 * 2) / 6, (1 * 2 + 0 * 4) / 6])
    weighted_predicted = np.array([0.6, 0.4])
    unweighted_observed = np.array([0.5, 0.5])

    np.testing.assert_allclose(df["observed_absolute_risk"].to_numpy(), weighted_observed, atol=1e-10)
    # the weights genuinely change the observed rate (else the test proves nothing)
    assert not np.allclose(df["observed_absolute_risk"].to_numpy(), unweighted_observed)

    expected_ratio = weighted_predicted / weighted_observed
    np.testing.assert_allclose(df["expected_by_observed_ratio"].to_numpy(), expected_ratio, atol=1e-10)

    # Recover the per-bin observed-rate stddev from the absolute-risk CI (= observed +/- Z*stddev)
    # and confirm the E/O CI is the log-Wald interval built from that same stddev.
    observed = df["observed_absolute_risk"].to_numpy()
    stddev = (df["upper_ci_absolute_risk"].to_numpy() - observed) / Z
    se_log = stddev / observed
    exp_lower = np.exp(np.log(expected_ratio) - Z * se_log)
    exp_upper = np.exp(np.log(expected_ratio) + Z * se_log)
    np.testing.assert_allclose(df["lower_ci_expected_by_observed_ratio"].to_numpy(), exp_lower, atol=1e-10)
    np.testing.assert_allclose(df["upper_ci_expected_by_observed_ratio"].to_numpy(), exp_upper, atol=1e-10)


def test_helper_degenerate_bins_are_nan_without_warnings():
    """Zero observed events (or zero predicted risk) -> NaN ratio and CI, and no RuntimeWarning
    leaks out of the helper."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        ratio, lower, upper = calculate_expected_by_observed_ratio_per_category(
            np.array([0.0, 0.5, 0.3]),  # bin 0: no observed events
            np.array([0.1, 0.4, 0.0]),  # bin 2: no predicted risk
            np.array([0.0, 0.01, 0.02]),
        )

    for idx in (0, 2):
        assert np.isnan(ratio[idx]) and np.isnan(lower[idx]) and np.isnan(upper[idx])
    assert np.isfinite([ratio[1], lower[1], upper[1]]).all()


def test_cohort_zero_observed_bin_serializes_to_null():
    """An empty (zero-observed) bin yields NaN in the E/O columns, which serializes to JSON null
    via the same ``to_json(orient='records')`` path used in ``package_validation_results_to_dict``."""
    lp = [1.0] * 10 + [2.0] * 10 + [3.0] * 10
    outcome = [0] * 10 + ([1] * 4 + [0] * 6) + ([1] * 7 + [0] * 3)  # lowest bin has zero events
    risk = [0.1] * 10 + [0.35] * 10 + [0.75] * 10
    study = pd.DataFrame(
        {"linear_predictors": lp, "observed_outcome": outcome, "risk_estimates": risk}
    )

    mv = _unweighted_mv(study)
    mv._categorize_risk_scores([1.5, 2.5], 3)
    with warnings.catch_warnings():
        # the existing HL chi-square divides by the zero-variance empty bin; that pre-existing
        # RuntimeWarning is unrelated to the E/O computation.
        warnings.simplefilter("ignore", RuntimeWarning)
        mv._calculate_calibration()
    df = mv.results.category_specific_calibration

    eo = df["expected_by_observed_ratio"].to_numpy()
    assert np.isnan(eo[0])
    assert np.isfinite(eo[1:]).all()

    records = json.loads(df.to_json(orient="records"))
    assert records[0]["expected_by_observed_ratio"] is None
    assert records[0]["lower_ci_expected_by_observed_ratio"] is None
    assert records[0]["upper_ci_expected_by_observed_ratio"] is None
    # a well-populated bin round-trips to a finite number
    assert records[1]["expected_by_observed_ratio"] is not None


def test_eo_columns_align_with_realized_bins_when_bins_merge():
    """When tied scores merge bins below the requested count, the E/O columns have exactly the
    realized number of rows and stay aligned with observed/predicted per bin."""
    values = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    events = (1, 2, 3, 4, 5, 2)
    lp, outcome, risk = [], [], []
    for v, e in zip(values, events):
        lp += [v] * 6
        outcome += [1] * e + [0] * (6 - e)
        risk += [0.9 * e / 6] * 6
    study = pd.DataFrame(
        {"linear_predictors": lp, "observed_outcome": outcome, "risk_estimates": risk}
    )

    mv = _unweighted_mv(study)
    mv._categorize_risk_scores(None, 10)  # 6 distinct scores -> bins merge below 10
    mv._calculate_calibration()
    df = mv.results.category_specific_calibration

    k = mv.study_data["linear_predictors_category"].nunique()
    assert k < 10
    assert len(df) == k
    for col in (
        "expected_by_observed_ratio",
        "lower_ci_expected_by_observed_ratio",
        "upper_ci_expected_by_observed_ratio",
    ):
        assert col in df.columns
        assert len(df[col]) == k

    np.testing.assert_allclose(
        df["expected_by_observed_ratio"].to_numpy(),
        df["predicted_absolute_risk"].to_numpy() / df["observed_absolute_risk"].to_numpy(),
        atol=1e-10,
    )

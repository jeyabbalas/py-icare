"""Unit tests for the Hosmer-Lemeshow risk-score binning in ``ModelValidation``.

These need no R and no model inputs. They cover the fix for the unweighted calibration
path, which previously called ``pd.qcut(..., q=number_of_percentiles)`` with pandas'
default ``duplicates='raise'`` and so **crashed** on a coarse/tied linear predictor whose
quantile edges collide -- where R's ``quantcut`` degrades gracefully. The unweighted path
now routes through the same dups-aware ``weighted_quantcut`` as the weighted path, and the
calibration degrees of freedom / RR derivative use the *realized* bin count (which can be
fewer than ``number_of_percentiles`` once tied bins merge).

We drive ``ModelValidation`` internals via ``__new__`` (as in ``test_auc_ties.py``) to
build tiny tie-heavy frames without supplying full model files.
"""
import numpy as np
import pandas as pd
import pytest

from icare.model_validation import ModelValidation, ModelValidationResults, weighted_quantcut


# A coarse score: 6 distinct values, so 10 requested quantile edges must collide.
TIED_SCORES = pd.Series([v for v in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0) for _ in range(6)])


def test_pd_qcut_would_have_raised_on_tied_scores():
    """Pins the old failure mode: the pre-fix unweighted path raised on this input."""
    with pytest.raises(ValueError):
        pd.qcut(TIED_SCORES, q=10)


def test_weighted_quantcut_unweighted_degrades_instead_of_raising():
    result = weighted_quantcut(TIED_SCORES, None, 10)
    assert len(result) == len(TIED_SCORES)
    assert result.notna().all()          # every subject assigned a bin
    assert result.nunique() < 10         # tied bins merged, like R's quantcut


def test_weighted_quantcut_weighted_degrades_instead_of_raising():
    weights = pd.Series(np.full(len(TIED_SCORES), 2.0))  # frequency = 1 / sampling_weight
    result = weighted_quantcut(TIED_SCORES, weights, 10)
    assert len(result) == len(TIED_SCORES)
    assert result.notna().all()
    assert result.nunique() < 10


def test_weighted_quantcut_matches_qcut_membership_when_tie_free():
    """The unification must not move any subject when there are no ties."""
    x = pd.Series(np.random.default_rng(0).normal(size=500))
    via_helper = weighted_quantcut(x, None, 10)
    via_qcut = pd.qcut(x, q=10)
    assert via_helper.nunique() == 10
    assert (via_helper.cat.codes.values == via_qcut.cat.codes.values).all()


def _unweighted_mv(study: pd.DataFrame) -> ModelValidation:
    mv = ModelValidation.__new__(ModelValidation)
    mv.nested_case_control_study = False
    mv.results = ModelValidationResults()
    mv.study_data = study
    return mv


def test_calibration_df_uses_realized_bin_count_when_bins_merge():
    """When tied scores merge bins below 10, df_ar == k and df_rr == k - 1 (not 10 / 9)."""
    values = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    events = (1, 2, 3, 4, 5, 2)  # events out of 6 per value-group => observed prob in (0, 1)
    lp, outcome, risk = [], [], []
    for v, e in zip(values, events):
        lp += [v] * 6
        outcome += [1] * e + [0] * (6 - e)
        risk += [0.9 * e / 6] * 6  # predicted near (but != ) observed, so chi-square > 0
    study = pd.DataFrame({
        "linear_predictors": lp,
        "observed_outcome": outcome,
        "risk_estimates": risk,
    })

    mv = _unweighted_mv(study)
    mv._categorize_risk_scores(None, 10)
    mv._calculate_calibration()

    k = mv.study_data["linear_predictors_category"].nunique()
    assert k < 10  # confirms the tie path was exercised

    ar = mv.results.calibration["absolute_risk"]
    rr = mv.results.calibration["relative_risk"]
    assert ar["parameter"]["degrees_of_freedom"] == k
    assert rr["parameter"]["degrees_of_freedom"] == k - 1
    assert np.isfinite(ar["statistic"]["chi_square"]) and ar["statistic"]["chi_square"] > 0
    assert np.isfinite(rr["statistic"]["chi_square"])

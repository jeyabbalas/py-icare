"""Fast smoke tests: does py-icare import and run under the upgraded dependency
stack (numpy 2.4, pandas 3.0, patsy 1.0, scipy 1.18)?

These are the first line of defence against dependency-induced breakage and run
in well under a second each (the detailed numerical cross-validation against R
lives in test_bpc3_* and test_icare_lit_*). They also assert that the library's
documented output-reading idiom works on pandas 3.0, where ``pd.read_json`` no
longer accepts a raw JSON string.
"""

import io

import pandas as pd
import pytest

import icare
from icare_test_utils import BPC3_DIR, FIXTURES_DIR, ICARE_LIT_DIR

DISEASE_INC = str(BPC3_DIR / "age_specific_breast_cancer_incidence_rates.csv")
COMPETING_INC = str(BPC3_DIR / "age_specific_all_cause_mortality_rates.csv")
FORMULA = str(BPC3_DIR / "breast_cancer_covariate_model_formula.txt")
LOG_OR = str(BPC3_DIR / "breast_cancer_model_log_odds_ratios.json")
LOG_OR_POST50 = str(BPC3_DIR / "breast_cancer_model_log_odds_ratios_post_50.json")
REFERENCE = str(BPC3_DIR / "reference_covariate_data.csv")
REFERENCE_POST50 = str(BPC3_DIR / "reference_covariate_data_post_50.csv")
QUERY_COV = str(BPC3_DIR / "query_covariate_profile.csv")


def test_import_and_public_api():
    assert icare.__version__
    for name in ("compute_absolute_risk", "compute_absolute_risk_split_interval",
                 "validate_absolute_risk_model"):
        assert callable(getattr(icare, name))


def test_compute_absolute_risk_runs():
    result = icare.compute_absolute_risk(
        apply_age_start=50, apply_age_interval_length=30,
        model_disease_incidence_rates_path=DISEASE_INC,
        model_competing_incidence_rates_path=COMPETING_INC,
        model_covariate_formula_path=FORMULA,
        model_log_relative_risk_path=LOG_OR,
        model_reference_dataset_path=REFERENCE,
        apply_covariate_profile_path=QUERY_COV,
    )
    assert set(result) >= {"model", "profile", "method"}
    # The documented idiom must work under pandas 3.0 (regression guard for the fix).
    profile = pd.read_json(io.StringIO(result["profile"]), orient="records")
    assert "risk_estimates" in profile.columns
    assert (profile["risk_estimates"].between(0, 1)).all()


def test_compute_absolute_risk_split_interval_runs():
    """Regression guard for the pandas-3 pd.read_json(<str>) fix in misc.py."""
    result = icare.compute_absolute_risk_split_interval(
        apply_age_start=30, apply_age_interval_length=40, cutpoint=50,
        model_disease_incidence_rates_path=DISEASE_INC,
        model_competing_incidence_rates_path=COMPETING_INC,
        model_covariate_formula_before_cutpoint_path=FORMULA,
        model_covariate_formula_after_cutpoint_path=FORMULA,
        model_log_relative_risk_before_cutpoint_path=LOG_OR,
        model_log_relative_risk_after_cutpoint_path=LOG_OR_POST50,
        model_reference_dataset_before_cutpoint_path=REFERENCE,
        model_reference_dataset_after_cutpoint_path=REFERENCE_POST50,
        apply_covariate_profile_before_cutpoint_path=QUERY_COV,
        apply_covariate_profile_after_cutpoint_path=QUERY_COV,
    )
    profile = pd.read_json(io.StringIO(result["profile"]), orient="records")
    assert "risk_estimates" in profile.columns
    assert (profile["risk_estimates"].between(0, 1)).all()


def test_validate_absolute_risk_model_runs(tmp_path):
    """Regression guard for the pandas-3 Categorical fix in model_validation.py.

    Uses a small slice of the iCARE-Lit validation cohort to stay fast.
    """
    study = pd.read_csv(FIXTURES_DIR / "icare_lit_validation_study.csv").head(300)
    covariates = pd.read_csv(FIXTURES_DIR / "icare_lit_validation_covariates.csv").head(300)
    study_path = tmp_path / "smoke_study.csv"
    cov_path = tmp_path / "smoke_covariates.csv"
    study.to_csv(study_path, index=False)
    covariates.to_csv(cov_path, index=False)

    result = icare.validate_absolute_risk_model(
        study_data_path=str(study_path),
        predicted_risk_interval="total-followup",
        icare_model_parameters=dict(
            model_disease_incidence_rates_path=str(ICARE_LIT_DIR / "age_specific_breast_cancer_incidence_rates.csv"),
            model_competing_incidence_rates_path=str(ICARE_LIT_DIR / "age_specific_all_cause_mortality_rates.csv"),
            model_covariate_formula_path=str(ICARE_LIT_DIR / "model_formula_ge50.txt"),
            model_log_relative_risk_path=str(ICARE_LIT_DIR / "model_log_odds_ratios_ge50.json"),
            model_reference_dataset_path=str(ICARE_LIT_DIR / "reference_covariate_data_ge50.csv"),
            apply_covariate_profile_path=str(cov_path),
        ),
        number_of_percentiles=5,
        seed=50,
    )
    assert set(result) >= {"auc", "brier_score", "expected_by_observed_ratio", "calibration"}
    assert 0.0 <= result["auc"]["auc"] <= 1.0
    assert set(result["brier_score"]) == {"brier_score", "variance", "lower_ci", "upper_ci"}
    assert 0.0 <= result["brier_score"]["brier_score"] <= 1.0

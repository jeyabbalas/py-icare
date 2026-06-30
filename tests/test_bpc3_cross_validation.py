"""Cross-validate py-icare against R iCARE on the BPC3 breast-cancer model.

The BPC3 files in ``data/BPC3/`` are the same model as R iCARE's built-in
``bc_data`` (re-encoded). Golden reference values come from
``tests/r_reference/generate_bpc3_references.R`` and live in
``tests/r_reference/expected/``.

Comparison strategy (see ``tests/README.md``):
  * deterministic workflows (covariate-only, split-interval covariate-only) are
    compared per-subject to a tight tolerance;
  * SNP workflows are stochastic across R/Python RNGs, so per-subject risks are
    compared loosely and population distributions via stable summary statistics;
  * validation metrics use the identical (R-exported) sampling weights.
"""

import numpy as np
import pandas as pd
import pytest

import icare
from icare_test_utils import (
    ATOL_AUC,
    ATOL_DETERMINISTIC,
    ATOL_DISTRIBUTION,
    ATOL_EO,
    ATOL_STOCHASTIC,
    BPC3_DIR,
    FIXTURES_DIR,
    GOLDEN_SEED,
    HL_ALPHA,
    load_golden,
    read_profile,
    reference_population_risks,
    assert_distribution_close,
)

# --- BPC3 input file paths -------------------------------------------------
DISEASE_INC = str(BPC3_DIR / "age_specific_breast_cancer_incidence_rates.csv")
COMPETING_INC = str(BPC3_DIR / "age_specific_all_cause_mortality_rates.csv")
FORMULA = str(BPC3_DIR / "breast_cancer_covariate_model_formula.txt")
LOG_OR = str(BPC3_DIR / "breast_cancer_model_log_odds_ratios.json")
LOG_OR_POST50 = str(BPC3_DIR / "breast_cancer_model_log_odds_ratios_post_50.json")
REFERENCE = str(BPC3_DIR / "reference_covariate_data.csv")
REFERENCE_POST50 = str(BPC3_DIR / "reference_covariate_data_post_50.csv")
SNP_INFO = str(BPC3_DIR / "breast_cancer_72_snps_info.csv")
QUERY_COV = str(BPC3_DIR / "query_covariate_profile.csv")
QUERY_SNP = str(BPC3_DIR / "query_snp_profile.csv")
DIST_KEYS = ("q1", "median", "mean", "q3")  # exclude RNG-sensitive min/max


def test_covariate_only():
    """Deterministic: per-subject risks, linear predictors, reference distribution."""
    golden = load_golden("bpc3_covariate_only.json")
    result = icare.compute_absolute_risk(
        apply_age_start=golden["age_start"],
        apply_age_interval_length=golden["age_interval_length"],
        model_disease_incidence_rates_path=DISEASE_INC,
        model_competing_incidence_rates_path=COMPETING_INC,
        model_covariate_formula_path=FORMULA,
        model_log_relative_risk_path=LOG_OR,
        model_reference_dataset_path=REFERENCE,
        apply_covariate_profile_path=QUERY_COV,
        return_linear_predictors=True,
        return_reference_risks=True,
    )
    profile = read_profile(result)
    np.testing.assert_allclose(profile["risk_estimates"], golden["risks"], atol=ATOL_DETERMINISTIC)
    np.testing.assert_allclose(
        profile["linear_predictors"], golden["linear_predictors"], atol=ATOL_DETERMINISTIC
    )
    assert_distribution_close(
        reference_population_risks(result), golden["reference_risk_summary"],
        ATOL_DISTRIBUTION, keys=DIST_KEYS,
    )


def test_snp_only_no_profile():
    """Stochastic: imputed reference population distribution (stable summaries)."""
    golden = load_golden("bpc3_snp_only_no_profile.json")
    result = icare.compute_absolute_risk(
        apply_age_start=golden["age_start"],
        apply_age_interval_length=golden["age_interval_length"],
        model_disease_incidence_rates_path=DISEASE_INC,
        model_competing_incidence_rates_path=COMPETING_INC,
        model_snp_info_path=SNP_INFO,
        return_reference_risks=True,
        seed=GOLDEN_SEED,
    )
    assert_distribution_close(
        reference_population_risks(result), golden["reference_risk_summary"],
        ATOL_DISTRIBUTION, keys=DIST_KEYS,
    )


def test_snp_only_with_profile():
    """Stochastic (SNP imputation): per-subject risks loose, reference dist stable."""
    golden = load_golden("bpc3_snp_only_with_profile.json")
    result = icare.compute_absolute_risk(
        apply_age_start=golden["age_start"],
        apply_age_interval_length=golden["age_interval_length"],
        model_disease_incidence_rates_path=DISEASE_INC,
        model_competing_incidence_rates_path=COMPETING_INC,
        model_snp_info_path=SNP_INFO,
        apply_snp_profile_path=QUERY_SNP,
        return_reference_risks=True,
        seed=GOLDEN_SEED,
    )
    profile = read_profile(result)
    np.testing.assert_allclose(profile["risk_estimates"], golden["risks"], atol=ATOL_STOCHASTIC)
    assert_distribution_close(
        reference_population_risks(result), golden["reference_risk_summary"],
        ATOL_DISTRIBUTION, keys=DIST_KEYS,
    )


def test_combined_covariate_and_snp():
    """Stochastic (SNP imputation): per-subject risks loose, reference dist stable."""
    golden = load_golden("bpc3_combined.json")
    result = icare.compute_absolute_risk(
        apply_age_start=golden["age_start"],
        apply_age_interval_length=golden["age_interval_length"],
        model_disease_incidence_rates_path=DISEASE_INC,
        model_competing_incidence_rates_path=COMPETING_INC,
        model_covariate_formula_path=FORMULA,
        model_log_relative_risk_path=LOG_OR,
        model_reference_dataset_path=REFERENCE,
        model_snp_info_path=SNP_INFO,
        model_family_history_variable_name="family_history",
        apply_covariate_profile_path=QUERY_COV,
        apply_snp_profile_path=QUERY_SNP,
        return_reference_risks=True,
        seed=GOLDEN_SEED,
    )
    profile = read_profile(result)
    np.testing.assert_allclose(profile["risk_estimates"], golden["risks"], atol=ATOL_STOCHASTIC)
    assert_distribution_close(
        reference_population_risks(result), golden["reference_risk_summary"],
        ATOL_DISTRIBUTION, keys=DIST_KEYS,
    )


def test_split_interval_covariate_only():
    """Deterministic: combined pre/post-cutpoint risks match R per-subject."""
    golden = load_golden("bpc3_split_interval_covariate_only.json")
    result = icare.compute_absolute_risk_split_interval(
        apply_age_start=golden["age_start"],
        apply_age_interval_length=golden["age_interval_length"],
        cutpoint=golden["cutpoint"],
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
    profile = read_profile(result)
    np.testing.assert_allclose(profile["risk_estimates"], golden["risks"], atol=ATOL_DETERMINISTIC)


def test_split_interval_combined():
    """Stochastic (SNP imputation): combined split-interval risks compared loosely."""
    golden = load_golden("bpc3_split_interval_combined.json")
    result = icare.compute_absolute_risk_split_interval(
        apply_age_start=golden["age_start"],
        apply_age_interval_length=golden["age_interval_length"],
        cutpoint=golden["cutpoint"],
        model_disease_incidence_rates_path=DISEASE_INC,
        model_competing_incidence_rates_path=COMPETING_INC,
        model_covariate_formula_before_cutpoint_path=FORMULA,
        model_covariate_formula_after_cutpoint_path=FORMULA,
        model_log_relative_risk_before_cutpoint_path=LOG_OR,
        model_log_relative_risk_after_cutpoint_path=LOG_OR_POST50,
        model_reference_dataset_before_cutpoint_path=REFERENCE,
        model_reference_dataset_after_cutpoint_path=REFERENCE_POST50,
        model_snp_info_path=SNP_INFO,
        model_family_history_variable_name_before_cutpoint="family_history",
        model_family_history_variable_name_after_cutpoint="family_history",
        apply_covariate_profile_before_cutpoint_path=QUERY_COV,
        apply_covariate_profile_after_cutpoint_path=QUERY_COV,
        apply_snp_profile_path=QUERY_SNP,
        seed=GOLDEN_SEED,
    )
    profile = read_profile(result)
    np.testing.assert_allclose(profile["risk_estimates"], golden["risks"], atol=ATOL_STOCHASTIC)


def _build_validation_study(tmp_path):
    """Write a study-data CSV using the R-exported glm sampling weights.

    R iCARE's weighted categorization fails on the dataset's shipped weights, so
    the vignette (and our R generator) recompute them via the inclusion model.
    We reuse those exact weights so both engines validate identically.
    """
    study = pd.read_csv(BPC3_DIR / "validation_nested_case_control_data.csv")
    weights = pd.read_csv(FIXTURES_DIR / "bpc3_nested_cc_glm_weights.csv")
    study["sampling_weights"] = study["id"].map(dict(zip(weights["id"], weights["sampling_weights"])))
    path = tmp_path / "bpc3_validation_study.csv"
    study.to_csv(path, index=False)
    return str(path)


def _assert_validation_metrics(result, golden):
    # Expected/observed ratio and AUC are categorization-independent and agree
    # numerically (AUC has a known ~5e-3 systematic offset, see ATOL_AUC).
    np.testing.assert_allclose(
        result["expected_by_observed_ratio"]["ratio"], golden["eo_ratio"], atol=ATOL_EO
    )
    np.testing.assert_allclose(result["auc"]["auc"], golden["auc"], atol=ATOL_AUC)
    # The Hosmer-Lemeshow chi-square magnitude differs (R vs py weighted binning),
    # so require only that the calibration conclusion agrees at HL_ALPHA.
    py_p = result["calibration"]["absolute_risk"]["p_value"]
    assert (py_p < HL_ALPHA) == (golden["hl_pvalue"] < HL_ALPHA), (
        f"HL calibration conclusion differs: py p={py_p}, R p={golden['hl_pvalue']}"
    )


@pytest.mark.slow
def test_validation_covariate_only(tmp_path):
    """Deterministic risks: E/O matches tightly; AUC within the documented gap."""
    golden = load_golden("bpc3_validation_covariate_only.json")
    study_path = _build_validation_study(tmp_path)
    result = icare.validate_absolute_risk_model(
        study_data_path=study_path,
        predicted_risk_interval="total-followup",
        icare_model_parameters=dict(
            model_disease_incidence_rates_path=DISEASE_INC,
            model_competing_incidence_rates_path=COMPETING_INC,
            model_covariate_formula_path=FORMULA,
            model_log_relative_risk_path=LOG_OR,
            model_reference_dataset_path=REFERENCE,
            model_family_history_variable_name="family_history",
            apply_covariate_profile_path=str(BPC3_DIR / "validation_nested_case_control_covariate_data.csv"),
        ),
        number_of_percentiles=10,
        seed=GOLDEN_SEED,
    )
    _assert_validation_metrics(result, golden)


@pytest.mark.slow
def test_validation_combined(tmp_path):
    """Combined SNP model (stochastic): E/O and AUC within documented tolerances."""
    golden = load_golden("bpc3_validation_combined.json")
    study_path = _build_validation_study(tmp_path)
    result = icare.validate_absolute_risk_model(
        study_data_path=study_path,
        predicted_risk_interval="total-followup",
        icare_model_parameters=dict(
            model_disease_incidence_rates_path=DISEASE_INC,
            model_competing_incidence_rates_path=COMPETING_INC,
            model_covariate_formula_path=FORMULA,
            model_log_relative_risk_path=LOG_OR,
            model_reference_dataset_path=REFERENCE,
            model_snp_info_path=SNP_INFO,
            model_family_history_variable_name="family_history",
            apply_covariate_profile_path=str(BPC3_DIR / "validation_nested_case_control_covariate_data.csv"),
            apply_snp_profile_path=str(BPC3_DIR / "validation_nested_case_control_snp_data.csv"),
        ),
        number_of_percentiles=10,
        seed=GOLDEN_SEED,
    )
    _assert_validation_metrics(result, golden)

"""Cross-validate py-icare against R iCARE on the iCARE-Lit model.

iCARE-Lit is an age-stratified breast-cancer model (sub-models lt50 and ge50; the
ge50 sub-model includes an HRT x BMI interaction). It is NOT shipped with the R
package, so the R side translates the py-icare Patsy specification into R-native
form via ``tests/r_reference/helpers.R`` (a verified, asserted translation).
Golden values come from ``tests/r_reference/generate_icare_lit_references.R``.

These covariate models have no SNP component, so the per-subject risks and linear
predictors are deterministic and compared to a tight tolerance. Query individuals
and the (subsampled) validation cohort are committed fixtures so both engines
score the identical rows.
"""

import numpy as np
import pytest

import icare
from icare_test_utils import (
    ATOL_AUC,
    ATOL_DETERMINISTIC,
    ATOL_DISTRIBUTION,
    ATOL_EO,
    FIXTURES_DIR,
    GOLDEN_SEED,
    HL_ALPHA,
    ICARE_LIT_DIR,
    load_golden,
    read_profile,
    reference_population_risks,
    assert_distribution_close,
)

DISEASE_INC = str(ICARE_LIT_DIR / "age_specific_breast_cancer_incidence_rates.csv")
COMPETING_INC = str(ICARE_LIT_DIR / "age_specific_all_cause_mortality_rates.csv")
DIST_KEYS = ("q1", "median", "mean", "q3")

# Per-sub-model file sets (formula, log-OR, reference, query fixture).
SUBMODELS = {
    "lt50": dict(
        formula=str(ICARE_LIT_DIR / "model_formula_lt50.txt"),
        log_or=str(ICARE_LIT_DIR / "model_log_odds_ratios_lt50.json"),
        reference=str(ICARE_LIT_DIR / "reference_covariate_data_lt50.csv"),
        query=str(FIXTURES_DIR / "icare_lit_query_lt50.csv"),
    ),
    "ge50": dict(
        formula=str(ICARE_LIT_DIR / "model_formula_ge50.txt"),
        log_or=str(ICARE_LIT_DIR / "model_log_odds_ratios_ge50.json"),
        reference=str(ICARE_LIT_DIR / "reference_covariate_data_ge50.csv"),
        query=str(FIXTURES_DIR / "icare_lit_query_ge50.csv"),
    ),
}


@pytest.mark.parametrize("submodel", ["lt50", "ge50"])
def test_covariate_only(submodel):
    """Deterministic per-subject risks, linear predictors, and reference distribution.

    ge50 exercises the HRT x BMI interaction translation.
    """
    files = SUBMODELS[submodel]
    golden = load_golden(f"icare_lit_covariate_only_{submodel}.json")
    result = icare.compute_absolute_risk(
        apply_age_start=golden["age_start"],
        apply_age_interval_length=golden["age_interval_length"],
        model_disease_incidence_rates_path=DISEASE_INC,
        model_competing_incidence_rates_path=COMPETING_INC,
        model_covariate_formula_path=files["formula"],
        model_log_relative_risk_path=files["log_or"],
        model_reference_dataset_path=files["reference"],
        apply_covariate_profile_path=files["query"],
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


def test_split_interval():
    """Deterministic: lt50 before age 50, ge50 after; combined risk matches R."""
    golden = load_golden("icare_lit_split_interval.json")
    result = icare.compute_absolute_risk_split_interval(
        apply_age_start=golden["age_start"],
        apply_age_interval_length=golden["age_interval_length"],
        cutpoint=golden["cutpoint"],
        model_disease_incidence_rates_path=DISEASE_INC,
        model_competing_incidence_rates_path=COMPETING_INC,
        model_covariate_formula_before_cutpoint_path=SUBMODELS["lt50"]["formula"],
        model_covariate_formula_after_cutpoint_path=SUBMODELS["ge50"]["formula"],
        model_log_relative_risk_before_cutpoint_path=SUBMODELS["lt50"]["log_or"],
        model_log_relative_risk_after_cutpoint_path=SUBMODELS["ge50"]["log_or"],
        model_reference_dataset_before_cutpoint_path=SUBMODELS["lt50"]["reference"],
        model_reference_dataset_after_cutpoint_path=SUBMODELS["ge50"]["reference"],
        apply_covariate_profile_before_cutpoint_path=SUBMODELS["lt50"]["query"],
        apply_covariate_profile_after_cutpoint_path=SUBMODELS["ge50"]["query"],
    )
    profile = read_profile(result)
    np.testing.assert_allclose(profile["risk_estimates"], golden["risks"], atol=ATOL_DETERMINISTIC)


@pytest.mark.slow
def test_validation():
    """Full-cohort (unweighted) validation with the ge50 model.

    Exercises the non-weighted validation path. Deterministic risks => E/O matches
    tightly and AUC within the documented offset; HL conclusion must agree.
    """
    golden = load_golden("icare_lit_validation.json")
    result = icare.validate_absolute_risk_model(
        study_data_path=str(FIXTURES_DIR / "icare_lit_validation_study.csv"),
        predicted_risk_interval="total-followup",
        icare_model_parameters=dict(
            model_disease_incidence_rates_path=DISEASE_INC,
            model_competing_incidence_rates_path=COMPETING_INC,
            model_covariate_formula_path=SUBMODELS["ge50"]["formula"],
            model_log_relative_risk_path=SUBMODELS["ge50"]["log_or"],
            model_reference_dataset_path=SUBMODELS["ge50"]["reference"],
            apply_covariate_profile_path=str(FIXTURES_DIR / "icare_lit_validation_covariates.csv"),
        ),
        number_of_percentiles=10,
        seed=GOLDEN_SEED,
    )
    np.testing.assert_allclose(
        result["expected_by_observed_ratio"]["ratio"], golden["eo_ratio"], atol=ATOL_EO
    )
    np.testing.assert_allclose(result["auc"]["auc"], golden["auc"], atol=ATOL_AUC)
    py_p = result["calibration"]["absolute_risk"]["p_value"]
    assert (py_p < HL_ALPHA) == (golden["hl_pvalue"] < HL_ALPHA), (
        f"HL calibration conclusion differs: py p={py_p}, R p={golden['hl_pvalue']}"
    )

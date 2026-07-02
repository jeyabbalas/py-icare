"""Tests for the fit-once / apply-many path: ``build_absolute_risk_model`` +
``AbsoluteRiskModel.apply_to_profile``.

These assert that a model built once and applied to a covariate profile batch is bit-for-bit identical
to the one-shot ``compute_absolute_risk`` (same reference, formula, betas, hazards), that one built model
can be applied to many batches, and — the point of the split — that the reference dataset is read exactly
once regardless of how many batches are scored.
"""

import numpy as np
import pandas as pd
import pytest

import icare
from icare import utils

from icare_test_utils import (
    BPC3_DIR,
    ATOL_DETERMINISTIC,
    load_golden,
    read_profile,
)

# The default 'json' output goes through to_json(), which rounds to ~10 significant digits; the
# 'dataframe' output does not. Compare the two modes at this tolerance (mirrors test_in_memory_io.py).
ATOL_JSON_ROUND = 1e-8

# --- BPC3 covariate-only inputs (mirror test_bpc3_cross_validation.py) ------
DISEASE_INC = str(BPC3_DIR / "age_specific_breast_cancer_incidence_rates.csv")
COMPETING_INC = str(BPC3_DIR / "age_specific_all_cause_mortality_rates.csv")
FORMULA = str(BPC3_DIR / "breast_cancer_covariate_model_formula.txt")
LOG_OR = str(BPC3_DIR / "breast_cancer_model_log_odds_ratios.json")
REFERENCE = str(BPC3_DIR / "reference_covariate_data.csv")
QUERY_COV = str(BPC3_DIR / "query_covariate_profile.csv")
SNP_INFO = str(BPC3_DIR / "breast_cancer_72_snps_info.csv")


def _build_model():
    return icare.build_absolute_risk_model(
        model_disease_incidence_rates_path=DISEASE_INC,
        model_competing_incidence_rates_path=COMPETING_INC,
        model_covariate_formula_path=FORMULA,
        model_log_relative_risk_path=LOG_OR,
        model_reference_dataset_path=REFERENCE,
    )


def _compute_once(profile_path, age_start, age_interval_length):
    return icare.compute_absolute_risk(
        apply_age_start=age_start,
        apply_age_interval_length=age_interval_length,
        model_disease_incidence_rates_path=DISEASE_INC,
        model_competing_incidence_rates_path=COMPETING_INC,
        model_covariate_formula_path=FORMULA,
        model_log_relative_risk_path=LOG_OR,
        model_reference_dataset_path=REFERENCE,
        apply_covariate_profile_path=profile_path,
        return_linear_predictors=True,
        return_reference_risks=True,
    )


def test_build_apply_parity_covariate_only():
    """build + apply_to_profile == one-shot compute_absolute_risk, bit-for-bit, and == the golden."""
    golden = load_golden("bpc3_covariate_only.json")
    age_start, age_interval_length = golden["age_start"], golden["age_interval_length"]

    model = _build_model()
    applied = model.apply_to_profile(
        apply_age_start=age_start,
        apply_age_interval_length=age_interval_length,
        apply_covariate_profile_path=QUERY_COV,
        return_linear_predictors=True,
        return_reference_risks=True,
    )
    one_shot = _compute_once(QUERY_COV, age_start, age_interval_length)

    applied_profile, one_shot_profile = read_profile(applied), read_profile(one_shot)
    # Same reference / formula / betas / hazards / z_profile => identical results (no tolerance needed).
    np.testing.assert_array_equal(applied_profile["risk_estimates"], one_shot_profile["risk_estimates"])
    np.testing.assert_array_equal(
        applied_profile["linear_predictors"], one_shot_profile["linear_predictors"]
    )
    assert applied["model"] == one_shot["model"]
    np.testing.assert_array_equal(
        np.asarray(applied["reference_risks"][0]["population_risks"], dtype=float),
        np.asarray(one_shot["reference_risks"][0]["population_risks"], dtype=float),
    )

    # And the built-model output still matches the R-derived golden within the deterministic tolerance.
    np.testing.assert_allclose(applied_profile["risk_estimates"], golden["risks"], atol=ATOL_DETERMINISTIC)
    np.testing.assert_allclose(
        applied_profile["linear_predictors"], golden["linear_predictors"], atol=ATOL_DETERMINISTIC
    )


def test_apply_reuse_across_batches():
    """One built model applied to two different batches each equals its own one-shot compute."""
    golden = load_golden("bpc3_covariate_only.json")
    age_start, age_interval_length = golden["age_start"], golden["age_interval_length"]

    full = pd.read_csv(QUERY_COV)
    batch_a = full.iloc[:2].copy()   # first two profiles
    batch_b = full.iloc[2:].copy()   # the rest

    model = _build_model()
    for batch in (batch_a, batch_b, batch_a):  # reuse a batch to prove the model is not consumed
        applied = model.apply_to_profile(
            apply_age_start=age_start,
            apply_age_interval_length=age_interval_length,
            apply_covariate_profile_path=batch,
            return_linear_predictors=True,
        )
        one_shot = _compute_once(batch.copy(), age_start, age_interval_length)
        np.testing.assert_array_equal(
            read_profile(applied)["risk_estimates"], read_profile(one_shot)["risk_estimates"]
        )


def test_reference_read_exactly_once_across_many_applies():
    """The reference dataset is read once (at build); each apply reads only its own profile."""
    golden = load_golden("bpc3_covariate_only.json")
    age_start, age_interval_length = golden["age_start"], golden["age_interval_length"]

    real_read_df = utils.read_file_to_dataframe
    real_read_dtype = utils.read_file_to_dataframe_given_dtype
    reference_reads = {"count": 0}
    profile_reads = {"count": 0}

    def counting_read_df(file, *args, **kwargs):
        if file == REFERENCE:
            reference_reads["count"] += 1
        return real_read_df(file, *args, **kwargs)

    def counting_read_dtype(file, *args, **kwargs):
        profile_reads["count"] += 1
        return real_read_dtype(file, *args, **kwargs)

    utils.read_file_to_dataframe = counting_read_df
    utils.read_file_to_dataframe_given_dtype = counting_read_dtype
    try:
        model = _build_model()
        assert reference_reads["count"] == 1  # read once, at build
        n_applies = 4
        for _ in range(n_applies):
            model.apply_to_profile(
                apply_age_start=age_start,
                apply_age_interval_length=age_interval_length,
                apply_covariate_profile_path=QUERY_COV,
            )
        assert reference_reads["count"] == 1  # NOT re-read per apply — the whole point of the split
        assert profile_reads["count"] == n_applies  # profile read once per apply
    finally:
        utils.read_file_to_dataframe = real_read_df
        utils.read_file_to_dataframe_given_dtype = real_read_dtype


def test_apply_output_format_dataframe_parity():
    """output_format='dataframe' returns a DataFrame matching read_json of the 'json' form."""
    golden = load_golden("bpc3_covariate_only.json")
    age_start, age_interval_length = golden["age_start"], golden["age_interval_length"]

    model = _build_model()
    kwargs = dict(
        apply_age_start=age_start,
        apply_age_interval_length=age_interval_length,
        apply_covariate_profile_path=QUERY_COV,
        return_linear_predictors=True,
        return_reference_risks=True,
    )
    as_json = model.apply_to_profile(output_format="json", **kwargs)
    as_df = model.apply_to_profile(output_format="dataframe", **kwargs)

    profile_json = read_profile(as_json)
    profile_df = as_df["profile"]
    assert isinstance(profile_df, pd.DataFrame)
    assert list(profile_df.columns) == list(profile_json.columns)
    # Numeric result columns agree (within the JSON 10-digit rounding); echoed covariate columns differ
    # only in dtype between the two modes (a pre-existing output_format behavior), so are not compared here.
    np.testing.assert_allclose(
        profile_df["risk_estimates"].to_numpy(), profile_json["risk_estimates"].to_numpy(),
        atol=ATOL_JSON_ROUND,
    )
    np.testing.assert_allclose(
        profile_df["linear_predictors"].to_numpy(), profile_json["linear_predictors"].to_numpy(),
        atol=ATOL_JSON_ROUND,
    )
    # dataframe mode yields population_risks as a float64 ndarray, json mode as a list; they agree exactly
    # (reference_risks is not serialized through to_json()).
    assert isinstance(as_json["reference_risks"][0]["population_risks"], list)
    population_risks_df = as_df["reference_risks"][0]["population_risks"]
    assert isinstance(population_risks_df, np.ndarray)
    np.testing.assert_array_equal(
        population_risks_df, np.asarray(as_json["reference_risks"][0]["population_risks"], dtype=float)
    )
    assert as_df["model"] == as_json["model"]


def test_snp_model_not_supported_on_build_path():
    """SNP is out of scope for the fit/apply path; build_absolute_risk_model has no SNP parameter."""
    with pytest.raises(TypeError):
        icare.build_absolute_risk_model(
            model_disease_incidence_rates_path=DISEASE_INC,
            model_competing_incidence_rates_path=COMPETING_INC,
            model_covariate_formula_path=FORMULA,
            model_log_relative_risk_path=LOG_OR,
            model_reference_dataset_path=REFERENCE,
            model_snp_info_path=SNP_INFO,  # not a parameter of build_absolute_risk_model
        )

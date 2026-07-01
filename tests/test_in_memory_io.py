"""In-memory I/O parity and output_format equivalence (py-icare 1.2.0).

Two layers:
  * reader unit tests -- the widened ``icare.utils`` readers accept in-memory objects and reproduce the
    file-based path exactly (including the delicate ``read_file_to_dataframe_given_dtype`` coupling and the
    inline-formula detection);
  * API tests on the real BPC3 fixtures -- in-memory inputs exactly equal path-driven inputs, and the new
    ``output_format='dataframe'`` mode is value-equivalent to (and structurally mirrors) the default JSON.

Default behavior (paths, ``output_format='json'``) is covered by the existing cross-validation suite.
"""

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

import icare
from icare import utils
from icare_test_utils import (
    ATOL_DETERMINISTIC,
    BPC3_DIR,
    FIXTURES_DIR,
    GOLDEN_SEED,
    load_golden,
    read_profile,
)
from test_bpc3_cross_validation import (
    COMPETING_INC,
    DISEASE_INC,
    FORMULA,
    LOG_OR,
    LOG_OR_POST50,
    QUERY_COV,
    QUERY_SNP,
    REFERENCE,
    REFERENCE_POST50,
    SNP_INFO,
    _assert_validation_metrics,
)


def _text(path):
    with open(path) as handle:
        return handle.read()


def _dict(path):
    with open(path) as handle:
        return json.load(handle)


# pandas to_json() rounds to 10 decimal places (double_precision=10), so a value read back from the JSON
# profile differs from the full-precision dataframe-mode value by <=5e-11; this is the equivalence ceiling.
ATOL_JSON_ROUND = 1e-8


# ============================ reader unit tests (fast, no model) ============================

def test_read_file_to_string_inline_and_path(tmp_path):
    text = "y ~ a + b +\n  C(race)"
    expected = "y ~ a + b +   C(race)"
    formula_file = tmp_path / "formula.txt"
    formula_file.write_text(text)

    assert utils.read_file_to_string(str(formula_file)) == expected        # str path
    assert utils.read_file_to_string(pathlib.Path(formula_file)) == expected  # Path
    assert utils.read_file_to_string(text) == expected                     # inline (multi-line)

    # Regression guard: a long inline formula must be treated as inline, not crash. os.path.exists()
    # returns False for an over-long string; pathlib.Path(...).exists() would raise OSError instead.
    long_formula = " + ".join(f"C(x{i})" for i in range(400))  # > 1000 chars, no newlines
    assert utils.read_file_to_string(long_formula) == long_formula


def test_read_file_to_dict_in_memory(tmp_path):
    payload = {"a": 1.5, "b": -2.0}
    path = tmp_path / "log_or.json"
    path.write_text(json.dumps(payload))

    assert utils.read_file_to_dict(path) == payload  # from file
    assert utils.read_file_to_dict(payload) is payload  # dict pass-through (same object)


def test_read_file_to_dataframe_in_memory_and_copy():
    frame = pd.DataFrame({"id": [1, 2, 3], "x": [4, 5, 6]})
    result = utils.read_file_to_dataframe(frame, allow_integers=False)

    assert result.index.name == "id"
    assert result["x"].dtype == float  # allow_integers=False casts numerics to float
    # The caller's frame is not mutated: id is still a column, x still integer, no index set.
    assert "id" in frame.columns
    assert frame.index.name is None
    assert frame["x"].dtype != float


def test_read_file_to_dataframe_raw_copies():
    frame = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = utils.read_file_to_dataframe_raw(frame)
    result.loc[0, "a"] = 99
    assert list(frame["a"]) == [1, 2]  # original untouched


def test_given_dtype_in_memory_matches_csv(tmp_path):
    # Mixed columns exercise the reference->profile coupling: a label-string column, a float column with a
    # NaN, and an integer-valued numeric that must be cast to float, all with string ids.
    frame = pd.DataFrame({
        "id": ["R-001", "R-002", "R-003"],
        "race": ["A", "B", "A"],
        "dose": [0.5, np.nan, 1.5],
        "count": [1, 2, 3],
    })
    csv = tmp_path / "profile.csv"
    frame.to_csv(csv, index=False)
    dtype = {"race": object, "dose": float, "count": float}

    from_csv = utils.read_file_to_dataframe_given_dtype(str(csv), dtype)
    from_df = utils.read_file_to_dataframe_given_dtype(frame, dtype)

    pd.testing.assert_frame_equal(from_df, from_csv)
    assert from_df.index.name == "id"


def test_given_dtype_id_already_index():
    frame = pd.DataFrame({"id": ["a", "b"], "count": [1, 2]}).set_index("id")
    result = utils.read_file_to_dataframe_given_dtype(frame, {"count": float})
    assert result.index.name == "id"
    assert result["count"].dtype == float


def test_given_dtype_tolerates_extra_keys():
    # read_csv(dtype=...) silently ignores keys absent from the data; the DataFrame branch must too
    # (astype would otherwise raise KeyError and change the error contract).
    frame = pd.DataFrame({"id": ["a", "b"], "x": [1.0, 2.0]})
    result = utils.read_file_to_dataframe_given_dtype(frame, {"x": float, "not_a_column": float})
    assert list(result.columns) == ["x"]
    assert result.index.name == "id"


# ============================ API parity / equivalence (real BPC3 fixtures) ============================

def test_input_parity_covariate_only():
    """In-memory inputs (DataFrame / dict / inline formula) exactly equal path-driven inputs."""
    golden = load_golden("bpc3_covariate_only.json")
    common = dict(
        apply_age_start=golden["age_start"],
        apply_age_interval_length=golden["age_interval_length"],
        return_linear_predictors=True,
        return_reference_risks=True,
    )
    result_path = icare.compute_absolute_risk(
        model_disease_incidence_rates_path=DISEASE_INC,
        model_competing_incidence_rates_path=COMPETING_INC,
        model_covariate_formula_path=FORMULA,
        model_log_relative_risk_path=LOG_OR,
        model_reference_dataset_path=REFERENCE,
        apply_covariate_profile_path=QUERY_COV,
        **common,
    )
    result_mem = icare.compute_absolute_risk(
        model_disease_incidence_rates_path=pd.read_csv(DISEASE_INC),
        model_competing_incidence_rates_path=pd.read_csv(COMPETING_INC),
        model_covariate_formula_path=_text(FORMULA),
        model_log_relative_risk_path=_dict(LOG_OR),
        model_reference_dataset_path=pd.read_csv(REFERENCE),
        apply_covariate_profile_path=pd.read_csv(QUERY_COV),
        **common,
    )

    profile_path, profile_mem = read_profile(result_path), read_profile(result_mem)
    np.testing.assert_array_equal(
        profile_mem["risk_estimates"].to_numpy(), profile_path["risk_estimates"].to_numpy()
    )
    np.testing.assert_array_equal(
        profile_mem["linear_predictors"].to_numpy(), profile_path["linear_predictors"].to_numpy()
    )
    assert result_mem["model"] == result_path["model"]


def test_input_parity_combined():
    """Combined covariate + SNP model exercises the given-dtype float path and family history; with a
    fixed seed, in-memory and path-driven results are bit-identical."""
    golden = load_golden("bpc3_combined.json")
    common = dict(
        apply_age_start=golden["age_start"],
        apply_age_interval_length=golden["age_interval_length"],
        model_family_history_variable_name="family_history",
        return_reference_risks=True,
        seed=GOLDEN_SEED,
    )
    result_path = icare.compute_absolute_risk(
        model_disease_incidence_rates_path=DISEASE_INC,
        model_competing_incidence_rates_path=COMPETING_INC,
        model_covariate_formula_path=FORMULA,
        model_log_relative_risk_path=LOG_OR,
        model_reference_dataset_path=REFERENCE,
        model_snp_info_path=SNP_INFO,
        apply_covariate_profile_path=QUERY_COV,
        apply_snp_profile_path=QUERY_SNP,
        **common,
    )
    result_mem = icare.compute_absolute_risk(
        model_disease_incidence_rates_path=pd.read_csv(DISEASE_INC),
        model_competing_incidence_rates_path=pd.read_csv(COMPETING_INC),
        model_covariate_formula_path=_text(FORMULA),
        model_log_relative_risk_path=_dict(LOG_OR),
        model_reference_dataset_path=pd.read_csv(REFERENCE),
        model_snp_info_path=pd.read_csv(SNP_INFO),
        apply_covariate_profile_path=pd.read_csv(QUERY_COV),
        apply_snp_profile_path=pd.read_csv(QUERY_SNP),
        **common,
    )

    np.testing.assert_array_equal(
        read_profile(result_mem)["risk_estimates"].to_numpy(),
        read_profile(result_path)["risk_estimates"].to_numpy(),
    )


def test_output_format_equivalence_compute_absolute_risk():
    """output_format='dataframe' is value-equivalent to (and mirrors) the default JSON records."""
    golden = load_golden("bpc3_covariate_only.json")
    kwargs = dict(
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
    result_json = icare.compute_absolute_risk(output_format="json", **kwargs)
    result_df = icare.compute_absolute_risk(output_format="dataframe", **kwargs)

    profile_json = read_profile(result_json)
    profile_df = result_df["profile"]
    assert isinstance(profile_df, pd.DataFrame)
    assert list(profile_df.columns) == list(profile_json.columns)
    np.testing.assert_allclose(
        profile_df["risk_estimates"].to_numpy(), profile_json["risk_estimates"].to_numpy(),
        atol=ATOL_JSON_ROUND,
    )
    np.testing.assert_allclose(
        profile_df["linear_predictors"].to_numpy(), profile_json["linear_predictors"].to_numpy(),
        atol=ATOL_JSON_ROUND,
    )

    # reference_risks: plain list in JSON mode, contiguous float64 ndarray in dataframe mode. It is not
    # serialized through to_json(), so the two modes agree exactly (no 10-digit rounding).
    assert isinstance(result_json["reference_risks"][0]["population_risks"], list)
    population_risks_df = result_df["reference_risks"][0]["population_risks"]
    assert isinstance(population_risks_df, np.ndarray)
    assert population_risks_df.dtype == np.float64
    np.testing.assert_array_equal(
        population_risks_df, np.asarray(result_json["reference_risks"][0]["population_risks"])
    )

    assert result_df["model"] == result_json["model"]
    assert result_df["method"] == result_json["method"]


def test_split_interval_output_format_and_reference_risks():
    """Split-interval JSON == dataframe == golden, and the (otherwise untested) DataFrame-native combine
    converts nested reference_risks per output_format."""
    golden = load_golden("bpc3_split_interval_covariate_only.json")
    kwargs = dict(
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
        return_reference_risks=True,
    )
    result_json = icare.compute_absolute_risk_split_interval(output_format="json", **kwargs)
    result_df = icare.compute_absolute_risk_split_interval(output_format="dataframe", **kwargs)

    risks_json = read_profile(result_json)["risk_estimates"].to_numpy()
    risks_df = result_df["profile"]["risk_estimates"].to_numpy()
    np.testing.assert_allclose(risks_df, risks_json, atol=ATOL_JSON_ROUND)
    np.testing.assert_allclose(risks_json, golden["risks"], atol=ATOL_DETERMINISTIC)

    before_json = result_json["reference_risks"]["before_cutpoint"][0]["population_risks"]
    before_df = result_df["reference_risks"]["before_cutpoint"][0]["population_risks"]
    assert isinstance(before_json, list)
    assert isinstance(before_df, np.ndarray)
    assert before_df.dtype == np.float64
    np.testing.assert_array_equal(before_df, np.asarray(before_json))


@pytest.mark.slow
def test_validation_in_memory_study_data_and_dataframe_output(tmp_path):
    """Validation with an in-memory study DataFrame + in-memory icare_model_parameters, in dataframe mode.

    Metrics (auc/eo/calibration are plain dicts, format-independent) match the golden, the serialized keys
    come back as DataFrames, and the caller's study frame is not mutated.
    """
    golden = load_golden("bpc3_validation_covariate_only.json")

    # Build the study frame in memory (mirrors _build_validation_study but keeps it off disk).
    study = pd.read_csv(BPC3_DIR / "validation_nested_case_control_data.csv")
    weights = pd.read_csv(FIXTURES_DIR / "bpc3_nested_cc_glm_weights.csv")
    study["sampling_weights"] = study["id"].map(dict(zip(weights["id"], weights["sampling_weights"])))
    study_snapshot = study.copy(deep=True)

    params = dict(
        model_disease_incidence_rates_path=pd.read_csv(DISEASE_INC),
        model_competing_incidence_rates_path=pd.read_csv(COMPETING_INC),
        model_covariate_formula_path=_text(FORMULA),
        model_log_relative_risk_path=_dict(LOG_OR),
        model_reference_dataset_path=pd.read_csv(REFERENCE),
        model_family_history_variable_name="family_history",
        apply_covariate_profile_path=pd.read_csv(
            BPC3_DIR / "validation_nested_case_control_covariate_data.csv"
        ),
    )
    result = icare.validate_absolute_risk_model(
        study_data_path=study,
        predicted_risk_interval="total-followup",
        icare_model_parameters=params,
        number_of_percentiles=10,
        seed=GOLDEN_SEED,
        output_format="dataframe",
    )

    _assert_validation_metrics(result, golden)  # auc / eo / calibration are format-independent dicts
    for key in ("study_data", "incidence_rates", "category_specific_calibration"):
        assert isinstance(result[key], pd.DataFrame)
    assert isinstance(result["auc"], dict)  # plain-dict keys unchanged in dataframe mode
    pd.testing.assert_frame_equal(study, study_snapshot)  # raw reader copied; caller not mutated

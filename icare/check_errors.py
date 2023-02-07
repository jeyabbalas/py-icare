from typing import Union, List, Tuple

import numpy as np
import pandas as pd


def check_age_lengths(apply_age_start, apply_age_interval_length, match, match_name):
    if isinstance(apply_age_start, int):
        apply_age_start = np.full((match.shape[0],), apply_age_start)

    if isinstance(apply_age_interval_length, int):
        apply_age_interval_length = np.full((match.shape[0],), apply_age_interval_length)

    if apply_age_start.shape[0] != apply_age_interval_length.shape[0]:
        raise ValueError("ERROR: 'apply_age_start and 'apply_age_interval_length' must have the same length.")

    if apply_age_start.shape[0] != match.shape[0]:
        raise ValueError(f"ERROR: 'apply_age_start' and number of rows in '{match_name}' must match.")

    if (sum(np.isnan(apply_age_start)) + sum(np.isnan(apply_age_interval_length))) > 0:
        raise ValueError("ERROR: 'apply_age_start' and 'apply_age_interval_length' must not contain missing values.")

    if ((sum(apply_age_start < 0)) + sum(apply_age_interval_length < 0)) > 0:
        raise ValueError("ERROR: 'apply_age_start' and 'apply_age_interval_length' must contain positive values.")

    return apply_age_start, apply_age_interval_length


def check_snp_info(model_snp_info: pd.DataFrame) -> None:
    if any([x not in model_snp_info.columns for x in ["snp_name", "snp_odds_ratio", "snp_freq"]]):
        raise ValueError("ERROR: 'model_snp_info' must have columns 'snp_name', 'snp_odds_ratio', and 'snp_freq'.")


def format_flexible_rate_inputs(data):
    if len(data.columns) == 3:
        integer_ages = np.array(range(np.min(data["start_age"]), np.min(data["end_age"])))
        data_formatted = pd.DataFrame(columns=["age", "rate"], index=np.array(range(integer_ages.shape[0])))
        data_formatted["age"] = integer_ages

        for i in range(data.shape[0]):
            idxs = np.where((data_formatted["age"] >= data.loc[i, "start_age"]) &
                            (data_formatted["age"] <= data.loc[i, "end_age"]))
            data_formatted.loc[idxs, "rate"] = data_formatted.loc[i, "rate"] / idxs[0].shape[0]

        return data_formatted
    else:
        return data


def check_flexible_rate_inputs(data, data_name):
    if not isinstance(data, pd.DataFrame):
        raise ValueError(f"ERROR: argument '{data_name}' requires a Pandas DataFrame as its input.")

    if len(data.columns) != 2 and len(data.columns) != 3:
        raise ValueError(f"ERROR: argument '{data_name}' requires a Pandas DataFrame with either 2 ('age', 'rate') or "
                         f"3 ('start_age', 'end_age', 'rate') columns.")

    if len(data.columns) == 2:
        if "age" not in data.columns or "rate" not in data.columns:
            raise ValueError(f"ERROR: argument '{data_name}' requires a Pandas DataFrame with columns:"
                             f" ['age', 'rate'].")

        if sum(data["age"] % 1) != 0:
            raise ValueError(f"ERROR: The 'age' column in the Pandas DataFrame, passed into argument {data_name}, "
                             f"should be integers.")

    if len(data.columns) == 3:
        if "start_age" not in data.columns or "end_age" not in data.columns or "rate" not in data.columns:
            raise ValueError(f"ERROR: argument '{data_name}' requires a Pandas DataFrame with columns:"
                             f" ['start_age', 'end_age', 'rate'].")

        if data.shape[0] > 1 and (sum(data.loc[1:, "start_age"] - data.loc[:data.shape[0] - 1, "end_age"]) != 0):
            raise ValueError(f"ERROR: The rates provided in that Pandas DataFrame in the argument '{data_name}' must "
                             f"cover sequential age intervals (i.e. if an interval ends at age 30, the next interval "
                             f"must start at age 31).")

    if (sum(data["rate"] < 0.0) + sum(data["rate"] > 1.0)) > 0:
        raise ValueError("ERROR: The rates should be probabilities between 0 and 1.")

    return format_flexible_rate_inputs(data)


def check_rates(model_competing_incidence_rates, model_disease_incidence_rates, apply_age_start,
                apply_age_interval_length):
    lambda_vals = check_flexible_rate_inputs(model_disease_incidence_rates, "model_disease_incidence_rates")

    if model_competing_incidence_rates is None:
        model_competing_incidence_rates = pd.DataFrame(data=np.vstack((lambda_vals["age"],
                                                                       np.zeros(lambda_vals.shape[0]))).T,
                                                       columns=["age", "rate"])

    model_competing_incidence_rates = check_flexible_rate_inputs(model_competing_incidence_rates,
                                                                 "model_competing_incidence_rates")

    if sum([x not in lambda_vals["age"] for
            x in range(np.min(apply_age_start), np.max(apply_age_start + apply_age_interval_length))]) > 0:
        raise ValueError("ERROR: The 'model_disease_incidence_rates' input must have age-specific rates for each "
                         "integer age covered by the prediction intervals defined by 'apply_age_start' and "
                         "'apply_age_interval_length'. You must make these inputs consistent with one "
                         "another to proceed.")

    if sum([x not in model_competing_incidence_rates["age"] for
            x in range(np.min(apply_age_start), np.max(apply_age_start + apply_age_interval_length))]) > 0:
        raise ValueError("ERROR: The 'model_competing_incidence_rates' input must have age-specific rates for each "
                         "integer age covered by the prediction intervals defined by 'apply_age_start' and "
                         "'apply_age_interval_length'. You must make these inputs consistent with one "
                         "another to proceed.")

    return lambda_vals, model_competing_incidence_rates


def check_age_interval_types(
        age_start: Union[int, List[int]],
        age_interval_length: Union[int, List[int]]) -> None:
    if not isinstance(age_start, int) or not isinstance(age_start, list):
        raise ValueError("ERROR: The argument 'apply_age_start' must be an integer or a list of integers.")

    if not isinstance(age_interval_length, int) or not isinstance(age_interval_length, list):
        raise ValueError("ERROR: The argument 'apply_age_interval_length' must be an integer or a list of integers.")

    if isinstance(age_start, list):
        if any([not isinstance(x, int) for x in age_start]):
            raise ValueError("ERROR: The argument 'apply_age_start' must be an integer or a list of integers.")

    if isinstance(age_interval_length, list):
        if any([not isinstance(x, int) for x in age_interval_length]):
            raise ValueError("ERROR: The argument 'apply_age_interval_length' must be an integer or a list of "
                             "integers.")


def check_age_intervals(age_start: List[int], age_interval_length: List[int]) -> None:
    if any([x < 0 for x in age_start]) or any([x < 0 for x in age_interval_length]):
        raise ValueError("ERROR: The 'apply_age_start' and 'apply_age_interval_length' inputs must not contain "
                         "any negative values.")


def check_snp_profile(apply_snp_profile: pd.DataFrame, snp_names: List[str]) -> None:
    if apply_snp_profile.shape[1] != len(snp_names):
        raise ValueError("ERROR: The 'apply_snp_profile' input must have the same number of columns as the "
                         "number of SNPs in the 'model_snp_info' input.")

    if not all(apply_snp_profile.columns == snp_names):
        raise ValueError("ERROR: The 'apply_snp_profile' input must have the same SNPs as those listed in "
                         "'model_snp_info' input.")


def check_profiles(apply_snp_profile: pd.DataFrame, apply_covariate_profile: pd.DataFrame) -> None:
    if len(apply_snp_profile) != len(apply_covariate_profile):
        raise ValueError("ERROR: The data in 'apply_snp_profile' and 'apply_covariate_profile' inputs must have "
                         "the same number of rows.")


def check_family_history(model_family_history_variable_name: str, model_reference_dataset: pd.DataFrame,
                         apply_covariate_profile: pd.DataFrame) -> None:
    if isinstance(model_family_history_variable_name, str):
        raise ValueError("ERROR: The argument 'model_family_history_variable_name' must be a string.")

    if model_family_history_variable_name not in model_reference_dataset.columns:
        raise ValueError("ERROR: The 'model_family_history_variable_name' input must be a column in the "
                         "'model_reference_dataset' input.")

    if model_family_history_variable_name not in apply_covariate_profile.columns:
        raise ValueError("ERROR: The 'model_family_history_variable_name' input must be a column in the "
                         "'apply_covariate_profile' input.")

    reference_fh_unique = model_reference_dataset[model_family_history_variable_name].dropna().unique().astype(int)
    if reference_fh_unique.shape[0] != 2 or any([x not in reference_fh_unique for x in [0, 1]]):
        raise ValueError("ERROR: Family history variable ('model_family_history_variable_name') in the "
                         "'model_reference_dataset' input must be a binary variable.")

    profile_fh_unique = apply_covariate_profile[model_family_history_variable_name].dropna().unique().astype(int)
    if profile_fh_unique.shape[0] != 2 or any([x not in profile_fh_unique for x in [0, 1]]):
        raise ValueError("ERROR: Family history variable ('model_family_history_variable_name') in the "
                         "'apply_covariate_profile' input must be a binary variable.")


def check_population_weights(reference_dataset_weights: List[float], reference_dataset: pd.DataFrame) -> None:
    if len(reference_dataset_weights) != len(reference_dataset):
        raise ValueError("ERROR: the number of values in 'model_reference_dataset_weights' must match the number "
                         "of rows in 'model_reference_dataset'.")

    if any([x is None for x in reference_dataset_weights]):
        raise ValueError("ERROR: the values in 'model_reference_dataset_weights' must not be missing.")

    if any([x < 0 for x in reference_dataset_weights]):
        raise ValueError("ERROR: the values in 'model_reference_dataset_weights' must be greater than or equal to "
                         "zero.")

    if sum(reference_dataset_weights) == 0:
        raise ValueError("ERROR: the sum of the values in 'model_reference_dataset_weights' must be greater than "
                         "zero.")


def check_covariate_parameters(log_relative_risk: dict, reference_dataset: pd.DataFrame,
                               profile: pd.DataFrame) -> None:
    if reference_dataset.shape[0] == 0:
        raise ValueError("ERROR: the 'model_reference_dataset' input must not be empty.")

    if reference_dataset.shape[0] < 200:
        raise ValueError("ERROR: the 'model_reference_dataset' input must contain at least 200 rows.")

    if reference_dataset.isnull().values.any():
        raise ValueError("ERROR: the 'model_reference_dataset' input must not contain any missing values.")

    if len(reference_dataset.columns.difference(profile.columns)):
        raise ValueError("ERROR: the 'model_reference_dataset' input must contain the same columns in the "
                         "'apply_covariate_profile' input.")

    for key, value in log_relative_risk.items():
        if not isinstance(key, str):
            raise ValueError("ERROR: the keys in the 'log_relative_risk' input must be strings corresponding "
                             "to the variable names in the design matrix.")
        if not isinstance(value, float):
            raise ValueError("ERROR: the values in the 'log_relative_risk' input must be floats corresponding "
                             "to the log relative risk associated with each covariate specified in the key.")

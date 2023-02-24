from typing import Union, List

import numpy as np
import pandas as pd


def check_snp_info(model_snp_info: pd.DataFrame) -> None:
    if any([x not in model_snp_info.columns for x in ["snp_name", "snp_odds_ratio", "snp_freq"]]):
        raise ValueError("ERROR: 'model_snp_info_path' must have columns 'snp_name', 'snp_odds_ratio', and 'snp_freq'.")


def check_age_interval_types(
        age_start: Union[int, List[int]],
        age_interval_length: Union[int, List[int]]) -> None:
    if not isinstance(age_start, int) and not isinstance(age_start, list):
        raise ValueError("ERROR: The argument 'apply_age_start' must be an integer or a list of integers.")

    if not isinstance(age_interval_length, int) and not isinstance(age_interval_length, list):
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
        raise ValueError("ERROR: The 'apply_snp_profile_path' input must have the same number of columns as the "
                         "number of SNPs in the 'model_snp_info_path' input.")

    if not all(apply_snp_profile.columns == snp_names):
        raise ValueError("ERROR: The 'apply_snp_profile_path' input must have the same SNPs as those listed in "
                         "'model_snp_info_path' input.")


def check_reference_populations(covariate_population: pd.DataFrame, snp_population: pd.DataFrame) -> None:
    if len(covariate_population) != len(snp_population):
        print("Number of rows in 'model_reference_dataset_path':", len(covariate_population))
        print("Number of rows in the simulated SNP dataset:", len(snp_population))
        raise ValueError("ERROR: The data in the 'model_reference_dataset_path' and the simulated SNP dataset must "
                         "have the same number of rows.")


def check_profiles(covariate_profile: pd.DataFrame, snp_profile: pd.DataFrame) -> None:
    if len(covariate_profile) != len(snp_profile):
        print("Number of rows in 'apply_covariate_profile_path':", len(covariate_profile))
        print("Number of rows in 'apply_snp_profile_path':", len(snp_profile))
        raise ValueError("ERROR: The data in 'apply_covariate_profile_path' and 'apply_snp_profile_path' inputs "
                         "must have the same number of rows.")


def check_population_weights(reference_dataset_weights: List[float], reference_dataset: pd.DataFrame) -> None:
    if len(reference_dataset_weights) != len(reference_dataset):
        raise ValueError("ERROR: the number of values in 'model_reference_dataset_weights' must match the number "
                         "of rows in 'model_reference_dataset_path'.")

    if any([x is None for x in reference_dataset_weights]):
        raise ValueError("ERROR: the values in 'model_reference_dataset_weights' must not be missing.")

    if any([x < 0 for x in reference_dataset_weights]):
        raise ValueError("ERROR: the values in 'model_reference_dataset_weights' must be greater than or equal to "
                         "zero.")

    if sum(reference_dataset_weights) == 0:
        raise ValueError("ERROR: the sum of the values in 'model_reference_dataset_weights' must be greater than "
                         "zero.")


def check_covariate_reference_dataset(reference_dataset: pd.DataFrame) -> None:
    if reference_dataset.shape[0] < 200:
        raise ValueError("ERROR: the 'model_reference_dataset_path' input must contain at least 200 rows.")

    if reference_dataset.isnull().values.any():
        raise ValueError("ERROR: the 'model_reference_dataset_path' input must not contain any missing values.")


def check_covariate_log_relative_risk(log_relative_risk: dict, population_distribution: pd.DataFrame) -> None:
    if len(log_relative_risk) == 0:
        raise ValueError("ERROR: the 'log_relative_risk' input must not be empty.")

    if any([not isinstance(x, str) for x in log_relative_risk.keys()]):
        raise ValueError("ERROR: the keys in the 'log_relative_risk' input must be design matrix variable "
                         "names as strings.")

    if any([not isinstance(x, float) for x in log_relative_risk.values()]):
        raise ValueError("ERROR: the values in the 'log_relative_risk' input must be floats corresponding "
                         "to the log relative risk associated with the design matrix variable.")

    if any([x not in population_distribution.columns for x in log_relative_risk.keys()]):
        print(f"'model_reference_dataset_path' design matrix columns: {population_distribution.columns}")
        print(f"'model_log_relative_risk' keys: {log_relative_risk.keys()}")
        raise ValueError("ERROR: the keys in the 'log_relative_risk' input must correspond to the column "
                         "names in the 'population_distribution' design matrix resulting from the input "
                         " Patsy formula in 'model_covariate_formula_path'.")


def check_covariate_profile(reference_dataset: pd.DataFrame, profile: pd.DataFrame) -> None:
    if len(reference_dataset.columns.difference(profile.columns)):
        raise ValueError("ERROR: the 'model_reference_dataset_path' input must contain the same columns in the "
                         "'apply_covariate_profile_path' input.")


def check_num_imputations(num_imputations: int) -> None:
    if not isinstance(num_imputations, int):
        raise ValueError("ERROR: The argument 'num_imputations' must be an integer.")

    if num_imputations < 1 or num_imputations > 20:
        raise ValueError("ERROR: The argument 'num_imputations' must be between 1 and 20.")


def check_covariate_profile_against_reference_population(profile: pd.DataFrame,
                                                         population_distribution: pd.DataFrame) -> None:
    if not all(profile.columns == population_distribution.columns):
        print(f"'model_reference_dataset_path' design matrix columns: {population_distribution.columns}")
        print(f"'apply_covariate_profile_path' design matrix columns: {profile.columns}")
        raise ValueError("ERROR: The design matrix, resulting from the Patsy formula in 'model_covariate_formula_path'"
                         ", for 'apply_covariate_profile_path' do not match the design matrix resulting from the "
                         "'model_reference_dataset_path' input.")


def check_family_history_variable_name_type(family_history_variable_name: str) -> None:
    if not isinstance(family_history_variable_name, str):
        raise ValueError("ERROR: The argument 'family_history_variable_name' must be a string corresponding to "
                         "the variable name of the binary family history variable in the "
                         "'model_reference_dataset_path'.")


def check_family_history_variable(family_history_variable_name: str, profile: pd.DataFrame,
                                  population_distribution: pd.DataFrame) -> None:
    if family_history_variable_name not in profile.columns:
        print(f"'model_family_history_variable_name' inferred from 'model_covariate_formula_path': "
              f"{family_history_variable_name}")
        print(f"'apply_covariate_profile_path' design matrix columns: {profile.columns}")
        raise ValueError("ERROR: The 'model_family_history_variable_name' input must be a column in the "
                         "design matrix of the 'apply_covariate_profile_path' input data.")

    if family_history_variable_name not in population_distribution.columns:
        print(f"'model_family_history_variable_name' inferred from 'model_covariate_formula_path': "
              f"{family_history_variable_name}")
        print(f"'model_reference_dataset_path' design matrix columns: {population_distribution.columns}")
        raise ValueError("ERROR: The 'model_family_history_variable_name' input must be a column in the "
                         "design matrix of the 'model_reference_dataset_path' input data.")

    profile_fh_unique = profile[family_history_variable_name].dropna().unique().astype(int)
    if any([x not in [0, 1] for x in profile_fh_unique]):
        print(f"Observed values in 'apply_covariate_profile_path' for 'model_family_history_variable_name': "
              f"{profile_fh_unique}")
        raise ValueError("ERROR: The 'model_family_history_variable_name' input must be a binary variable in the "
                         "'apply_covariate_profile_path' input.")

    reference_fh_unique = population_distribution[family_history_variable_name].unique().astype(int)
    if any([x not in [0, 1] for x in reference_fh_unique]):
        print(f"Observed values in 'model_reference_dataset_path' for 'model_family_history_variable_name': "
              f"{reference_fh_unique}")
        raise ValueError("ERROR: The 'model_family_history_variable_name' input must be a binary variable in the "
                         "'model_reference_dataset_path' input.")


def check_population_weights_are_equal(covariate_weights: np.ndarray, snp_weights: np.ndarray) -> None:
    if not np.allclose(covariate_weights, snp_weights):
        print(f"Population weights inferred from the 'model_reference_dataset_path' input: {covariate_weights}")
        print(f"Population weights inferred from the 'model_snp_dataset' input: {snp_weights}")
        raise ValueError("ERROR: The population weights inferred from the 'model_reference_dataset_path' input must be "
                         "the same as the population weights inferred from the 'model_snp_dataset' input.")


def check_covariate_reference_dataset_weights_name(reference_dataset_weights_name: str,
                                                   reference_dataset: pd.DataFrame) -> None:
    if reference_dataset_weights_name not in reference_dataset.columns:
        raise ValueError(f"ERROR: The 'model_reference_dataset_weights_name' ({reference_dataset.columns}) input "
                         f"must be a column in the 'model_reference_dataset_path' input data.")


def check_rate_format(rates: pd.DataFrame, argument_name: str) -> None:
    if rates.shape[1] not in [2, 3]:
        print(f"Number of columns in '{argument_name}': {rates.shape[1]}")
        raise ValueError(f"ERROR: The '{argument_name}' input must have either 2 or 3 columns. If the number "
                         f"of columns is 2, their names should be 'age' and 'rate'. If the number of columns is 3, "
                         f"their names should be start_age, end_age, and rate.")

    if rates.shape[1] == 2:
        if "age" not in rates.columns or "rate" not in rates.columns:
            print(f"Column names in '{argument_name}': {rates.columns}")
            raise ValueError(f"ERROR: The '{argument_name}' input must have either 2 or 3 columns. If the number "
                             f"of columns is 2, their names should be 'age' and 'rate'. If the number of columns is 3, "
                             f"their names should be start_age, end_age, and rate.")

        if rates['age'].dtype != int:
            raise ValueError(f"ERROR: The 'age' column in the '{argument_name}' input must only contain integer "
                             f"values.")

    if rates.shape[1] == 3:
        if "start_age" not in rates.columns or "end_age" not in rates.columns or "rate" not in rates.columns:
            print(f"Column names in '{argument_name}': {rates.columns}")
            raise ValueError(f"ERROR: The '{argument_name}' input must have either 2 or 3 columns. If the number "
                             f"of columns is 2, their names should be 'age' and 'rate'. If the number of columns is 3, "
                             f"their names should be start_age, end_age, and rate.")

        if rates['start_age'].dtype != int or rates['end_age'].dtype != int:
            raise ValueError(f"ERROR: The 'start_age' and 'end_age' columns in the '{argument_name}' input must only "
                             f"contain integer values.")

        if np.sum(rates['start_age'].values[1:] - rates['end_age'].values[:-1]) != 0:
            raise ValueError(f"ERROR: The 'start_age' and 'end_age' columns in the '{argument_name}' input must "
                             f"be sequential i.e. the end age of one row must be the start age of the next row.")

    if rates['rate'].dtype != float:
        raise ValueError(f"ERROR: The 'rate' column in the '{argument_name}' input must only contain float values.")

    if rates['rate'].min() < 0 or rates['rate'].max() > 1:
        raise ValueError(f"ERROR: The 'rate' column in the '{argument_name}' input are probabilities and so, they "
                         f"must only contain values between 0 and 1.")


def check_rate_covers_all_ages(rates: pd.Series, age_start: List[int], age_interval_length: List[int],
                               argument_name: str) -> None:
    step = 1
    age_range = list(range(np.min(np.array(age_start)),
                           np.max(np.array(age_start) + np.array(age_interval_length)), step))
    ages_not_covered = [age for age in age_range if ((age not in rates.index) or (pd.isna(rates.loc[age])))]
    if len(ages_not_covered) > 0:
        raise ValueError(f"ERROR: The '{argument_name}' input must cover all ages from 'apply_age_start' to "
                         f"'apply_age_start' + 'apply_age_interval_length'. \n"
                         f"The following ages are not covered: {ages_not_covered}.")


def check_return_population_risks_type(return_reference_risks: bool) -> None:
    if not isinstance(return_reference_risks, bool):
        raise ValueError("ERROR: The 'return_reference_risks' input must be a boolean.")

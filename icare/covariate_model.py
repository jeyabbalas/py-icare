import pathlib
from typing import List, Union, Optional

import numpy as np
import pandas as pd

from icare import utils, check_errors


class CovariateModel:
    """A general-purpose covariate model."""
    age_start: np.ndarray
    age_interval_length: np.ndarray
    beta_estimates: np.ndarray
    z_profile: pd.DataFrame
    population_distribution: pd.DataFrame
    population_weights: np.ndarray

    def __init__(
            self,
            formula_path: Union[str, pathlib.Path, None],
            log_relative_risk_path: Union[str, pathlib.Path, None],
            reference_dataset_path: Union[str, pathlib.Path, None],
            profile_path: Union[str, pathlib.Path, None],
            reference_dataset_weights: Optional[List[float]],
            age_start: Union[int, List[int]],
            age_interval_length: Union[int, List[int]]) -> None:
        parameters = [formula_path, log_relative_risk_path, reference_dataset_path, profile_path]
        any_parameter_missing = any([x is None for x in parameters])

        if any_parameter_missing:
            raise ValueError("ERROR: Either all or none of the covariate parameters— 'model_covariate_formula', "
                             "'model_log_relative_risk', 'model_reference_dataset', and 'apply_covariate_profile'"
                             "— should be specified. If none of them are specified, it implies a SNP-only model.")

        formula = utils.read_file_to_string(formula_path)
        log_relative_risk = utils.read_file_to_dict(log_relative_risk_path)
        reference_dataset = utils.read_file_to_dataframe(reference_dataset_path)
        profile = utils.read_file_to_dataframe(profile_path)
        self.set_age_intervals(age_start, age_interval_length, profile)

        check_errors.check_covariate_parameters(formula, log_relative_risk, reference_dataset, profile)
        self.set_population_weights(reference_dataset_weights, reference_dataset)

    def set_age_intervals(
            self,
            age_start: Union[int, List[int]],
            age_interval_length: Union[int, List[int]],
            profile: pd.DataFrame) -> None:
        check_errors.check_age_interval_types(age_start, age_interval_length)

        if isinstance(age_start, int):
            age_start = [age_start]*len(profile)

        if isinstance(age_interval_length, int):
            age_interval_length = [age_interval_length]*len(profile)

        if len(age_start) != len(profile) or len(age_interval_length) != len(profile):
            raise ValueError("ERROR: the number of values in 'apply_age_start' and 'apply_age_interval_length', "
                             "and the number of rows in 'apply_covariate_profile' must match.")

        age_start, age_interval_length = np.array(age_start).astype(float), np.array(age_interval_length).astype(float)
        check_errors.check_age_intervals(age_start, age_interval_length)

        self.age_start = age_start
        self.age_interval_length = age_interval_length

    def set_population_weights(
            self,
            reference_dataset_weights: Optional[List[float]],
            reference_dataset: pd.DataFrame) -> None:
        if reference_dataset_weights is None:
            self.population_weights = np.ones(len(reference_dataset)) / len(reference_dataset)
        else:
            check_errors.check_population_weights(reference_dataset_weights, reference_dataset)
            self.population_weights = np.array(reference_dataset_weights) / sum(reference_dataset_weights)

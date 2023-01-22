import pathlib
from typing import Union, List, Optional

import numpy as np
import pandas as pd

from icare import check_errors, utils
from icare.covariate_model import CovariateModel
from icare.snp_model import SnpModel


class AbsoluteRiskModel:
    """Something"""
    covariate_model: Optional[CovariateModel] = None
    snp_model: Optional[SnpModel] = None

    age_start: List[int]
    age_interval_length: List[int]
    baseline_hazard: float
    beta_estimates: np.ndarray
    z_profile: pd.DataFrame
    population_distribution: pd.DataFrame
    population_weights: np.ndarray

    def __init__(
            self,
            apply_age_start: Union[int, List[int]],
            apply_age_interval_length: Union[int, List[int]],
            model_disease_incidence_rates: Union[str, pathlib.Path],
            model_covariate_formula: Union[str, pathlib.Path, None],
            model_snp_info: Union[str, pathlib.Path, None],
            model_log_relative_risk: Union[str, pathlib.Path, None],
            model_reference_dataset: Union[str, pathlib.Path, None],
            model_reference_dataset_weights: Optional[List[float]],
            model_competing_incidence_rates: Union[str, pathlib.Path, None],
            model_family_history_variable_name: str,
            num_imputations: int,
            apply_covariate_profile: Union[str, pathlib.Path, None],
            apply_snp_profile: Union[str, pathlib.Path, None]) -> None:
        covariate_parameters = [model_covariate_formula, model_log_relative_risk, model_reference_dataset,
                                apply_covariate_profile]
        any_covariate_parameter_specified = any([x is not None for x in covariate_parameters])

        if any_covariate_parameter_specified:
            self.covariate_model = CovariateModel(
                model_covariate_formula, model_log_relative_risk, model_reference_dataset, apply_covariate_profile,
                model_reference_dataset_weights, apply_age_start, apply_age_interval_length
            )

            if model_snp_info is not None:
                self.snp_model = SnpModel(
                    model_snp_info, apply_snp_profile, model_family_history_variable_name, apply_age_start,
                    apply_age_interval_length, self.covariate_model
                )
        else:
            self.snp_model = SnpModel(
                model_snp_info, apply_snp_profile, model_family_history_variable_name, apply_age_start,
                apply_age_interval_length, self.covariate_model
            )

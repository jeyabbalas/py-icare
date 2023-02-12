import pathlib
from typing import Union, List, Optional, Dict

import numpy as np
import pandas as pd

from icare import check_errors, utils
from icare.covariate_model import CovariateModel
from icare.snp_model import SnpModel


class AbsoluteRiskModel:
    """Absolute risk model"""
    covariate_model: Optional[CovariateModel] = None
    snp_model: Optional[SnpModel] = None

    age_start: Union[int, List[int]]
    age_interval_length: Union[int, List[int]]
    beta_estimates: np.ndarray
    z_profile: pd.DataFrame
    population_distribution: pd.DataFrame
    population_weights: np.ndarray
    num_imputations: int

    baseline_hazard_function: Dict[int, float]
    competing_incidence_rates_function: Dict[int, float]

    def __init__(self,
                 apply_age_start: Union[int, List[int]],
                 apply_age_interval_length: Union[int, List[int]],
                 disease_incidence_rates_path: Union[str, pathlib.Path],
                 formula_path: Union[str, pathlib.Path, None],
                 snp_info_path: Union[str, pathlib.Path, None],
                 log_relative_risk_path: Union[str, pathlib.Path, None],
                 reference_dataset_path: Union[str, pathlib.Path, None],
                 model_reference_dataset_weights_variable_name: Optional[str],
                 competing_incidence_rates_path: Union[str, pathlib.Path, None],
                 model_family_history_variable_name: str,
                 num_imputations: int,
                 covariate_profile_path: Union[str, pathlib.Path, None],
                 snp_profile_path: Union[str, pathlib.Path, None]) -> None:
        check_errors.check_age_interval_types(apply_age_start, apply_age_interval_length)
        self.age_start, self.age_interval_length = apply_age_start, apply_age_interval_length

        check_errors.check_num_imputations(num_imputations)
        self.num_imputations = num_imputations

        covariate_parameters = [formula_path, log_relative_risk_path, reference_dataset_path, covariate_profile_path]
        any_covariate_parameter_specified = any([x is not None for x in covariate_parameters])
        instantiate_special_snp_model = snp_info_path is not None

        if any_covariate_parameter_specified:
            self.covariate_model = CovariateModel(
                formula_path, log_relative_risk_path, reference_dataset_path, covariate_profile_path,
                model_reference_dataset_weights_variable_name, self.age_start, self.age_interval_length
            )
            self.age_start = self.covariate_model.age_start
            self.age_interval_length = self.covariate_model.age_interval_length

        if instantiate_special_snp_model:
            self.snp_model = SnpModel(
                snp_info_path, snp_profile_path, model_family_history_variable_name, self.age_start,
                self.age_interval_length, self.covariate_model
            )

            if self.covariate_model is None:
                self.age_start = self.snp_model.age_start
                self.age_interval_length = self.snp_model.age_interval_length
            check_errors.check_profiles(self.covariate_model.z_profile, self.snp_model.z_profile)
            # merge everything here (check before merging)
        else:
            if self.covariate_model is None:
                raise ValueError("ERROR: Since you did not provide any covariate model parameters, it is assumed"
                                 " that you are fitting a SNP-only model, and thus you must provide relevant data"
                                 " to the 'model_snp_info' parameter.")

    def _set_z_profile(self) -> None:
        if self.covariate_model is not None and self.snp_model is not None:
            check_errors.check_profiles(self.covariate_model.z_profile, self.snp_model.z_profile)
            self.z_profile = pd.concat([self.covariate_model.z_profile, self.snp_model.z_profile], axis=1)
        elif self.covariate_model is not None:
            self.z_profile = self.covariate_model.z_profile
        elif self.snp_model is not None:
            self.z_profile = self.snp_model.z_profile
        else:
            raise ValueError("ERROR: No query profiles were set. Please pass inputs for arguments "
                             "'apply_snp_profile' and/or 'apply_covariate_profile'.")

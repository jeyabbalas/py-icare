import pathlib
from typing import Union, List, Optional

import numpy as np
import pandas as pd

from icare import check_errors, utils
from icare.covariate_model import CovariateModel
from icare.snp_model import SnpModel


def format_rates(rates: pd.DataFrame) -> pd.Series:
    if len(rates.columns) == 3:
        age = list(range(rates["start_age"].min(), rates["end_age"].max() + 1))
        rate = np.zeros(len(age), dtype=float)
        formatted_rates = pd.DataFrame({"age": age, "rate": rate})

        for _, row in rates.iterrows():
            ages_within_interval = (formatted_rates["age"] >= row["start_age"]) & \
                                   (formatted_rates["age"] < row["end_age"])
            formatted_rates.loc[ages_within_interval, "rate"] = row["rate"] / len(formatted_rates[ages_within_interval])

        formatted_rates = pd.Series(data=formatted_rates["rate"].values, index=formatted_rates["age"], name="rate")
        formatted_rates.index.name = "age"

        return formatted_rates

    rates = pd.Series(data=rates["rate"].values, index=rates["age"], name="rate")
    rates.index.name = "age"

    return rates


def estimate_baseline_hazard(marginal_disease_incidence_rates: pd.Series, beta: np.ndarray, z: pd.DataFrame,
                             w: np.ndarray) -> pd.Series:
    """
    Baseline hazard: age-specific disease incidence rates when all covariates take their baseline values.
    """
    expected_risk_score_current = np.repeat(np.average(np.exp(np.matmul(z, beta)), weights=w),
                                            len(marginal_disease_incidence_rates))
    expected_risk_score_previous = expected_risk_score_current - 1
    epsilon = 1e-3

    while np.sum(np.abs(expected_risk_score_current - expected_risk_score_previous)) > epsilon:
        expected_risk_score_previous = expected_risk_score_current

        # Update baseline hazard using the new expected risk score
        baseline_hazard = marginal_disease_incidence_rates.values / expected_risk_score_previous

        # Update expected risk score using the new baseline hazard (under the proportional hazard model)
        weight_matrix = np.repeat(w.reshape(-1, 1), len(baseline_hazard), axis=1)
        hazard_scale = -np.exp(np.matmul(z.values, beta))
        cumsum_baseline_hazard = np.concatenate((np.array([0.]), np.cumsum(baseline_hazard)[:-1]))
        numerator = np.exp(
            np.matmul(hazard_scale.reshape(-1, 1), cumsum_baseline_hazard.reshape(1, -1))) * weight_matrix
        denominator = np.sum(numerator, axis=0)
        probability_z_given_t = numerator / denominator
        expected_risk_score_current = np.sum(probability_z_given_t * np.exp(np.matmul(z.values, beta)).reshape(-1, 1),
                                             axis=0)

    baseline_hazard = marginal_disease_incidence_rates.values / expected_risk_score_current
    baseline_hazard = pd.Series(data=baseline_hazard, index=marginal_disease_incidence_rates.index, name="rate")
    baseline_hazard.index.name = "age"

    return baseline_hazard


class AbsoluteRiskModel:
    """Absolute risk model"""
    age_start: Union[int, List[int]]
    age_interval_length: Union[int, List[int]]
    beta_estimates: np.ndarray
    z_profile: pd.DataFrame
    population_distribution: pd.DataFrame
    population_weights: np.ndarray
    num_imputations: int

    baseline_hazard: pd.Series
    competing_incidence_rates: pd.Series

    def __init__(self,
                 apply_age_start: Union[int, List[int]],
                 apply_age_interval_length: Union[int, List[int]],
                 age_specific_disease_incidence_rates_path: Union[str, pathlib.Path],
                 formula_path: Union[str, pathlib.Path, None],
                 snp_info_path: Union[str, pathlib.Path, None],
                 log_relative_risk_path: Union[str, pathlib.Path, None],
                 reference_dataset_path: Union[str, pathlib.Path, None],
                 model_reference_dataset_weights_variable_name: Optional[str],
                 age_specific_competing_incidence_rates_path: Union[str, pathlib.Path, None],
                 model_family_history_variable_name: str,
                 num_imputations: int,
                 covariate_profile_path: Union[str, pathlib.Path, None],
                 snp_profile_path: Union[str, pathlib.Path, None]) -> None:
        covariate_model: Optional[CovariateModel] = None
        snp_model: Optional[SnpModel] = None

        check_errors.check_age_interval_types(apply_age_start, apply_age_interval_length)
        self.age_start, self.age_interval_length = apply_age_start, apply_age_interval_length

        check_errors.check_num_imputations(num_imputations)
        self.num_imputations = num_imputations

        covariate_parameters = [formula_path, log_relative_risk_path, reference_dataset_path, covariate_profile_path]
        any_covariate_parameter_specified = any([x is not None for x in covariate_parameters])
        instantiate_special_snp_model = snp_info_path is not None

        if any_covariate_parameter_specified:
            covariate_model = CovariateModel(
                formula_path, log_relative_risk_path, reference_dataset_path, covariate_profile_path,
                model_reference_dataset_weights_variable_name, self.age_start, self.age_interval_length)
            self.age_start = covariate_model.age_start
            self.age_interval_length = covariate_model.age_interval_length

        if instantiate_special_snp_model:
            snp_model = SnpModel(
                snp_info_path, snp_profile_path, model_family_history_variable_name, self.age_start,
                self.age_interval_length, self.num_imputations, covariate_model)

            if covariate_model is None:
                self.age_start = snp_model.age_start
                self.age_interval_length = snp_model.age_interval_length
            check_errors.check_profiles(covariate_model.z_profile, snp_model.z_profile)
        else:
            if covariate_model is None:
                raise ValueError("ERROR: Since you did not provide any covariate model parameters, it is assumed"
                                 " that you are fitting a SNP-only model, and thus you must provide relevant data"
                                 " to the 'model_snp_info_path' parameter.")

        self._set_population_distribution(covariate_model, snp_model)
        self._set_population_weights(covariate_model, snp_model)
        self._set_beta_estimates(covariate_model, snp_model)
        self._set_z_profile(covariate_model, snp_model)
        self._set_baseline_hazard(age_specific_disease_incidence_rates_path)
        self._set_competing_incidence_rates(age_specific_competing_incidence_rates_path)

    def _set_population_distribution(self, covariate_model: Optional[CovariateModel],
                                     snp_model: Optional[SnpModel]) -> None:
        if covariate_model is not None and snp_model is not None:
            check_errors.check_reference_populations(covariate_model.population_distribution,
                                                     snp_model.population_distribution)
            self.population_distribution = pd.concat(
                [covariate_model.population_distribution, snp_model.population_distribution], axis=1)
        elif covariate_model is not None:
            self.population_distribution = covariate_model.population_distribution
        elif snp_model is not None:
            self.population_distribution = snp_model.population_distribution
        else:
            raise ValueError("ERROR: No reference populations were set. Please pass inputs for arguments "
                             "'model_reference_dataset_path' and, optionally 'model_snp_info_path'.")

    def _set_population_weights(self, covariate_model: Optional[CovariateModel], snp_model: Optional[SnpModel]) -> None:
        if covariate_model is not None and snp_model is not None:
            check_errors.check_population_weights_are_equal(covariate_model.population_weights,
                                                            snp_model.population_weights)
            self.population_weights = covariate_model.population_weights
        elif covariate_model is not None:
            self.population_weights = covariate_model.population_weights
        elif snp_model is not None:
            self.population_weights = snp_model.population_weights
        else:
            raise ValueError("ERROR: No reference population weights were set. Please pass inputs for argument "
                             "'model_reference_dataset_weights_variable_name'.")

    def _set_beta_estimates(self, covariate_model: Optional[CovariateModel], snp_model: Optional[SnpModel]) -> None:
        if covariate_model is not None and snp_model is not None:
            self.beta_estimates = np.concatenate((covariate_model.beta_estimates, snp_model.beta_estimates))
        elif covariate_model is not None:
            self.beta_estimates = covariate_model.beta_estimates
        elif snp_model is not None:
            self.beta_estimates = snp_model.beta_estimates
        else:
            raise ValueError("ERROR: No beta estimates were set. Please pass inputs for arguments "
                             "'model_log_relative_risk_path' and/or 'model_snp_info_path'.")

    def _set_z_profile(self, covariate_model: Optional[CovariateModel], snp_model: Optional[SnpModel]) -> None:
        if covariate_model is not None and snp_model is not None:
            check_errors.check_profiles(covariate_model.z_profile, snp_model.z_profile)
            self.z_profile = pd.concat([covariate_model.z_profile, snp_model.z_profile], axis=1)
            self.z_profile.index = covariate_model.z_profile.index
        elif covariate_model is not None:
            self.z_profile = covariate_model.z_profile
        elif snp_model is not None:
            self.z_profile = snp_model.z_profile
        else:
            raise ValueError("ERROR: No query profiles were set. Please pass inputs for arguments "
                             "'apply_covariate_profile_path' and/or 'apply_snp_profile_path'.")

    def _set_baseline_hazard(self, marginal_disease_incidence_rates_path: Union[str, pathlib.Path]) -> None:
        marginal_disease_incidence_rates = utils.read_file_to_dataframe(marginal_disease_incidence_rates_path)
        check_errors.check_rate_format(marginal_disease_incidence_rates, "model_disease_incidence_rates_path")
        marginal_disease_incidence_rates = format_rates(marginal_disease_incidence_rates)
        check_errors.check_rate_covers_all_ages(marginal_disease_incidence_rates, self.age_start,
                                                self.age_interval_length, "model_disease_incidence_rates_path")
        self.baseline_hazard = estimate_baseline_hazard(marginal_disease_incidence_rates, self.beta_estimates,
                                                        self.population_distribution, self.population_weights)

    def _set_competing_incidence_rates(self, competing_incidence_rates_path: Union[str, pathlib.Path, None]):
        if competing_incidence_rates_path is None:
            self.competing_incidence_rates = pd.Series(data=np.zeros(len(self.baseline_hazard)),
                                                       index=self.baseline_hazard.index, name="rate")
            self.competing_incidence_rates.index.name = "age"
        else:
            competing_incidence_rates = utils.read_file_to_dataframe(competing_incidence_rates_path)
            check_errors.check_rate_format(competing_incidence_rates, "model_competing_incidence_rates_path")
            competing_incidence_rates = format_rates(competing_incidence_rates)
            check_errors.check_rate_covers_all_ages(competing_incidence_rates, self.age_start, self.age_interval_length,
                                                    "model_competing_incidence_rates_path")
            self.competing_incidence_rates = competing_incidence_rates

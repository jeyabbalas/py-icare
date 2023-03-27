import pathlib
from typing import Union, List, Optional, Tuple

import numpy as np
import pandas as pd

from icare import check_errors, utils
from icare.covariate_model import CovariateModel
from icare.snp_model import SnpModel


class AbsoluteRiskResults:
    """
    A data structure to hold the results from the calculation of absolute risks.
    """
    age_interval_start: np.ndarray
    age_interval_end: np.ndarray
    linear_predictors: pd.Series
    risk_estimates: pd.Series
    population_risk_estimates: pd.DataFrame

    def set_ages(self, age_start: List[int], age_interval_length: List[int]) -> None:
        self.age_interval_start = np.array(age_start)
        self.age_interval_end = self.age_interval_start + np.array(age_interval_length)

    def set_linear_predictors(self, linear_predictors: np.ndarray, indices: List) -> None:
        self.linear_predictors = pd.Series(data=linear_predictors, index=indices, name="linear_predictors", dtype=float)

    def set_risk_estimates(self, risk_estimates: np.ndarray, indices: List) -> None:
        self.risk_estimates = pd.Series(data=risk_estimates, index=indices, name="risk_estimates", dtype=float)

    def set_population_risk_estimates(self, population_risk_estimates):
        self.population_risk_estimates = population_risk_estimates


def format_rates(rates: pd.DataFrame) -> pd.Series:
    if len(rates.columns) == 3:
        age = list(range(rates["start_age"].min(), rates["end_age"].max() + 1))
        rate = np.zeros(len(age), dtype=float)
        formatted_rates = pd.DataFrame({"age": age, "rate": rate})

        for _, row in rates.iterrows():
            ages_within_interval = (formatted_rates["age"] >= row["start_age"]) & \
                                   (formatted_rates["age"] < row["end_age"])
            formatted_rates.loc[ages_within_interval, "rate"] = row["rate"] / len(formatted_rates[ages_within_interval])

        formatted_rates = pd.Series(data=formatted_rates["rate"].values, index=formatted_rates["age"], name="rate",
                                    dtype=float)
        formatted_rates.index.name = "age"

        return formatted_rates

    rates = pd.Series(data=rates["rate"].values, index=rates["age"], name="rate", dtype=float)
    rates.index.name = "age"

    return rates


def estimate_baseline_hazards(marginal_disease_incidence_rates: pd.Series, beta: np.ndarray, z: pd.DataFrame,
                              w: np.ndarray) -> pd.Series:
    """
    Baseline hazard: age-specific disease incidence rates when all covariates take their baseline values.
    """
    expected_risk_score_current = np.repeat(np.average(np.exp(z @ beta), weights=w),
                                            len(marginal_disease_incidence_rates))
    expected_risk_score_previous = expected_risk_score_current - 1
    epsilon = 1e-3

    while np.sum(np.abs(expected_risk_score_current - expected_risk_score_previous)) > epsilon:
        expected_risk_score_previous = expected_risk_score_current

        # Update baseline hazard using the new expected risk score
        baseline_hazards = marginal_disease_incidence_rates.values / expected_risk_score_previous

        # Update expected risk score using the new baseline hazard (under the proportional hazard model)
        weight_matrix = np.repeat(w.reshape(-1, 1), len(baseline_hazards), axis=1)
        hazard_scale = -np.exp(z.values @ beta)
        cumsum_baseline_hazards = np.concatenate((np.array([0.]), np.cumsum(baseline_hazards)[:-1]))
        numerator = np.exp(hazard_scale.reshape(-1, 1) @ cumsum_baseline_hazards.reshape(1, -1)) * weight_matrix
        denominator = np.sum(numerator, axis=0)
        probability_z_given_t = numerator / denominator
        expected_risk_score_current = np.sum(probability_z_given_t * np.exp(z.values @ beta).reshape(-1, 1), axis=0)

    baseline_hazards = marginal_disease_incidence_rates.copy(deep=True) / expected_risk_score_current

    return baseline_hazards


def calculate_absolute_risk_inner_integral(age_range: np.ndarray, age_mask: np.ndarray, baseline_hazards: pd.Series,
                                           competing_incidence_rates: pd.Series, betas: np.ndarray,
                                           z: pd.DataFrame) -> np.ndarray:
    baseline_hazards_matrix = np.repeat(
        baseline_hazards.loc[age_range].values.reshape(1, -1), len(z), axis=0) * age_mask
    competing_incidence_rates_matrix = np.repeat(
        competing_incidence_rates.loc[age_range].values.reshape(1, -1), len(z), axis=0) * age_mask

    inner_integral = baseline_hazards_matrix * np.exp(z @ betas).values.reshape(-1, 1) + \
                     competing_incidence_rates_matrix
    inner_integral = np.cumsum(inner_integral, axis=1) * age_mask
    inner_integral = np.concatenate((np.zeros((inner_integral.shape[0], 1)), inner_integral[:, :-1]), axis=1)

    return inner_integral


def estimate_absolute_risks(age_interval_starts: np.ndarray, age_interval_ends: np.ndarray,
                            baseline_hazards: pd.Series, competing_incidence_rates: pd.Series, betas: np.ndarray,
                            z: pd.DataFrame) -> np.ndarray:
    age_range = np.arange(np.min(age_interval_starts), np.max(age_interval_ends) + 1)
    age_range_matrix = np.repeat(age_range.reshape(1, -1), len(z), axis=0)
    age_mask = ((age_range_matrix >= age_interval_starts.reshape(-1, 1)) &
                (age_range_matrix < age_interval_ends.reshape(-1, 1))).astype(float)
    log_baseline_hazards_age_range = np.log(baseline_hazards.loc[age_range].values)

    # Calculate the inner integral
    inner_integral = calculate_absolute_risk_inner_integral(age_range, age_mask, baseline_hazards,
                                                            competing_incidence_rates, betas, z)

    # Calculate outer integral
    log_baseline_hazards_matrix = np.repeat(log_baseline_hazards_age_range.reshape(1, -1), len(z), axis=0)
    linear_predictor_matrix = np.repeat((z @ betas).values.reshape(-1, 1), len(age_range), axis=1)
    absolute_risks = np.sum(
        (np.exp(log_baseline_hazards_matrix + linear_predictor_matrix - inner_integral) * age_mask), axis=1)

    return absolute_risks


def get_significant_digits(values: np.ndarray) -> np.ndarray:
    MIN_STD_DEV = 1e-12
    standard_deviation = np.std(values)
    if standard_deviation < MIN_STD_DEV:
        standard_deviation = MIN_STD_DEV

    precision = 1. / (0.001 * standard_deviation)
    significant_digits = np.sum(precision >= 10 ** np.arange(1, 17))

    return significant_digits


def round_down_to_significant_digits(values: np.ndarray, significant_digits: np.ndarray) -> np.ndarray:
    return np.floor(values * 10 ** significant_digits) / 10 ** significant_digits


def get_cutpoints(values: pd.Series, quantiles: np.ndarray) -> np.ndarray:
    cutpoints = np.quantile(values, quantiles)
    cutpoints = round_down_to_significant_digits(cutpoints, get_significant_digits(cutpoints))
    if cutpoints.shape[0] < 2:
        raise ValueError(f"ERROR: Calculating cut-points for model-free imputation of missing values led to only "
                         f"{cutpoints.shape[0]} values. At least 2 cut-points are necessary to proceed. The "
                         f"calculated cut-points were: {cutpoints}.")
    cutpoints = np.append(np.unique(cutpoints), np.inf)
    return cutpoints


def assign_value_to_quantile(value: float, cutpoints: np.ndarray) -> Tuple[int, int]:
    cutpoint_lower_index = np.where(cutpoints <= value)[0][-1]
    cutpoint_upper_index = cutpoint_lower_index + 1
    return cutpoint_lower_index, cutpoint_upper_index


def get_samples_within_range(values: pd.Series, lower: float, upper: float) -> pd.Series:
    return values[(values >= lower) & (values < upper)]


def get_samples_from_expanded_quantile_range(values: pd.Series, lower_index: int, upper_index: int) -> pd.Series:
    lower_index -= 1
    upper_index += 1

    if lower_index < 0:
        lower_index = 0
    if upper_index > len(values) - 1:
        upper_index = len(values) - 1

    return values[(values >= lower_index) & (values < upper_index)]


def model_free_impute_absolute_risk(age_interval_starts: np.ndarray, age_interval_ends: np.ndarray,
                                    baseline_hazards: pd.Series, competing_incidence_rates: pd.Series,
                                    betas: np.ndarray, z_profile: pd.DataFrame, population_distribution: pd.DataFrame,
                                    population_weights: np.ndarray, num_imputations: int) -> Tuple[
        np.ndarray, np.ndarray]:
    profile_risks = np.zeros(len(age_interval_starts))
    profile_linear_predictors = np.zeros(len(age_interval_starts))
    population_linear_predictors = population_distribution @ betas

    NUM_CUTS = 100
    quantiles = np.arange(0.0, 1.0 + 1. / NUM_CUTS, 1. / NUM_CUTS)

    age_intervals = np.stack((age_interval_starts, age_interval_ends)).T
    unique_age_intervals = np.unique(age_intervals, axis=0)

    for (age_interval_start, age_interval_end) in unique_age_intervals:
        population_risks = estimate_absolute_risks(
            np.repeat(age_interval_start, len(population_distribution)),
            np.repeat(age_interval_end, len(population_distribution)), baseline_hazards, competing_incidence_rates,
            betas, population_distribution)

        profile_indices = np.where(np.all(age_intervals == (age_interval_start, age_interval_end), axis=1))[0]
        for profile_index in profile_indices:
            z = z_profile.iloc[profile_index].values
            variables_observed = np.where(~np.isnan(z))[0]
            betas_observed = betas[variables_observed]

            # If no variables are observed, the profile risk is set to the population average.
            if len(variables_observed) == 0:
                imputation_averaged_population_risks = population_risks.reshape(
                    -1, num_imputations, order="F").mean(axis=1)
                imputation_averaged_population_linear_predictors = population_linear_predictors.values.reshape(
                    -1, num_imputations, order="F").mean(axis=1)

                profile_risks[profile_index] = np.average(
                    imputation_averaged_population_risks,
                    weights=population_weights[:len(imputation_averaged_population_risks)])
                profile_linear_predictors[profile_index] = np.average(
                    imputation_averaged_population_linear_predictors, weights=population_weights)

                continue

            population_linear_predictors_observed = population_distribution.iloc[:, variables_observed] @ betas_observed
            profile_linear_predictors_observed = z[variables_observed] @ betas_observed

            cutpoints_population_linear_predictors_observed = get_cutpoints(
                population_linear_predictors_observed, quantiles)
            cutpoint_lower_index, cutpoint_upper_index = assign_value_to_quantile(
                profile_linear_predictors_observed, cutpoints_population_linear_predictors_observed)
            population_within_range = get_samples_within_range(
                population_linear_predictors_observed,
                cutpoints_population_linear_predictors_observed[cutpoint_lower_index],
                cutpoints_population_linear_predictors_observed[cutpoint_upper_index])

            while len(population_within_range) == 0:
                if cutpoint_lower_index == 0 and \
                        cutpoint_upper_index == len(cutpoints_population_linear_predictors_observed) - 1:
                    raise ValueError(f"ERROR: During model-free imputation, no samples were found in the reference "
                                     f"population using only the observed features in the profile ID: "
                                     f"{z_profile.index[profile_index]}.")
                population_within_range = get_samples_from_expanded_quantile_range(
                    population_linear_predictors_observed, cutpoint_lower_index, cutpoint_upper_index)

            selected = [population_linear_predictors.index.get_loc(index) for index in population_within_range.index]
            profile_risks[profile_index] = np.average(population_risks[selected], weights=population_weights[selected])
            profile_linear_predictors[profile_index] = np.average(population_linear_predictors[selected],
                                                                  weights=population_weights[selected])

    return profile_risks, profile_linear_predictors


def calculate_population_risks(age_interval_starts: np.ndarray, age_interval_ends: np.ndarray,
                               baseline_hazards: pd.Series, competing_incidence_rates: pd.Series,
                               betas: np.ndarray, population_distribution: pd.DataFrame,
                               num_imputations: int) -> pd.DataFrame:
    age_intervals = np.stack((age_interval_starts, age_interval_ends)).T
    unique_age_intervals = np.unique(age_intervals, axis=0)
    population_risks = np.zeros((len(unique_age_intervals), 2 + population_distribution.shape[0]))

    interval_id = 0
    for (age_interval_start, age_interval_end) in unique_age_intervals:
        population_risks_in_interval = estimate_absolute_risks(
            np.repeat(age_interval_start, len(population_distribution)),
            np.repeat(age_interval_end, len(population_distribution)), baseline_hazards, competing_incidence_rates,
            betas, population_distribution)
        population_risks[interval_id, 0] = age_interval_start
        population_risks[interval_id, 1] = age_interval_end
        population_risks[interval_id, 2:] = population_risks_in_interval.reshape(
            -1, num_imputations, order="F").mean(axis=1)
        interval_id += 1

    population_risks = pd.DataFrame(
        population_risks, columns=["age_interval_start", "age_interval_end", *population_distribution.index])

    return population_risks


class AbsoluteRiskModel:
    """Absolute risk model"""
    age_start: Union[int, List[int]]
    age_interval_length: Union[int, List[int]]
    beta_estimates: np.ndarray
    profile: pd.DataFrame
    z_profile: pd.DataFrame
    population_distribution: pd.DataFrame
    population_weights: np.ndarray
    num_imputations: int

    baseline_hazards: pd.Series
    competing_incidence_rates: pd.Series

    return_population_risks: bool
    results: AbsoluteRiskResults

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
                 snp_profile_path: Union[str, pathlib.Path, None],
                 return_reference_risks: bool) -> None:
        covariate_model: Optional[CovariateModel] = None
        snp_model: Optional[SnpModel] = None
        self.num_imputations = 1

        check_errors.check_return_population_risks_type(return_reference_risks)
        self.return_population_risks = return_reference_risks

        check_errors.check_age_interval_types(apply_age_start, apply_age_interval_length)
        self.age_start, self.age_interval_length = apply_age_start, apply_age_interval_length

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
            check_errors.check_num_imputations(num_imputations)
            self.num_imputations = num_imputations

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
        self._set_profile(covariate_model, snp_model)
        self._set_baseline_hazards(age_specific_disease_incidence_rates_path)
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
            self.z_profile.index = covariate_model.z_profile.index
        elif snp_model is not None:
            self.z_profile = snp_model.z_profile
            self.z_profile.index = snp_model.z_profile.index
        else:
            raise ValueError("ERROR: No query profile design matrices were set. Please pass inputs for arguments "
                             "'apply_covariate_profile_path' and/or 'apply_snp_profile_path'.")

    def _set_baseline_hazards(self, marginal_disease_incidence_rates_path: Union[str, pathlib.Path]) -> None:
        marginal_disease_incidence_rates = utils.read_file_to_dataframe(marginal_disease_incidence_rates_path)
        check_errors.check_rate_format(marginal_disease_incidence_rates, "model_disease_incidence_rates_path")
        marginal_disease_incidence_rates = format_rates(marginal_disease_incidence_rates)
        check_errors.check_rate_covers_all_ages(marginal_disease_incidence_rates, self.age_start,
                                                self.age_interval_length, "model_disease_incidence_rates_path")
        self.baseline_hazards = estimate_baseline_hazards(marginal_disease_incidence_rates, self.beta_estimates,
                                                          self.population_distribution, self.population_weights)

    def _set_competing_incidence_rates(self, competing_incidence_rates_path: Union[str, pathlib.Path, None]):
        if competing_incidence_rates_path is None:
            self.competing_incidence_rates = pd.Series(data=np.zeros(len(self.baseline_hazards)),
                                                       index=self.baseline_hazards.index, name="rate", dtype=float)
            self.competing_incidence_rates.index.name = "age"
        else:
            competing_incidence_rates = utils.read_file_to_dataframe(competing_incidence_rates_path)
            check_errors.check_rate_format(competing_incidence_rates, "model_competing_incidence_rates_path")
            competing_incidence_rates = format_rates(competing_incidence_rates)
            check_errors.check_rate_covers_all_ages(competing_incidence_rates, self.age_start, self.age_interval_length,
                                                    "model_competing_incidence_rates_path")
            self.competing_incidence_rates = competing_incidence_rates

    def compute_absolute_risks(self) -> AbsoluteRiskResults:
        self.results = AbsoluteRiskResults()
        self.results.set_ages(self.age_start, self.age_interval_length)
        linear_predictors = self.z_profile @ self.beta_estimates
        self.results.set_linear_predictors(linear_predictors, list(self.z_profile.index))

        risk_estimates = np.full(len(self.z_profile), np.nan)

        profiles_without_missing_values = ~linear_predictors.isnull()
        if profiles_without_missing_values.any():
            risk_estimates[profiles_without_missing_values] = estimate_absolute_risks(
                self.results.age_interval_start[profiles_without_missing_values],
                self.results.age_interval_end[profiles_without_missing_values], self.baseline_hazards,
                self.competing_incidence_rates, self.beta_estimates,
                self.z_profile.loc[profiles_without_missing_values])

        profiles_with_missing_values = linear_predictors.isnull()
        if profiles_with_missing_values.any():
            risk_estimates[profiles_with_missing_values], linear_predictors[profiles_with_missing_values] = \
                model_free_impute_absolute_risk(
                    self.results.age_interval_start[profiles_with_missing_values],
                    self.results.age_interval_end[profiles_with_missing_values], self.baseline_hazards,
                    self.competing_incidence_rates, self.beta_estimates,
                    self.z_profile.loc[profiles_with_missing_values], self.population_distribution,
                    self.population_weights, self.num_imputations)

        self.results.set_risk_estimates(risk_estimates, list(self.z_profile.index))
        self.results.set_linear_predictors(linear_predictors, list(self.z_profile.index))

        if self.return_population_risks:
            population_risk_estimates = calculate_population_risks(
                self.results.age_interval_start, self.results.age_interval_end, self.baseline_hazards,
                self.competing_incidence_rates, self.beta_estimates, self.population_distribution, self.num_imputations)
            self.results.set_population_risk_estimates(population_risk_estimates)

        return self.results

    def _set_profile(self, covariate_model: Optional[CovariateModel], snp_model: Optional[SnpModel]) -> None:
        if covariate_model is not None and snp_model is not None:
            check_errors.check_profiles(covariate_model.profile, snp_model.profile)
            self.profile = pd.concat([covariate_model.profile, snp_model.profile], axis=1)
            self.profile.index = covariate_model.profile.index
        elif covariate_model is not None:
            self.profile = covariate_model.profile
            self.profile.index = covariate_model.profile.index
        elif snp_model is not None:
            self.profile = snp_model.profile
            self.profile.index = snp_model.profile.index
        else:
            raise ValueError("ERROR: No query profiles were set. Please pass inputs for arguments "
                             "'apply_covariate_profile_path' and/or 'apply_snp_profile_path'.")

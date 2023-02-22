from typing import Tuple

import numpy as np
import pandas as pd

from icare import check_errors
from icare.absolute_risk_model import estimate_absolute_risks


def get_significant_digits(values: np.ndarray) -> np.ndarray:
    MIN_STD_DEV = 1e-12
    standard_deviation = np.std(values)
    if standard_deviation < MIN_STD_DEV:
        standard_deviation = MIN_STD_DEV

    precision = 1. / (0.001 * standard_deviation)
    significant_digits = np.sum(precision >= 10 ** np.arange(1, 17))

    return significant_digits


def round_down_to_significant_digits(values: np.ndarray, significant_digits: np.ndarray) -> np.ndarray:
    return np.floor(values * 10**significant_digits) / 10**significant_digits


def get_cutpoints(values: pd.Series, quantiles: np.ndarray) -> np.ndarray:
    cutpoints = np.quantile(values, quantiles)
    cutpoints = round_down_to_significant_digits(cutpoints, get_significant_digits(cutpoints))
    check_errors.check_cutpoints(cutpoints)
    cutpoints = np.append(np.unique(cutpoints), np.inf)
    return cutpoints


def assign_value_to_quantile(value: float, cutpoints: np.ndarray) -> Tuple[int, int]:
    cutpoint_lower_index = np.where(cutpoints <= value)[0][-1]
    cutpoint_upper_index = cutpoint_lower_index + 1
    return cutpoints[cutpoint_lower_index], cutpoints[cutpoint_upper_index]


def get_samples_within_range(values: pd.Series, lower: float, upper: float) -> pd.Series:
    return values[(values >= lower) & (values < upper)]


def model_free_impute_absolute_risk(age_interval_starts: np.ndarray, age_interval_ends: np.ndarray,
                                    baseline_hazards: pd.Series, competing_incidence_rates: pd.Series,
                                    betas: np.ndarray, z_profile: pd.DataFrame, population_distribution: pd.DataFrame,
                                    population_weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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

            # TODO: If no variables are observed, the profile risk is set to population average.
            if len(variables_observed) == 0:
                profile_risks[profile_index] = np.average(population_risks, weights=population_weights)
                profile_linear_predictors[profile_index] = np.average(population_linear_predictors,
                                                                      weights=population_weights)
                continue

            population_linear_predictors_observed = population_distribution.iloc[:, variables_observed] @ betas_observed
            profile_linear_predictors_observed = z[variables_observed] @ betas_observed
            
            cutpoints_population_linear_predictors_observed = get_cutpoints(
                population_linear_predictors_observed, quantiles)
            cutpoint_lower, cutpoint_upper = assign_value_to_quantile(
                profile_linear_predictors_observed, cutpoints_population_linear_predictors_observed)
            population_within_range = get_samples_within_range(
                population_linear_predictors_observed, cutpoint_lower, cutpoint_upper)

            # while loop
            selected = [population_linear_predictors.index.get_loc(index) for index in population_within_range.index]
            profile_risks[profile_index] = np.average(population_risks[selected], weights=population_weights[selected])
            profile_linear_predictors[profile_index] = np.average(population_linear_predictors[selected],
                                                                  weights=population_weights[selected])

    return profile_risks, profile_linear_predictors

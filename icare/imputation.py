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


def get_cutpoints(values: pd.Series, quantiles: np.ndarray) -> np.ndarray:
    cutpoints = np.quantile(values, quantiles)
    cutpoints = np.round(cutpoints, get_significant_digits(cutpoints))
    cutpoints = np.append(np.unique(cutpoints), np.inf)
    check_errors.check_cutpoints(cutpoints)
    return cutpoints


def model_free_impute_absolute_risk(age_interval_starts: np.ndarray, age_interval_ends: np.ndarray,
                                    baseline_hazards: pd.Series, competing_incidence_rates: pd.Series,
                                    betas: np.ndarray, z_profile: pd.DataFrame, population_distribution: pd.DataFrame,
                                    population_weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    estimated_risks = np.zeros(len(age_interval_starts))
    estimated_linear_predictors = np.zeros(len(age_interval_starts))

    NUM_CUTS = 100
    quantiles = np.arange(0.0, 1.0 + 1. / NUM_CUTS, 1. / NUM_CUTS)

    population_linear_predictors = population_distribution.values @ betas
    age_intervals = np.stack((age_interval_starts, age_interval_ends)).T
    unique_age_intervals = np.unique(age_intervals, axis=0)

    for (age_interval_start, age_interval_end) in unique_age_intervals:
        population_risks = estimate_absolute_risks(
            np.repeat(age_interval_start, len(population_distribution)),
            np.repeat(age_interval_end, len(population_distribution)), baseline_hazards, competing_incidence_rates,
            betas, population_distribution)

        profile_indices = np.where(np.all(age_intervals == (age_interval_start, age_interval_end), axis=1))[0]
        for index in profile_indices:
            z = z_profile.iloc[index].values
            variables_observed = np.where(~np.isnan(z))[0]
            betas_observed = betas[variables_observed]

            # TODO: If no variables are observed, the profile risk is set to population average.
            if len(variables_observed) == 0:
                estimated_risks[index] = np.average(population_risks, weights=population_weights)
                estimated_linear_predictors[index] = np.average(population_linear_predictors, weights=population_weights)

            population_linear_predictors_observed = population_distribution.iloc[:, variables_observed] @ betas_observed
            profile_linear_predictor_observed = z[variables_observed] @ betas_observed
            
            cutpoints_population_linear_predictors_observed = get_cutpoints(
                population_linear_predictors_observed, quantiles)

    return estimated_risks, estimated_linear_predictors

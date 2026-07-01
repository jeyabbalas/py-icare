import numpy as np
import pandas as pd

from icare.absolute_risk_model import AbsoluteRiskModel
from icare.model_validation import ModelValidation


def _serialize_df(df: pd.DataFrame, output_format: str):
    """Serialize a result DataFrame for the requested output format.

    ``'json'`` reproduces the legacy records-oriented JSON string; ``'dataframe'`` returns the DataFrame
    itself with a fresh RangeIndex so its columns match ``pd.read_json(..., orient='records')`` of the
    JSON form.
    """
    if output_format == 'json':
        return df.to_json(orient='records')
    return df.reset_index(drop=True)


def _format_reference_risks(reference_risks: list, output_format: str) -> list:
    """Return a new reference-risks list with per-interval ``population_risks`` in the requested form.

    ``'dataframe'`` yields a contiguous float64 ``np.ndarray`` (cheap to hand to the JS SDK as a typed
    array); ``'json'`` yields a plain ``list`` of floats. New dicts are built so the model's data is
    never mutated in place.
    """
    if output_format == 'dataframe':
        return [{**interval, 'population_risks': np.asarray(interval['population_risks'], dtype=float)}
                for interval in reference_risks]
    return [{**interval, 'population_risks': np.asarray(interval['population_risks'], dtype=float).tolist()}
            for interval in reference_risks]


def package_absolute_risk_results_to_dict(absolute_risk_model: AbsoluteRiskModel, return_linear_predictors: bool,
                                          return_reference_risks: bool, method_name: str,
                                          output_format: str = 'json') -> dict:
    results = dict()

    results['model'] = dict(zip(absolute_risk_model.population_distribution.columns.tolist(),
                                absolute_risk_model.beta_estimates.tolist()))

    profile = absolute_risk_model.profile.copy(deep=True)
    if return_linear_predictors:
        profile.insert(0, 'linear_predictors', absolute_risk_model.results.linear_predictors)
    profile.insert(0, 'risk_estimates', absolute_risk_model.results.risk_estimates)
    profile.insert(0, 'age_interval_end', absolute_risk_model.results.age_interval_end)
    profile.insert(0, 'age_interval_start', absolute_risk_model.results.age_interval_start)
    profile.insert(0, 'id', profile.index)
    results['profile'] = _serialize_df(profile, output_format)

    if return_reference_risks:
        if output_format == 'json':
            results['reference_risks'] = absolute_risk_model.results.population_risks_per_interval
        else:
            results['reference_risks'] = _format_reference_risks(
                absolute_risk_model.results.population_risks_per_interval, output_format)

    results['method'] = method_name

    return results


def combine_split_absolute_risk_results(results_before_cutpoint: dict, results_after_cutpoint: dict,
                                        return_linear_predictors: bool, return_reference_risks: bool,
                                        method_name: str, output_format: str = 'json') -> dict:
    results = dict()

    results['model'] = dict()
    results['model']['before_cutpoint'] = results_before_cutpoint['model']
    results['model']['after_cutpoint'] = results_after_cutpoint['model']

    # The children are computed with output_format='dataframe', so their profiles are DataFrames — operate
    # on them directly (no JSON round-trip, which previously re-parsed dtypes and rounded to 10 digits).
    profile_before_cutpoint = results_before_cutpoint['profile'].copy(deep=True)
    if 'id' in profile_before_cutpoint.columns:
        profile_before_cutpoint.set_index('id', inplace=True)

    profile_after_cutpoint = results_after_cutpoint['profile'].copy(deep=True)
    if 'id' in profile_after_cutpoint.columns:
        profile_after_cutpoint.set_index('id', inplace=True)

    profile = pd.DataFrame(index=profile_before_cutpoint.index)
    profile.index.name = profile_before_cutpoint.index.name

    profile['age_interval_start'] = profile_before_cutpoint['age_interval_start']
    profile['cutpoint'] = profile_after_cutpoint['age_interval_start']
    profile['age_interval_end'] = profile_after_cutpoint['age_interval_end']
    profile['age_interval_length'] = profile['age_interval_end'] - profile['age_interval_start']

    if return_linear_predictors:
        profile['linear_predictors_before_cutpoint'] = profile_before_cutpoint['linear_predictors']
        profile['linear_predictors_after_cutpoint'] = profile_after_cutpoint['linear_predictors']

    profile['risk_estimates'] = \
        profile_before_cutpoint['risk_estimates'] + \
        (1 - profile_before_cutpoint['risk_estimates']) * profile_after_cutpoint['risk_estimates']

    drop_columns = ['age_interval_start', 'age_interval_end', 'risk_estimates']
    if return_linear_predictors:
        drop_columns.append('linear_predictors')
    profile_before_cutpoint.drop(columns=drop_columns, inplace=True)
    profile_after_cutpoint.drop(columns=drop_columns, inplace=True)

    for column in profile_before_cutpoint.columns:
        if column in profile_after_cutpoint.columns:
            if profile_before_cutpoint[column].equals(profile_after_cutpoint[column]):
                profile[column] = profile_before_cutpoint[column]
            else:
                profile[column + '_before_cutpoint'] = profile_before_cutpoint[column]
                profile[column + '_after_cutpoint'] = profile_after_cutpoint[column]
        else:
            profile[column] = profile_before_cutpoint[column]

    for column in profile_after_cutpoint.columns:
        if column not in profile_before_cutpoint.columns:
            profile[column] = profile_after_cutpoint[column]

    profile.insert(0, 'id', profile.index)
    results['profile'] = _serialize_df(profile, output_format)

    if return_reference_risks:
        results['reference_risks'] = dict()
        results['reference_risks']['before_cutpoint'] = _format_reference_risks(
            results_before_cutpoint['reference_risks'], output_format)
        results['reference_risks']['after_cutpoint'] = _format_reference_risks(
            results_after_cutpoint['reference_risks'], output_format)

    results['method'] = method_name

    return results


def package_validation_results_to_dict(model_validation: ModelValidation, method_name: str,
                                       output_format: str = 'json') -> dict:
    results = dict()

    results['info'] = model_validation.results.info
    study_data = model_validation.study_data.copy(deep=True)
    study_data.insert(0, 'id', study_data.index)
    results['study_data'] = _serialize_df(study_data, output_format)
    if model_validation.results.reference is not None:
        results['reference'] = model_validation.results.reference
    results['incidence_rates'] = _serialize_df(model_validation.results.incidence_rates, output_format)
    results['auc'] = model_validation.results.auc
    results['brier_score'] = model_validation.results.brier_score
    results['expected_by_observed_ratio'] = model_validation.results.expected_by_observed_ratio
    results['calibration'] = model_validation.results.calibration
    category_specific_calibration = model_validation.results.category_specific_calibration.copy(deep=True)
    category_specific_calibration.insert(0, 'category', category_specific_calibration.index)
    results['category_specific_calibration'] = _serialize_df(category_specific_calibration, output_format)
    results['method'] = method_name

    return results

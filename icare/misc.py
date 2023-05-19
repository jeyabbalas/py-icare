import pandas as pd

from icare.absolute_risk_model import AbsoluteRiskModel
from icare.model_validation import ModelValidation


def package_absolute_risk_results_to_dict(absolute_risk_model: AbsoluteRiskModel, return_linear_predictors: bool,
                                          return_reference_risks: bool, method_name: str) -> dict:
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
    results['profile'] = profile.to_json(orient='records')

    if return_reference_risks:
        results['reference_risks'] = absolute_risk_model.results.population_risks_per_interval

    results['method'] = method_name

    return results


def combine_split_absolute_risk_results(results_before_cutpoint: dict, results_after_cutpoint: dict,
                                        return_linear_predictors: bool, return_reference_risks: bool,
                                        method_name: str) -> dict:
    results = dict()

    results['model'] = dict()
    results['model']['before_cutpoint'] = results_before_cutpoint['model']
    results['model']['after_cutpoint'] = results_after_cutpoint['model']

    profile_before_cutpoint = pd.read_json(results_before_cutpoint['profile'], orient='records')
    if 'id' in profile_before_cutpoint.columns:
        profile_before_cutpoint.set_index('id', inplace=True)

    profile_after_cutpoint = pd.read_json(results_after_cutpoint['profile'], orient='records')
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
    results['profile'] = profile.to_json(orient='records')

    if return_reference_risks:
        results['reference_risks'] = dict()
        results['reference_risks']['before_cutpoint'] = results_before_cutpoint['reference_risks']
        results['reference_risks']['after_cutpoint'] = results_after_cutpoint['reference_risks']

    results['method'] = method_name

    return results


def package_validation_results_to_dict(model_validation: ModelValidation, method_name: str) -> dict:
    results = dict()

    results['risk_prediction_interval'] = model_validation.results.risk_prediction_interval
    results['reference'] = model_validation.results.reference
    results['incidence_rates'] = model_validation.results.incidence_rates.to_json(orient='records')
    results['auc'] = model_validation.results.auc
    results['expected_by_observed_ratio'] = model_validation.results.expected_by_observed_ratio
    results['calibration'] = model_validation.results.calibration
    results['category_specific_calibration'] = model_validation.results.category_specific_calibration.to_json(
        orient='records')
    results['dataset_name'] = model_validation.results.dataset_name
    results['model_name'] = model_validation.results.model_name

    results['method'] = method_name

    return results

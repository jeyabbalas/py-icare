from icare.absolute_risk_model import AbsoluteRiskModel
from icare.model_validation import ModelValidation


def package_absolute_risk_results_to_dict(absolute_risk_model: AbsoluteRiskModel, return_linear_predictors: bool,
                                          return_reference_risks: bool) -> dict:
    results = dict()

    results["model"] = dict(zip(absolute_risk_model.population_distribution.columns.tolist(),
                                absolute_risk_model.beta_estimates.tolist()))

    profile = absolute_risk_model.profile.copy(deep=True)
    if return_linear_predictors:
        profile.insert(0, "linear_predictors", absolute_risk_model.results.linear_predictors)
    profile.insert(0, "risk_estimates", absolute_risk_model.results.risk_estimates)
    profile.insert(0, "age_interval_end", absolute_risk_model.results.age_interval_end)
    profile.insert(0, "age_interval_start", absolute_risk_model.results.age_interval_start)
    profile.insert(0, "id", profile.index)
    results["profile"] = profile.to_json(orient="records")

    if return_reference_risks:
        results["reference_risks"] = absolute_risk_model.results.population_risks_per_interval

    return results


def package_validation_results_to_dict(model_validation: ModelValidation) -> dict:
    results = dict()

    results["risk_prediction_interval"] = model_validation.timeframe
    results["dataset_name"] = model_validation.dataset_name
    results["model_name"] = model_validation.model_name

    return results

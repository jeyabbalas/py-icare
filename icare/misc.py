import numpy as np
import pandas as pd

from icare.absolute_risk_model import AbsoluteRiskModel


def package_results(final_risks, z_new, model_includes_covariates, handle_snps, apply_age_start,
                    apply_age_interval_length, apply_covariates_profile, model_log_relative_risk, beta_est,
                    apply_snp_profile, snp_names, return_lp, lps):
    result = dict()

    result["risk_estimate"] = final_risks.tolist()

    if model_includes_covariates:
        if handle_snps:
            data = np.concatenate(
                [apply_age_start.reshape(-1, 1),
                 (apply_age_start + apply_age_interval_length).reshape(-1, 1),
                 final_risks.reshape(-1, 1),
                 apply_snp_profile,
                 apply_covariates_profile],
                axis=1
            )
            columns = ["interval_start", "interval_end", "risk_estimate", *snp_names, *apply_covariates_profile.columns]
            result["details"] = pd.DataFrame(data=data, columns=columns).to_json(orient="records")
            beta_names = [*snp_names, *model_log_relative_risk["covariate_name"]]
        else:
            data = np.concatenate(
                [apply_age_start.reshape(-1, 1),
                 (apply_age_start + apply_age_interval_length).reshape(-1, 1),
                 final_risks.reshape(-1, 1),
                 apply_covariates_profile],
                axis=1
            )
            columns = ["interval_start", "interval_end", "risk_estimate", *apply_covariates_profile.columns]
            result["details"] = pd.DataFrame(data=data, columns=columns).to_json(orient="records")
            beta_names = [*model_log_relative_risk["covariate_name"]]
    else:
        data = np.concatenate(
            [apply_age_start.reshape(-1, 1),
             (apply_age_start + apply_age_interval_length).reshape(-1, 1),
             final_risks.reshape(-1, 1),
             apply_snp_profile],
            axis=1
        )
        columns = ["interval_start", "interval_end", "risk_estimate", *snp_names]
        result["details"] = pd.DataFrame(data=data, columns=columns).to_json(orient="records")
        beta_names = [*snp_names]

    result["beta_used"] = pd.DataFrame(
        data=np.concatenate([np.array(beta_names).reshape(-1, 1), beta_est.reshape(-1, 1)], axis=1),
        columns=["variable_name", "log_OR_used"]
    ).to_json(orient="records")

    if return_lp:
        result["lps"] = lps.tolist()

    return result


def package_absolute_risk_results_to_dict(absolute_risk_model: AbsoluteRiskModel, return_linear_predictors: bool,
                                          return_reference_risks: bool) -> dict:
    results = dict()

    results["beta_used"] = absolute_risk_model.beta_estimates.tolist()

    profile = absolute_risk_model.profile.copy(deep=True)
    if return_linear_predictors:
        profile.insert(0, "linear_predictors", absolute_risk_model.results.linear_predictors)
    profile.insert(0, "risk_estimates", absolute_risk_model.results.risk_estimates)
    profile.insert(0, "age_interval_end", absolute_risk_model.results.age_interval_end)
    profile.insert(0, "age_interval_start", absolute_risk_model.results.age_interval_start)
    profile.insert(0, "id", profile.index)
    results["profile"] = profile.to_json(orient="records")

    if return_reference_risks:
        results["reference_risks"] = absolute_risk_model.results.population_risk_estimates.to_json(orient="records")

    return results

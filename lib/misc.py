import numpy as np
import pandas as pd


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
            result["details"] = pd.DataFrame(data=data, columns=columns).to_json()
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
            result["details"] = pd.DataFrame(data=data, columns=columns).to_json()
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
        result["details"] = pd.DataFrame(data=data, columns=columns).to_json()
        beta_names = [*snp_names]

    result["beta_used"] = pd.DataFrame(
        data=np.concatenate([np.array(beta_names).reshape(-1, 1), beta_est.reshape(-1, 1)], axis=1),
        columns=["variable_name", "log_OR_used"]
    ).to_json()

    if return_lp:
        result["lps"] = lps.tolist()

    return result

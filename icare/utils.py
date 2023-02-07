import json
import pathlib
from typing import Union, List, Optional, Tuple

import numpy as np
import pandas as pd

from icare import check_errors


def process_snp_info(model_includes_covariates, apply_snp_profile, model_family_history_binary_variable_name,
                     apply_covariates_profile, model_reference_dataset, model_snp_info):
    if model_includes_covariates:
        if apply_snp_profile is None:
            apply_snp_profile = pd.DataFrame(
                data=np.full((apply_covariates_profile.shape[0], model_snp_info.shape[0]), np.nan),
                columns=model_snp_info["snp_name"]
            )
            print("Note: You included 'model_snp_info', but did not provide an 'apply_snp_profile'. "
                  "So, values for all SNPs will be imputed.")

        if apply_snp_profile.shape[0] != apply_covariates_profile.shape[0]:
            raise ValueError("ERROR: 'apply_covariates_profile' and 'apply_snp_profile' must have the same "
                             "number of rows.")

        if model_family_history_binary_variable_name is not None:
            if model_family_history_binary_variable_name not in apply_covariates_profile.columns:
                raise ValueError("ERROR: 'model_family_history_binary_variable_name' must contain the variable name of "
                                 "family history (matching a column name in 'apply_covariates_profile') if it is in the"
                                 " model, otherwise set its value to None.")
            else:
                attenuate_fh = True
                fh_pop = model_reference_dataset[model_family_history_binary_variable_name].values
                fh_cov = apply_covariates_profile[model_family_history_binary_variable_name].values

                if not ((fh_pop == 0) | (fh_pop == 1) | (np.isnan(fh_pop))).all():
                    raise ValueError("ERROR: The family history must be binary when using 'model_snp_info' "
                                     "functionality. Check input for 'model_reference_dataset'.")

                if not ((fh_cov == 0) | (fh_cov == 1) | (np.isnan(fh_cov))).all():
                    raise ValueError("ERROR: The family history must be binary when using 'model_snp_info' "
                                     "functionality. Check input for 'apply_covariates_profile'.")

        else:
            attenuate_fh = False
            fh_pop = np.zeros((10_000,), dtype=int)
            print("Note: As specified, the model does not adjust SNP imputations for family history, since "
                  "'model_family_history_binary_variable_name' = None.")
    else:
        attenuate_fh = False
        fh_pop = np.zeros((10_000,), dtype=int)
        print("Note: As specified, the model does not adjust SNP imputations for family history.")

    return attenuate_fh, fh_pop, apply_snp_profile


def sim_snps(snp_betas, snp_freqs, fh_status):
    snps = np.full((fh_status.shape[0], snp_freqs.shape[0]), np.nan)
    prob012_fh_no = np.column_stack(
        ((1.0 - snp_freqs) ** 2,
         2 * snp_freqs * (1 - snp_freqs),
         snp_freqs ** 2)
    )
    beta_mat = np.repeat(snp_betas, 3).reshape(-1, 3)
    top = np.exp(beta_mat * (np.tile(np.array([0., 1., 2.]), snp_betas.shape[0]).reshape(-1, 3) / 2.0)) * prob012_fh_no
    bottom = np.repeat(np.sum(top, axis=1), 3).reshape(-1, 3)
    prob012_fh_yes = top / bottom

    fh_no = np.where(fh_status == 0)
    fh_yes = np.where(fh_status == 1)

    vals = np.random.default_rng() \
        .uniform(size=fh_status.shape[0] * snp_betas.shape[0]) \
        .reshape(fh_status.shape[0], snp_betas.shape[0])

    if fh_no[0].shape[0]:
        snps[fh_no] = np.array(vals[fh_no] > prob012_fh_no[:, 0], dtype=int) + \
                      np.array(vals[fh_no] > np.sum(prob012_fh_no[:, :2], axis=1), dtype=int)

    if fh_yes[0].shape[0]:
        snps[fh_yes] = np.array(vals[fh_yes] > prob012_fh_yes[:, 0], dtype=int) + \
                       np.array(vals[fh_yes] > np.sum(prob012_fh_yes[:, :2], axis=1), dtype=int)

    return snps


def survival_given_x(lambda_0, beta_est, pop_dist_mat):
    # produces a matrix with Nobs x Ntimes
    # cumulative up to but not including current time
    mult = -np.exp(np.matmul(pop_dist_mat, beta_est))
    cum_lam = np.cumsum(np.append(np.array([0.]), lambda_0["rate"].values))[:lambda_0["rate"].shape[0]]
    return np.exp(np.matmul(mult.reshape(-1, 1), cum_lam.reshape(1, -1)))


def precise_lambda0(lambda_vals, approx_expectation_rr, beta_est, pop_dist_mat, pop_weights):
    lambda_0 = lambda_vals.copy(deep=True)
    precise_expectation_rr0 = approx_expectation_rr - 1.0
    precise_expectation_rr1 = approx_expectation_rr

    i = 0
    while np.sum(np.abs(precise_expectation_rr1 - precise_expectation_rr0)) > 0.001:
        i = i + 1
        precise_expectation_rr0 = precise_expectation_rr1
        # new expectation rr implies lambda0
        lambda_0["rate"] = lambda_vals["rate"].values / precise_expectation_rr0
        # that lambda0 implies new expectation rr
        this_survival = survival_given_x(lambda_0, beta_est, pop_dist_mat) * \
                        np.repeat(pop_weights, lambda_0.shape[0]).reshape(pop_weights.shape[0], lambda_0.shape[0])
        deno = 1. / np.sum(this_survival, axis=0)
        prob_x_given_t = this_survival * deno

        # to compute next iteration
        precise_expectation_rr1 = np.sum((prob_x_given_t.T * np.exp(np.matmul(pop_dist_mat, beta_est))).T, axis=0)

    lambda_0["rate"] = lambda_vals["rate"] / precise_expectation_rr1

    return lambda_0, precise_expectation_rr1


def pick_lambda(t, lambda_vals):
    a = np.where(t == lambda_vals["age"])[0][0]
    return lambda_vals["rate"].iloc[a]


def get_int(a, t, lambda_vals, z_new, beta_est, model_competing_incidence_rates, z_beta=None):
    holder = 0

    if z_beta is None:
        z_beta = np.exp(np.matmul(z_new.T, beta_est)).T

    for u in range(np.nanmin(a), np.nanmax(t) + 1):
        factor = 1 if ((u >= a) and (u < t)) else 0
        idx = np.where(u == model_competing_incidence_rates["age"])[0][0]
        holder = holder + factor * ((pick_lambda(u, lambda_vals) * z_beta) +
                                    model_competing_incidence_rates["rate"].iloc[idx])

    return holder


def comp_a_j(z_new, apply_age_start, apply_age_interval_length, lambda_vals, beta_est, model_competing_incidence_rates):
    z_beta = np.exp(np.matmul(beta_est.T, z_new)).T
    a_vec = apply_age_start + apply_age_interval_length
    t_vec = np.array(range(np.nanmin(apply_age_start), np.nanmax(a_vec) + 1))
    t_idxs_in_lambda = [x for i, x in enumerate(t_vec) if x in lambda_vals["age"]]
    c2 = np.matmul(z_new.T, beta_est)
    c3 = np.log(lambda_vals.iloc[t_idxs_in_lambda]["rate"].values)

    a_j_est = np.zeros((z_new.shape[1],))
    for i, t in enumerate(t_vec):
        temp = get_int(apply_age_start, t, lambda_vals, z_new, beta_est, model_competing_incidence_rates, z_beta=z_beta)
        factor = np.prod(np.array(t >= apply_age_start, dtype=int) * np.array(t < a_vec, dtype=int))
        this_year = factor * np.exp(c3[i] + c2 - temp)
        a_j_est = a_j_est + this_year

    return a_j_est


def handle_missing_data(apply_age_start, apply_age_interval_length, z_new, miss, present, n_cuts, final_risks, ref_pop,
                        pop_weights, lambda_0, beta_est, model_competing_incidence_rates, lps):
    ref_full_lp = np.matmul(ref_pop.T, beta_est)

    ###### Handle Missing Data  ##### if all times are the same
    if (np.unique(apply_age_start[miss]).shape[0] == 1) and (np.unique(apply_age_interval_length[miss]).shape[0] == 1):
        # change to single values so don't have to worry about dimension of ref_pop
        pop_apply_age_start = np.unique(apply_age_start)
        pop_apply_age_interval_length = np.unique(apply_age_interval_length)

        ###### Compute A_j_pop for ref_risks
        ref_risks = comp_a_j(
            ref_pop, pop_apply_age_start, pop_apply_age_interval_length,
            lambda_0, beta_est, model_competing_incidence_rates
        )

        probs = np.arange(0.0, 1.001, 1. / n_cuts)

        for miss_i in miss:
            # make sure LPs based on non-missing covariates for the observation with missing
            present = np.where(~np.isnan(z_new[:, miss_i]))[0]
            if present.shape[0] == 0:
                final_risks[miss_i] = 0

    ###### Handle Missing Data  ##### if all times are different

    return final_risks, lps


def get_refs_risk(ref_pop, apply_age_start, apply_age_interval_length, lambda_0, beta_est,
                  model_competing_incidence_rates, handle_snps, n_imp):
    refs_risk = comp_a_j(
        ref_pop, apply_age_start[0], apply_age_interval_length[0],
        lambda_0, beta_est, model_competing_incidence_rates
    )
    refs_lps = np.matmul(ref_pop.T, beta_est)

    if handle_snps:
        refs_risk = np.mean(refs_risk.reshape((-1, n_imp), order="F"), axis=1)
        refs_lps = np.mean(refs_lps.reshape((-1, n_imp), order="F"), axis=1)

    return refs_risk, refs_lps


def read_file_to_string(file: Union[str, pathlib.Path]) -> str:
    with open(file, mode="r") as f:
        return " ".join(f.read().splitlines())


def read_file_to_dict(file: Union[str, pathlib.Path]) -> dict:
    with open(file, mode="r") as f:
        return json.load(f)


def read_file_to_dataframe(file: Union[str, pathlib.Path]) -> pd.DataFrame:

    with open(file, mode="r") as f:
        header = f.readline().split(",")
        first_row = f.readline().split(",")

    dtype = dict()
    for variable, value in zip(header, first_row):
        if value.startswith('"') and value.endswith('"'):
            dtype[variable] = object

    df = pd.read_csv(file, dtype=dtype)

    if "id" in df.columns:
        df = df.set_index("id")

    return df


def set_age_intervals(age_start: Union[int, List[int]], age_interval_length: Union[int, List[int]],
                      profile: pd.DataFrame, profile_name: str) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(age_start, int):
        age_start = [age_start] * len(profile)

    if isinstance(age_interval_length, int):
        age_interval_length = [age_interval_length] * len(profile)

    if len(age_start) != len(profile) or len(age_interval_length) != len(profile):
        raise ValueError(f"ERROR: the number of values in 'apply_age_start' and 'apply_age_interval_length', "
                         f"and the number of rows in '{profile_name}' must match.")

    age_start, age_interval_length = np.array(age_start).astype(float), np.array(age_interval_length).astype(float)
    check_errors.check_age_intervals(age_start, age_interval_length)

    return age_start, age_interval_length


def configure_snp_and_covariates_model(
        model_snp_info: Union[str, pathlib.Path, None],
        apply_snp_profile: Union[str, pathlib.Path, None],
        model_family_history_variable_name: Optional[str],
        config: dict) -> None:
    config["snp_model"] = dict()

    if apply_snp_profile is not None:
        config["snp_model"]["profile"] = read_file_to_dataframe(apply_snp_profile)
        check_errors.check_snp_profile(config["snp_model"]["profile"], config["covariate_model"]["profile"],
                                       config["snp_model"]["snp_names"])
    else:
        num_instances_imputed = config["covariate_model"]["profile"].shape[0]
        config["snp_model"]["profile"] = pd.DataFrame(
            data=np.full((num_instances_imputed, config["snp_model"]["snp_names"].shape[0]), np.nan)
        )
        print(f"\nNote: You included 'model_snp_info' but did not provide an 'apply_snp_profile'. "
              f"iCARE will impute SNPs for {num_instances_imputed} individuals, the same number of"
              f" instances as the supplied 'apply_covariate_profile'.\n")

    if model_family_history_variable_name is not None:
        check_errors.check_family_history(
            model_family_history_variable_name, config["covariate_model"]["reference_dataset"],
            config["covariate_model"]["profile"])
        config["snp_model"]["family_history"] = dict()
        config["snp_model"]["family_history"]["population"] = \
            config["covariate_model"]["reference_dataset"][model_family_history_variable_name].values
        config["snp_model"]["family_history"]["profile"] = \
            config["covariate_model"]["profile"][model_family_history_variable_name].values
        config["snp_model"]["family_history"]["attenuate"] = True
    else:
        config["snp_model"]["family_history"] = dict()
        config["snp_model"]["family_history"]["population"] = np.repeat(
            0, len(config["covariate_model"]["reference_dataset"]))
        config["snp_model"]["family_history"]["profile"] = np.repeat(
            0, len(config["covariate_model"]["profile"]))
        config["snp_model"]["family_history"]["attenuate"] = False
        print("\nNote: You did not provide a 'model_family_history_variable_name', therefore "
              "the model will not adjust the SNP imputations for family history.\n")

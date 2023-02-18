import json
import pathlib
from typing import Union, List, Tuple, Type

import numpy as np
import pandas as pd

from icare import check_errors


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
        if (value.startswith('"') or value.startswith("'")) and \
                (value.endswith('"') or value.endswith("'")):
            dtype[variable] = object

    df = pd.read_csv(file, dtype=dtype)

    if "id" in df.columns:
        df.set_index("id", inplace=True)

    return df


def read_file_to_dataframe_given_dtype(
        file: Union[str, pathlib.Path],
        dtype: Union[dict, Type[float], Type[int], Type[str], Type[object]]) -> pd.DataFrame:
    header = pd.read_csv(file, nrows=1).columns
    if "id" in header:
        if isinstance(dtype, dict) and "id" not in dtype:
            dtype = {"id": str, **dtype}
        else:
            dtype = {"id": str, **{col: dtype for col in header if col != "id"}}

    df = pd.read_csv(file, dtype=dtype)

    if "id" in df.columns:
        df.set_index("id", inplace=True)

    return df


def set_age_intervals(age_start: Union[int, List[int]], age_interval_length: Union[int, List[int]],
                      num_samples_profile: int, profile_name: str) -> Tuple[List[int], List[int]]:
    if isinstance(age_start, int):
        age_start = [age_start] * num_samples_profile

    if isinstance(age_interval_length, int):
        age_interval_length = [age_interval_length] * num_samples_profile

    if len(age_start) != num_samples_profile or len(age_interval_length) != num_samples_profile:
        raise ValueError(f"ERROR: the number of values in 'apply_age_start' and 'apply_age_interval_length', "
                         f"and the number of rows in '{profile_name}' must match.")

    check_errors.check_age_intervals(age_start, age_interval_length)

    return age_start, age_interval_length

import pathlib
import time
from typing import Union, List, Optional

import numpy as np

from icare import check_errors, misc, utils
from icare.absolute_risk_model import AbsoluteRiskModel


def hello_world(name="world"):
    return f"Hello, {name}!"


def compute_absolute_risk(
        apply_age_start: Union[int, List[int]],
        apply_age_interval_length: Union[int, List[int]],
        model_disease_incidence_rates: Union[str, pathlib.Path],
        model_covariate_formula: Union[str, pathlib.Path, None] = None,
        model_snp_info: Union[str, pathlib.Path, None] = None,
        model_log_relative_risk: Union[str, pathlib.Path, None] = None,
        model_reference_dataset: Union[str, pathlib.Path, None] = None,
        model_reference_dataset_weights_variable_name: Optional[str] = None,
        model_competing_incidence_rates: Union[str, pathlib.Path, None] = None,
        model_family_history_variable_name: Optional[str] = None,
        num_imputations: int = 5,
        apply_covariate_profile: Union[str, pathlib.Path, None] = None,
        apply_snp_profile: Union[str, pathlib.Path, None] = None,
        use_c_code: bool = False,
        return_linear_predictors: bool = False,
        return_reference_risks: bool = False):
    """
    This function is used to build absolute risk models and apply them to estimate absolute risks.

    :param apply_age_start: ages for the start of the interval over which to compute the absolute risk. If a single int
        is provided, all instances in the 'apply_covariates_profile' and apply_snp_profile are assigned the same start age
        for the interval. If a different start age needs to be assigned for each instance in apply_covariates_profile
        and apply_snp_profile, provide a list of ints of the same length as the number of rows in these profiles.
    :param apply_age_interval_length:
    :param model_disease_incidence_rates:
    :param model_covariate_formula: a symbolic description (an R formula class object) of the model to be fitted,
        e.g. Y ~ parity + family_history.
        Please make sure that the variable name in your dataset is not from the namespace of the Python execution
        evironment, including Python standard library, numpy, pandas, patsy, and icare.
        For example, a variable name "C" and "Q" would conflict with Patsy built-in functions of the same name.
        Variable names with the R-style periods in them should be surrounded by the quote function Q(family.history).
        In Python periods are used to access attributes of objects, so they are not allowed in variable names unless
        surrounded by Q().
    :param model_snp_info: a dataframe with three columns, named: ["snp_name", "snp_odds_ratio", "snp_freq"].
    :param model_log_relative_risk: a list
    :param model_reference_dataset:
    :param model_reference_dataset_weights_variable_name:
    :param model_competing_incidence_rates:
    :param model_family_history_variable_name:
    :param num_imputations: the number of imputations (int) for handling missing SNPs.
    :param apply_covariate_profile: a dataframe containing covariate profiles for which,
    :param apply_snp_profile: a data frame with observed SNP data of allele dosages (coded 0, 1, 2, or “”). Missing
        values are allowed.
    :param return_linear_predictors: set True to return the linear predictor for each subject in apply_covariates_profile.
    :param return_reference_risks: set True to return the absolute risk prediction for each subject in model_reference_dataset.
    :return: This function returns a list of results objects, including—
        1) risk,
        2) details,
        3) beta_used,
        4) lps
        5) refs_risk
    """

    absolute_risk_model = AbsoluteRiskModel(
        apply_age_start, apply_age_interval_length, model_disease_incidence_rates, model_covariate_formula,
        model_snp_info, model_log_relative_risk, model_reference_dataset, model_reference_dataset_weights_variable_name,
        model_competing_incidence_rates, model_family_history_variable_name, num_imputations,
        apply_covariate_profile, apply_snp_profile
    )

    if handle_snps:
        attenuate_fh, fh_pop, apply_snp_profile = utils.process_snp_info(
            model_includes_covariates, apply_snp_profile,
            model_family_history_variable_name, apply_covariate_profile,
            model_reference_dataset, model_snp_info
        )

    if model_includes_covariates:
        if handle_snps:
            covariate_stack5 = None
    else:
        apply_age_start, apply_age_interval_length = check_errors.check_age_lengths(
            apply_age_start, apply_age_interval_length,
            apply_snp_profile, "apply_snp_profile"
        )

        if handle_snps:
            pop_dist_mat = utils.sim_snps(
                model_snp_info["snp_betas"].values,
                model_snp_info["snp_freq"].values,
                np.tile(fh_pop, num_imputations)
            )
            pop_weights = np.full((pop_dist_mat.shape[0],), 1.0 / pop_dist_mat.shape[0])
            beta_est = model_snp_info["snp_betas"].values
            z_new = apply_snp_profile.values.T

    lambda_vals, model_competing_incidence_rates = check_errors.check_rates(
        model_competing_incidence_rates, model_disease_incidence_rates,
        apply_age_start, apply_age_interval_length
    )
    approx_expectation_rr = np.average(np.exp(np.matmul(pop_dist_mat, beta_est)), weights=pop_weights)
    lambda_0, precise_expectation_rr = utils.precise_lambda0(
        lambda_vals, approx_expectation_rr, beta_est, pop_dist_mat, pop_weights
    )
    pop_dist_mat = pop_dist_mat.T

    # compute A_j for non-NAs
    final_risks = np.full((z_new.shape[1]), np.nan)
    lps = np.matmul(z_new.T, beta_est)
    idxs_not_nan = np.where(np.sum(np.array(np.isnan(z_new), dtype=int), axis=0) == 0)

    if idxs_not_nan[0].shape[0] > 0:
        final_risks[idxs_not_nan] = utils.comp_a_j(
            z_new[idxs_not_nan], apply_age_start[idxs_not_nan],
            apply_age_interval_length[idxs_not_nan], lambda_0,
            beta_est, model_competing_incidence_rates
        )

    miss = np.where(np.isnan(final_risks))[0]
    present = np.where(~np.isnan(final_risks))
    ref_pop = pop_dist_mat
    n_cuts = 100

    tic = time.time()
    final_risks, lps = utils.handle_missing_data(
        apply_age_start, apply_age_interval_length, z_new, miss, present, n_cuts, final_risks,
        ref_pop, pop_weights, lambda_0, beta_est, model_competing_incidence_rates, lps
    )

    these = np.where(np.sum(np.array(~np.isnan(z_new), dtype=int), axis=0) == 0)[0]
    if these.shape[0] > 0:
        ref_risks, _ = utils.get_refs_risk(
            ref_pop, apply_age_start, apply_age_interval_length, lambda_0, beta_est, model_competing_incidence_rates,
            handle_snps, num_imputations
        )
        final_risks[these] = np.average(
            ref_risks[~np.isnan(ref_risks)],
            weights=pop_weights[:ref_risks.shape[0]][~np.isnan(ref_risks)]
        )

    toc = time.time()
    print(f"Time elapsed: {toc - tic:.5} seconds.")

    result = misc.package_results(
        final_risks, z_new, model_includes_covariates, handle_snps, apply_age_start, apply_age_interval_length,
        apply_covariate_profile, model_log_relative_risk, beta_est, apply_snp_profile,
        model_snp_info["snp_name"].values, return_linear_predictors, lps
    )

    if return_reference_risks:
        result["reference_risk"], _ = utils.get_refs_risk(
            ref_pop, apply_age_start, apply_age_interval_length, lambda_0, beta_est, model_competing_incidence_rates,
            handle_snps, num_imputations
        )

    return result


def compute_absolute_risk_split_interval(
        apply_age_start,
        apply_age_interval_length,
        apply_cov_profile,
        model_formula,
        model_disease_incidence_rates,
        model_log_rr,
        model_ref_dataset,
        model_cov_info,
        model_ref_dataset_weights=None,
        model_competing_incidence_rates=None,
        return_lp=False,
        apply_snp_profile=None,
        model_snp_info=None,
        model_bin_fh_name=None,
        cut_time=None,
        apply_cov_profile_2=None,
        model_formula_2=None,
        model_log_rr_2=None,
        model_ref_dataset_2=None,
        model_ref_dataset_weights_2=None,
        model_cov_info_2=None,
        model_bin_fh_name_2=None,
        num_imputations=5,
        return_refs_risk=False):
    """
    This function is used to build an absolute risk model that incorporates different input parameters before and after
        a given time point. The model is then applied to estimate absolute risks.

    :param apply_age_start:
    :param apply_age_interval_length:
    :param apply_cov_profile:
    :param model_formula:
    :param model_disease_incidence_rates:
    :param model_log_rr:
    :param model_ref_dataset:
    :param model_cov_info:
    :param model_ref_dataset_weights:
    :param model_competing_incidence_rates:
    :param return_lp:
    :param apply_snp_profile:
    :param model_snp_info:
    :param model_bin_fh_name:
    :param cut_time:
    :param apply_cov_profile_2:
    :param model_formula_2:
    :param model_log_rr_2:
    :param model_ref_dataset_2:
    :param model_ref_dataset_weights_2:
    :param model_cov_info_2:
    :param model_bin_fh_name_2:
    :param num_imputations:
    :param return_refs_risk:
    """
    pass

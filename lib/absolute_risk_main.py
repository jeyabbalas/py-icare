import numpy as np

from lib import check_errors, utils


def compute_absolute_risk(
        apply_age_start,
        apply_age_interval_length, 
        model_disease_incidence_rates,
        model_formula=None,
        model_covariates_info=None,
        model_snp_info=None,
        model_log_relative_risk=None,
        model_reference_dataset=None,
        model_ref_dataset_weights=None,
        model_competing_incidence_rates=None,
        model_family_history_binary_variable_name=None,
        n_imp=5,
        apply_covariates_profile=None,
        apply_snp_profile=None,
        return_lp=False,
        return_refs_risk=False
):
    """
    This function is used to build absolute risk models and apply them to estimate absolute risks.

    :param apply_age_start: ages for the start of the interval over which to compute the absolute risk. If a single int
        is provided, all instances in the 'apply_covariates_profile' and apply_snp_profile are assigned the same start age
        for the interval. If a different start age needs to be assigned for each instance in apply_covariates_profile
        and apply_snp_profile, provide a list of ints of the same length as the number of rows in these profiles.
    :param apply_age_interval_length:
    :param model_disease_incidence_rates:
    :param model_formula: a symbolic description (an R formula class object) of the model to be fitted,
        e.g. Y~Parity+FamilyHistory.
    :param model_covariates_info: contains information (dict) about the risk factors in the model.
        The keys of the dict are the covariate names as strings. Each have a dict value associated with them
    :param model_snp_info: a dataframe with three columns, named: ["snp_name", "snp_odds_ratio", "snp_freq"].
    :param model_log_relative_risk: a list
    :param model_reference_dataset:
    :param model_ref_dataset_weights:
    :param model_competing_incidence_rates:
    :param model_family_history_binary_variable_name:
    :param n_imp: the number of imputations (int) for handling missing SNPs.
    :param apply_covariates_profile: a dataframe containing covairate profiles for which,
    :param apply_snp_profile: a data frame with observed SNP data of allele dosages (coded 0, 1, 2, or “”). Missing
        values are allowed.
    :param return_lp: set True to return the linear predictor for each subject in apply_covariates_profile.
    :param return_refs_risk: set True to return the absolute risk prediction for each subject in model_ref_dataset.
    :return: This function returns a list of results objects, including—
        1) risk,
        2) details,
        3) beta_used,
        4) lps
        5) refs_risk
    """
    model_includes_covariates, apply_snp_profile = utils.decide_if_snp_only(
        apply_covariates_profile, model_formula, model_log_relative_risk,
        model_reference_dataset, model_covariates_info, model_snp_info,
        apply_snp_profile, apply_age_start, apply_age_interval_length
    )

    handle_snps = model_snp_info is not None
    fh_pop = None

    if model_includes_covariates:
        apply_age_start, apply_age_interval_length = check_errors.check_age_lengths(
            apply_age_start, apply_age_interval_length, apply_covariates_profile, "apply_covariates_profile"
        )

    if handle_snps:
        check_errors.check_snp_info(model_snp_info)
        model_snp_info["snp_betas"] = np.log(model_snp_info["snp_odds_ratio"])
        attenuate_fh, fh_pop, apply_snp_profile = utils.process_snp_info(
            model_includes_covariates, apply_snp_profile,
            model_family_history_binary_variable_name, apply_covariates_profile,
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
            # , pop_weights, beta_est, z_new
            # pop.dist.mat <- sim_snps(snps.betas, snps.freqs, cbind( rep( fh.pop, n.imp)) )
            pop_dist_mat = utils.sim_snps(
                model_snp_info["snp_betas"].values, model_snp_info["snp_freq"].values, fh_pop
            )


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
        model_competing_incidence_rates = None,
        return_lp = False,
        apply_snp_profile = None,
        model_snp_info = None,
        model_bin_fh_name = None,
        cut_time = None,
        apply_cov_profile_2 = None,
        model_formula_2 = None,
        model_log_rr_2 = None,
        model_ref_dataset_2 = None,
        model_ref_dataset_weights_2 = None,
        model_cov_info_2 = None,
        model_bin_fh_name_2 = None,
        n_imp = 5,
        return_refs_risk=False
):
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
    :param n_imp:
    :param return_refs_risk:
    """
    pass

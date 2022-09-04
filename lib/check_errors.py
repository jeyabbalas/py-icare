import numpy as np
import pandas as pd


def check_age_lengths(apply_age_start, apply_age_interval_length, match, match_name):
    if isinstance(apply_age_start, int):
        apply_age_start = np.full((match.shape[0],), apply_age_start)

    if isinstance(apply_age_interval_length, int):
        apply_age_interval_length = np.full((match.shape[0],), apply_age_interval_length)

    if apply_age_start.shape[0] != apply_age_interval_length.shape[0]:
        raise ValueError("ERROR: 'apply_age_start and 'apply_age_interval_length' must have the same length.")

    if apply_age_start.shape[0] != match.shape[0]:
        raise ValueError(f"ERROR: 'apply_age_start' and number of rows in '{match_name}' must match.")

    if (sum(np.isnan(apply_age_start)) + sum(np.isnan(apply_age_interval_length))) > 0:
        raise ValueError("ERROR: 'apply_age_start' and 'apply_age_interval_length' must not contain missing values.")

    if ((sum(apply_age_start < 0)) + sum(apply_age_interval_length < 0)) > 0:
        raise ValueError("ERROR: 'apply_age_start' and 'apply_age_interval_length' must contain positive values.")

    return apply_age_start, apply_age_interval_length


def check_snp_info(model_snp_info):
    if not isinstance(model_snp_info, pd.DataFrame):
        raise ValueError("ERROR: If specified, the argument 'model_snp_info' requires a Pandas dataframe (that "
                         "contains the information on SNP names, odds ratios, and allele frequencies.")

    if "snp_name" not in model_snp_info.columns or \
            "snp_odds_ratio" not in model_snp_info.columns or \
            "snp_freq" not in model_snp_info.columns:
        raise ValueError("ERROR: If specified, the argument 'model_snp_info' must be a Pandas dataframe with at least "
                         "3 columns named: 'snp_name', 'snp_odds_ratio', and 'snp_freq'.")

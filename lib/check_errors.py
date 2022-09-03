import pandas as pd


def check_age_lengths(apply_age_start, apply_age_interval_length, match, match_name):
    # TODO
    pass


def check_snp_info(model_snp_info):
    if not isinstance(model_snp_info, pd.DataFrame):
        raise ValueError("ERROR: If specified, the argument `model_snp_info` requires a Pandas dataframe (that "
                         "contains the information on SNP names, odds ratios, and allele frequencies.")

    if "snp_name" not in model_snp_info.columns or \
            "snp_odds_ratio" not in model_snp_info.columns or \
            "snp_freq" not in model_snp_info.columns:
        raise ValueError("ERROR: If specified, the argument `model_snp_info` must be a Pandas dataframe with at least "
                         "3 columns named: 'snp_name', 'snp_odds_ratio', and 'snp_freq'.")

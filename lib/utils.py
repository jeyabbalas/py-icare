import pandas as pd


def decide_if_snp_only(
        apply_covariates_profile,
        model_formula,
        model_log_relative_risk,
        model_reference_dataset,
        model_covariates_info,
        model_snp_info,
        apply_snp_profile,
        apply_age_start,
        apply_age_interval_length
):
    if apply_covariates_profile is None and \
            model_formula is None and \
            model_log_relative_risk is None and \
            model_reference_dataset is None and \
            model_covariates_info is None:
        # SNP-only model, no covariates in model
        model_includes_covariates = False

        if model_snp_info is None:
            raise ValueError("ERROR: You appear to be fitting a SNP-only model, and thus must provide relevant data "
                             "to the `model_snp_info` argument.")

        if apply_snp_profile is None:
            if isinstance(apply_age_start, int) and isinstance(apply_age_interval_length, int):
                apply_snp_profile = pd.DataFrame(columns=model_snp_info["snp_info"],
                                                 index=range(10_000))
                print("\nNote: You did not provide an `apply_snp_profile`.  "
                      "iCARE will impute SNPs for 10000 individuals.\n")
                print("If you require more, please provide an input to `apply_snp_profile`.\n")
            else:
                apply_snp_profile = pd.DataFrame(columns=model_snp_info["snp_info"],
                                                 index=range(len(apply_age_start)))
                print(f"\nNote: You did not provide an `apply_snp_profile`.  "
                      f"iCARE will impute SNPs for {len(apply_age_start)} individuals, "
                      f"to match the specified number of age intervals.\n")
    else:
        # Model includes covariates
        if apply_covariates_profile is None or \
                model_formula is None or \
                model_log_relative_risk is None or \
                model_reference_dataset is None or \
                model_covariates_info is None:
            raise ValueError("ERROR: Either all or none of the arguments— `apply_covariates_profile`, `model_formula`, "
                             "`model_log_relative_risk`, `model_reference_dataset`, and `model_covariates_info`— "
                             "should be None. If all of them are None, it implies a SNP-only model.")

        model_includes_covariates = True

    return model_includes_covariates, apply_snp_profile

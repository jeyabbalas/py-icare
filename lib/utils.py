import numpy as np
import pandas as pd


def decide_if_snp_only(apply_covariates_profile, model_formula, model_log_relative_risk, model_reference_dataset,
                       model_covariates_info, model_snp_info, apply_snp_profile, apply_age_start,
                       apply_age_interval_length):
    if apply_covariates_profile is None and \
            model_formula is None and \
            model_log_relative_risk is None and \
            model_reference_dataset is None and \
            model_covariates_info is None:
        # SNP-only model, no covariates in model
        model_includes_covariates = False

        if model_snp_info is None:
            raise ValueError("ERROR: You appear to be fitting a SNP-only model, and thus must provide relevant data "
                             "to the 'model_snp_info' argument.")

        if apply_snp_profile is None:
            if isinstance(apply_age_start, int) and isinstance(apply_age_interval_length, int):
                apply_snp_profile = pd.DataFrame(columns=model_snp_info["snp_name"],
                                                 index=range(10_000))
                print("\nNote: You did not provide an 'apply_snp_profile'.  "
                      "iCARE will impute SNPs for 10000 individuals.\n")
                print("If you require more, please provide an input to 'apply_snp_profile'.\n")
            else:
                apply_snp_profile = pd.DataFrame(columns=model_snp_info["snp_info"],
                                                 index=range(len(apply_age_start)))
                print(f"\nNote: You did not provide an 'apply_snp_profile'.  "
                      f"iCARE will impute SNPs for {len(apply_age_start)} individuals, "
                      f"to match the specified number of age intervals.\n")
    else:
        # Model includes covariates
        if apply_covariates_profile is None or \
                model_formula is None or \
                model_log_relative_risk is None or \
                model_reference_dataset is None or \
                model_covariates_info is None:
            raise ValueError("ERROR: Either all or none of the arguments— 'apply_covariates_profile', 'model_formula', "
                             "'model_log_relative_risk', 'model_reference_dataset', and 'model_covariates_info'— "
                             "should be None. If all of them are None, it implies a SNP-only model.")

        model_includes_covariates = True

    return model_includes_covariates, apply_snp_profile


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
    return None
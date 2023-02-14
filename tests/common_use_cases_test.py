from icare import absolute_risk_main


def snp_risk_test():
    model_disease_incidence_rates = "data/age_specific_breast_cancer_incidence_rates.csv"
    model_competing_incidence_rates = "data/age_specific_all_cause_mortality_rates.csv"

    model_snp_info = "data/breast_cancer_72_snps_info.csv"
    apply_snp_profile = "data/query_snp_profile.csv"

    absolute_risk_main.compute_absolute_risk(
        model_snp_info_path=model_snp_info,
        model_disease_incidence_rates=model_disease_incidence_rates,
        model_competing_incidence_rates_path=model_competing_incidence_rates,
        apply_age_start=50,
        apply_age_interval_length=30,
        apply_snp_profile_path=apply_snp_profile,
        return_reference_risks=True
    )


def snp_and_covariate_risk_test():
    model_disease_incidence_rates = "data/age_specific_breast_cancer_incidence_rates.csv"
    model_competing_incidence_rates = "data/age_specific_all_cause_mortality_rates.csv"

    model_snp_info = "data/breast_cancer_72_snps_info.csv"
    apply_snp_profile = "data/query_snp_profile.csv"

    model_covariate_formula = "data/breast_cancer_covariate_model_formula.txt"
    model_log_relative_risk = "data/breast_cancer_log_relative_risk.json"
    model_reference_dataset = "data/reference_covariate_data.csv"
    model_family_history_variable_name = "family_history"
    apply_covariate_profile = "data/query_covariate_profile.csv"

    absolute_risk_main.compute_absolute_risk(
        model_snp_info_path=model_snp_info,
        model_covariate_formula_path=model_covariate_formula,
        model_log_relative_risk_path=model_log_relative_risk,
        model_reference_dataset_path=model_reference_dataset,
        model_family_history_variable_name=model_family_history_variable_name,
        model_disease_incidence_rates=model_disease_incidence_rates,
        model_competing_incidence_rates_path=model_competing_incidence_rates,
        apply_age_start=50,
        apply_age_interval_length=30,
        apply_snp_profile_path=apply_snp_profile,
        apply_covariate_profile_path=apply_covariate_profile,
        return_reference_risks=True
    )

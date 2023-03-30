import icare


def test_snp_risk_with_profile():
    model_disease_incidence_rates_path = "../data/age_specific_breast_cancer_incidence_rates.csv"
    model_competing_incidence_rates_path = "../data/age_specific_all_cause_mortality_rates.csv"

    model_snp_info_path = "../data/breast_cancer_72_snps_info.csv"
    apply_snp_profile_path = "../data/query_snp_profile.csv"

    icare.compute_absolute_risk(
        model_snp_info_path=model_snp_info_path,
        model_disease_incidence_rates_path=model_disease_incidence_rates_path,
        model_competing_incidence_rates_path=model_competing_incidence_rates_path,
        apply_age_start=50,
        apply_age_interval_length=30,
        apply_snp_profile_path=apply_snp_profile_path,
        return_reference_risks=True
    )


def test_snp_risk_without_profile():
    model_disease_incidence_rates_path = "../data/age_specific_breast_cancer_incidence_rates.csv"
    model_competing_incidence_rates_path = "../data/age_specific_all_cause_mortality_rates.csv"

    model_snp_info_path = "../data/breast_cancer_72_snps_info.csv"

    icare.compute_absolute_risk(
        model_snp_info_path=model_snp_info_path,
        model_disease_incidence_rates_path=model_disease_incidence_rates_path,
        model_competing_incidence_rates_path=model_competing_incidence_rates_path,
        apply_age_start=50,
        apply_age_interval_length=30,
        return_reference_risks=True
    )


def test_snp_and_covariate_risk():
    model_disease_incidence_rates_path = "../data/age_specific_breast_cancer_incidence_rates.csv"
    model_competing_incidence_rates_path = "../data/age_specific_all_cause_mortality_rates.csv"

    model_snp_info_path = "../data/breast_cancer_72_snps_info.csv"
    apply_snp_profile_path = "../data/query_snp_profile.csv"

    model_covariate_formula_path = "../data/breast_cancer_covariate_model_formula.txt"
    model_log_relative_risk_path = "../data/breast_cancer_log_relative_risk.json"
    model_reference_dataset_path = "../data/reference_covariate_data.csv"
    model_family_history_variable_name = "family_history"
    apply_covariate_profile_path = "../data/query_covariate_profile.csv"

    icare.compute_absolute_risk(
        model_snp_info_path=model_snp_info_path,
        model_covariate_formula_path=model_covariate_formula_path,
        model_log_relative_risk_path=model_log_relative_risk_path,
        model_reference_dataset_path=model_reference_dataset_path,
        model_family_history_variable_name=model_family_history_variable_name,
        model_disease_incidence_rates_path=model_disease_incidence_rates_path,
        model_competing_incidence_rates_path=model_competing_incidence_rates_path,
        apply_age_start=50,
        apply_age_interval_length=30,
        apply_snp_profile_path=apply_snp_profile_path,
        apply_covariate_profile_path=apply_covariate_profile_path,
        return_reference_risks=True
    )

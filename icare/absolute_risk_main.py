import pathlib
from typing import Union, List, Optional

from icare import misc
from icare.absolute_risk_model import AbsoluteRiskModel


def hello_world(name="world"):
    return f"Hello, {name}!"


def compute_absolute_risk(apply_age_start: Union[int, List[int]],
                          apply_age_interval_length: Union[int, List[int]],
                          model_disease_incidence_rates_path: Union[str, pathlib.Path],
                          model_competing_incidence_rates_path: Union[str, pathlib.Path, None] = None,
                          model_covariate_formula_path: Union[str, pathlib.Path, None] = None,
                          model_log_relative_risk_path: Union[str, pathlib.Path, None] = None,
                          model_reference_dataset_path: Union[str, pathlib.Path, None] = None,
                          model_reference_dataset_weights_variable_name: Optional[str] = None,
                          model_snp_info_path: Union[str, pathlib.Path, None] = None,
                          model_family_history_variable_name: Optional[str] = None,
                          num_imputations: int = 5,
                          apply_covariate_profile_path: Union[str, pathlib.Path, None] = None,
                          apply_snp_profile_path: Union[str, pathlib.Path, None] = None,
                          return_linear_predictors: bool = False,
                          return_reference_risks: bool = False) -> dict:
    """
    This function is used to build absolute risk models and apply them to estimate absolute risks.

    :param apply_age_start: Age(s) for the start of the interval, over which, to compute the absolute risk. If a single
        integer is provided, all instances in the profiles ('apply_covariate_profile_path' and/or
        'apply_snp_profile_path') are assigned this start age for the interval. If a different start age needs to be
        assigned for each instance, provide a list of ages as integers of the same length as the number of instances in
        these profiles.
    :param apply_age_interval_length: Number of years over which to compute the absolute risk. That is to say that the
        age at the end of the interval is 'apply_age_start' + 'apply_age_interval_length'. If a single integer is
        provided, all instances in the profiles ('apply_covariate_profile_path' and/or 'apply_snp_profile_path') are
        assigned this interval length. If a different interval length needs to be assigned for each instance, provide a
        list of interval lengths as integers of the same length as the number of instances in these profiles.
    :param model_disease_incidence_rates_path:
        A path to a CSV file containing the age-specific disease incidence rates for the population of interest. The
        data in the file must either contain two columns, named: ['age', 'rate'], to specify the incidence rates
        associated with each age group; or three columns, named: ['start_age', 'end_age', 'rate'], to specify the
        incidence rates associated with each age interval. The age ranges must fully cover the age intervals specified
        using parameters 'apply_age_start' and 'apply_age_interval_length'.
    :param model_competing_incidence_rates_path:
        A path to a CSV file containing the age-specific incidence rates for competing events in the population of
        interest. The data in the file must either contain two columns, named: ['age', 'rate'], to specify the
        incidence rates associated with each age group; or three columns, named: ['start_age', 'end_age', 'rate'], to
        specify the incidence rates associated with each age interval. The age ranges must fully cover the age
        intervals specified using parameters 'apply_age_start' and 'apply_age_interval_length'.
    :param model_covariate_formula_path:
        A path to a text file containing a Patsy symbolic description string of the model to be fitted,
        e.g. Y ~ parity + family_history.
        Reference: https://patsy.readthedocs.io/en/latest/formulas.html#the-formula-language
        Please make sure that the variable name in your dataset is not from the namespace of the Python execution
        context, including Python standard library, numpy, pandas, patsy, and icare. For example, a variable name "C"
        and "Q" would conflict with Patsy built-in functions of the same name. Variable names with the R-style periods
        in them should be surrounded by the quote function Q(family.history). In Python, periods are used to access
        attributes of objects, so they are not allowed in Patsy variable names unless surrounded by Q().
        Patsy language is similar to R's formula object (https://patsy.readthedocs.io/en/latest/R-comparison.html).
    :param model_log_relative_risk_path:
        A path to a JSON file containing the log odds ratios, of the variables in the model except the intercept term,
        in association with the disease. The first-level JSON keys should correspond to the variable names generated by
        Patsy when building the design matrix. Their values should correspond to the log odds ratios of the variable's
        association with the disease.
    :param model_reference_dataset_path:
        A path to a CSV file containing the reference dataset with risk factor distribution that is representative of
        the population of interest. No missing values are permitted in this dataset.
    :param model_reference_dataset_weights_variable_name:
        A string specifying the name of the variable in the dataset at 'model_reference_dataset_path' that indicates
        the sampling weight for each instance. If set to None (default), then a uniform weight will be assigned to each
        instance.
    :param model_snp_info_path:
        A path to a CSV file containing the information about the SNPs in the model. The data should contain three
        columns, named: ['snp_name', 'snp_odds_ratio', 'snp_freq'] corresponding to the SNP ID, the odds ratio of the
        SNP in association with the disease, and the minor allele frequency, respectively.
    :param model_family_history_variable_name:
        A string specifying the name of the binary variable (values: {0, 1}; missing values are permitted) in the
        model formula ('model_covariate_formula_path') that represents the family history of the disease. This needs to
        be specified when using the special SNP model option so that the effect of family history can be adjusted for
        the presence of the SNPs.
    :param num_imputations:
        The number of imputations for handling missing SNPs.
    :param apply_covariate_profile_path:
        A path to a CSV file containing the covariate (risk factor) profiles of the individuals for whom the absolute
        risk is to be computed. Missing values are permitted.
    :param apply_snp_profile_path:
        A path to a CSV file containing the SNP profiles (values: {0: homozygous reference alleles, 1: heterozygous,
        2: homozygous alternate alleles}) of the individuals for whom the absolute risk is to be computed. Missing
        values are permitted.
    :param return_linear_predictors:
        Set True to return the calculated linear predictor values for each individual in the
        'apply_covariate_profile_path' and/or 'apply_snp_profile_path' datasets.
    :param return_reference_risks:
        Set True to return the absolute risk estimates for each individual in the 'model_reference_dataset_path'
        dataset.
    :return:
        This function returns a dictionary, with the following keys—
            1) 'beta_used':
                A dictionary of feature names and the associated beta values that were used to compute the absolute risk
                estimates.
            2) 'profile':
                A records-oriented JSON of the input profile data, the specified age intervals, and the calculated
                absolute risk estimates. If 'return_linear_predictors' is set to True, they are also included as an
                additional column.
                A Pandas DataFrame can be reconstructed using the following code:
                    import pandas as pd
                    results = compute_absolute_risk(...)
                    pd.read_json(results["profile"], orient="records")
            3) 'reference_risks':
                If 'return_reference_risks' is True, this key will be present in the returned dictionary. It will
                contain a list of dictionaries, one per unique combination of the specified age intervals, containing
                age at the start of interval ('age_interval_start'), age at the end of interval ('age_interval_end'),
                and a list absolute risk estimates for the individuals in the reference dataset ('population_risks').
    """

    absolute_risk_model = AbsoluteRiskModel(
        apply_age_start, apply_age_interval_length, model_disease_incidence_rates_path, model_covariate_formula_path,
        model_snp_info_path, model_log_relative_risk_path, model_reference_dataset_path,
        model_reference_dataset_weights_variable_name, model_competing_incidence_rates_path,
        model_family_history_variable_name, num_imputations, apply_covariate_profile_path, apply_snp_profile_path,
        return_reference_risks)

    absolute_risk_model.compute_absolute_risks()

    return misc.package_absolute_risk_results_to_dict(absolute_risk_model, return_linear_predictors,
                                                      return_reference_risks)


def compute_absolute_risk_split_interval(apply_age_start: Union[int, List[int]],
                                         apply_age_interval_length: Union[int, List[int]],
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

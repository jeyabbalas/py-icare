import pathlib
from typing import Union, List, Optional

import numpy as np
import pandas as pd

from icare import check_errors, utils


class CovariateModel:
    """A generic covariate model."""
    formula: str
    log_relative_risk: dict
    reference_dataset: pd.DataFrame
    profile: pd.DataFrame

    def __init__(
            self,
            formula: Union[str, pathlib.Path, None],
            log_relative_risk: Union[str, pathlib.Path, None],
            reference_dataset: Union[str, pathlib.Path, None],
            profile: Union[str, pathlib.Path, None]) -> None:
        parameters = [formula, log_relative_risk, reference_dataset, profile]
        any_parameter_missing = any([x is None for x in parameters])

        if any_parameter_missing:
            raise ValueError("ERROR: Either all or none of the following arguments— 'apply_covariate_profile', "
                             "'model_covariate_formula', 'model_log_relative_risk', and 'model_reference_dataset'"
                             "— should be specified. If none of them are specified, it implies a SNP-only model.")

        self.formula = utils.read_file_to_string(formula)
        self.log_relative_risk = utils.read_file_to_dict(log_relative_risk)
        self.reference_dataset = utils.read_file_to_dataframe(reference_dataset)
        self.profile = utils.read_file_to_dataframe(profile)


class SnpModel:
    """
    iCARE's special option for the SNP model. This option
    removes the need for specifying a reference dataset that the typical
    covariate model requires.
    """
    snp_names: np.ndarray
    betas: np.ndarray
    frequencies: np.ndarray
    profile: pd.DataFrame

    def __init__(
            self,
            info: Union[str, pathlib.Path, None],
            profile: Union[str, pathlib.Path, None],
            family_history_variable_name: Optional[str],
            age_start: Union[int, List[int]],
            age_interval_length: Union[int, List[int]],
            covariate_model: Optional[CovariateModel]) -> None:
        if covariate_model is None:
            self.instantiate_snp_only_model()
        else:
            self.instantiate_joint_snp_and_covariate_model()

    def instantiate_snp_only_model(
            self,
            info: Union[str, pathlib.Path, None],
            profile: Union[str, pathlib.Path, None],
            age_start: Union[int, List[int]],
            age_interval_length: Union[int, List[int]]) -> None:
        if info is None:
            raise ValueError(
                "ERROR: You appear to be fitting a SNP-only model, and thus you must provide relevant data "
                "to the 'model_snp_info' argument.")

        self.extract_snp_info(info)

        num_instances_imputed = 10_000
        if profile is not None:
            self.profile = utils.read_file_to_dataframe(profile)
        else:
            if isinstance(age_start, int) and isinstance(age_interval_length, int):
                self.profile = pd.DataFrame(data=np.full((num_instances_imputed, self.snp_names.shape[0]), np.nan))
                print(f"\nNote: You did not provide an 'apply_snp_profile'. "
                      f"iCARE will impute SNPs for {num_instances_imputed:,} individuals.")
                print("If you want more/less, please specify an input to 'apply_snp_profile'.\n")
            else:
                self.profile = pd.DataFrame(data=np.full((len(age_start), self.snp_names.shape[0]), np.nan))
                print(f"\nNote: You did not provide an 'apply_snp_profile'. "
                      f"iCARE will impute SNPs for {len(age_start)} individuals, "
                      f"to match the specified number of age intervals.\n")

        config["snp_model"]["family_history"] = dict()
        config["snp_model"]["family_history"]["population"] = np.repeat(0, num_instances_imputed)
        config["snp_model"]["family_history"]["profile"] = np.repeat(0, len(self.profile))
        config["snp_model"]["family_history"]["attenuate"] = False
        print("\nNote: You did not provide a 'model_family_history_variable_name', therefore "
              "the model will not adjust the SNP imputations for family history.\n")

    def instantiate_joint_snp_and_covariate_model(self):
        pass

    def extract_snp_info(self, snp_info_path: Union[str, pathlib.Path, None]) -> None:
        snp_info = utils.read_file_to_dataframe(snp_info_path)
        check_errors.check_snp_info(snp_info)
        self.snp_names = snp_info["snp_name"].values
        self.betas = np.log(snp_info["snp_odds_ratio"].values)
        self.frequencies = snp_info["snp_freq"].values


class AbsoluteRiskModel:
    """Something"""
    covariate_model: CovariateModel = None
    snp_model: SnpModel = None

    age_start: List[int]
    age_interval_length: List[int]
    beta: np.ndarray
    profile: pd.DataFrame
    population_distribution: pd.DataFrame
    population_weights: np.ndarray

    def __init__(
            self,
            apply_age_start: Union[int, List[int]],
            apply_age_interval_length: Union[int, List[int]],
            model_disease_incidence_rates: Union[str, pathlib.Path],
            model_covariate_formula: Union[str, pathlib.Path, None],
            model_snp_info: Union[str, pathlib.Path, None],
            model_log_relative_risk: Union[str, pathlib.Path, None],
            model_reference_dataset: Union[str, pathlib.Path, None],
            model_reference_dataset_weights: Optional[List[float]],
            model_competing_incidence_rates: Union[str, pathlib.Path, None],
            model_family_history_variable_name: str,
            num_imputations: int,
            apply_covariate_profile: Union[str, pathlib.Path, None],
            apply_snp_profile: Union[str, pathlib.Path, None]) -> None:
        covariate_parameters = [model_covariate_formula, model_log_relative_risk, model_reference_dataset,
                                apply_covariate_profile]
        any_covariate_parameter_specified = any([x is not None for x in covariate_parameters])

        if any_covariate_parameter_specified:
            self.covariate_model = CovariateModel(
                model_covariate_formula, model_log_relative_risk, model_reference_dataset, apply_covariate_profile
            )

        self.snp_model = SnpModel(
            model_snp_info, apply_snp_profile, model_family_history_variable_name, apply_age_start,
            apply_age_interval_length, self.covariate_model
        )


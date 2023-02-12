import pathlib
from typing import List, Union, Optional, Tuple

import numpy as np
import pandas as pd

from icare import utils, check_errors, design_matrix
from icare.covariate_model import CovariateModel


def extract_snp_info(info_path: Union[str, pathlib.Path, None]) -> Tuple[List[str], np.ndarray, np.ndarray]:
    info = utils.read_file_to_dataframe(info_path)
    check_errors.check_snp_info(info)
    snp_names = info["snp_name"].tolist()
    betas = np.log(info["snp_odds_ratio"].values)
    frequencies = info["snp_freq"].values
    return snp_names, betas, frequencies


def create_empty_snp_profile(num_samples: int, columns: List[str]) -> pd.DataFrame:
    return pd.DataFrame(data=np.full((num_samples, len(columns)), np.nan), columns=columns)


def set_snp_profile(profile_path: Union[str, pathlib.Path, None], age_start: Union[int, List[int]],
                    snp_names: List[str], covariate_model: Optional[CovariateModel],
                    num_samples_imputed: int) -> pd.DataFrame:
    if profile_path is not None:
        profile = utils.read_file_to_dataframe_given_dtype(profile_path, dtype=float)
        profile = profile[snp_names]
        check_errors.check_snp_profile(profile, snp_names)
        if covariate_model is not None:
            if len(profile) != len(covariate_model.z_profile):
                raise ValueError(f"ERROR: The number of individuals in the 'apply_snp_profile' ({len(profile)})"
                                 f" does not match the number of individuals in the 'apply_covariate_profile'"
                                 f"({len(covariate_model.z_profile)}).")
        return profile

    if covariate_model is None:
        if isinstance(age_start, int):
            num_samples = num_samples_imputed
            print(f"\nNote: You included 'model_snp_info' but did not provide an 'apply_snp_profile'. "
                  f"iCARE will impute SNPs for {num_samples} individuals. If you require more, "
                  f"please provide an input to 'apply_snp_profile' input.\n")
            return create_empty_snp_profile(num_samples, snp_names)
        else:
            num_samples = len(age_start)
            print(f"\nNote: You included 'model_snp_info' but did not provide an 'apply_snp_profile'. "
                  f"iCARE will impute SNPs for {num_samples} individuals, matching the number of"
                  f" age intervals specified.\n")
            return create_empty_snp_profile(num_samples, snp_names)
    else:
        num_samples = len(covariate_model.z_profile)
        print(f"\nNote: You included 'model_snp_info' but did not provide an 'apply_snp_profile'. "
              f"iCARE will impute SNPs for {num_samples} individuals, matching the number of"
              f" individuals in the specified 'apply_covariate_profile'.\n")
        return create_empty_snp_profile(num_samples, snp_names)


class FamilyHistory:
    """ Data structure to contain family history information. """
    population: np.ndarray
    profile: np.ndarray
    attenuate: bool

    def __init__(self, covariate_model: Optional[CovariateModel], family_history_variable_name: Optional[str],
                 profile: pd.DataFrame, snp_names: List[str], num_samples_imputed: int) -> None:
        if covariate_model is None:
            self.population = np.repeat(0, num_samples_imputed)
            self.profile = np.repeat(0, len(profile))
            self.attenuate = False

            print("\nNote: You did not provide a 'model_family_history_variable_name', therefore "
                  "the model will not adjust the SNP imputations for family history.\n")
        else:
            if family_history_variable_name is None:
                self.population = np.repeat(0, len(covariate_model.population_distribution))
                self.profile = np.repeat(0, len(profile))
                self.attenuate = False

                print("\nNote: You did not provide a 'model_family_history_variable_name', therefore "
                      "the model will not adjust the SNP imputations for family history.\n")
            else:
                check_errors.check_family_history_variable_name_type(family_history_variable_name)
                family_history_variable_name = design_matrix.get_design_matrix_column_name_matching_target_column_name(
                    covariate_model.population_distribution, family_history_variable_name
                )
                check_errors.check_family_history_variable(
                    family_history_variable_name, covariate_model.z_profile, covariate_model.population_distribution
                )

                self.population = covariate_model.population_distribution[family_history_variable_name].values
                self.profile = covariate_model.z_profile[family_history_variable_name].values
                self.attenuate = True


class SnpModel:
    """
    iCARE's special option for specifying a SNP model without the need to provide a
    reference dataset that the general-purpose covariate model requires.
    """
    age_start: Union[int, List[int]]
    age_interval_length: Union[int, List[int]]
    beta_estimates: np.ndarray
    z_profile: pd.DataFrame
    population_distribution: pd.DataFrame
    population_weights: np.ndarray
    family_history: FamilyHistory

    NUM_SAMPLES_IMPUTED: int = 10_000

    def __init__(self,
                 info_path: Union[str, pathlib.Path, None],
                 profile_path: Union[str, pathlib.Path, None],
                 family_history_variable_name: Optional[str],
                 age_start: Union[int, List[int]],
                 age_interval_length: Union[int, List[int]],
                 covariate_model: Optional[CovariateModel]) -> None:
        snp_names, betas, frequencies = extract_snp_info(info_path)
        profile = set_snp_profile(
            profile_path, age_start, snp_names, covariate_model, self.NUM_SAMPLES_IMPUTED
        )

        self.age_start, self.age_interval_length = utils.set_age_intervals(
            age_start, age_interval_length, profile, "apply_snp_profile"
        )
        self.family_history = FamilyHistory(covariate_model, family_history_variable_name, profile, snp_names,
                                            self.NUM_SAMPLES_IMPUTED)

import pathlib
from typing import List, Union, Optional, Tuple

import numpy as np
import pandas as pd

from icare import utils, check_errors
from icare.covariate_model import CovariateModel


class SnpModel:
    """
    iCARE's special option for the SNP model. This option
    removes the need for specifying a reference dataset that the typical
    covariate model requires.
    """
    age_start: List[int]
    age_interval_length: List[int]
    beta_estimates: np.ndarray
    z_profile: pd.DataFrame
    population_distribution: pd.DataFrame
    population_weights: np.ndarray

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
            self.instantiate_joint_snp_and_covariate_model(info, profile, family_history_variable_name)

    def instantiate_snp_only_model(
            self,
            info: Union[str, pathlib.Path, None],
            profile: Union[str, pathlib.Path, None],
            age_start: Union[int, List[int]],
            age_interval_length: Union[int, List[int]]) -> None:
        if info is None:
            raise ValueError("ERROR: You appear to be fitting a SNP-only model, and thus you "
                             "must provide relevant data to the 'model_snp_info' argument.")

        snp_names, betas, frequencies = self.extract_snp_info(info)

        num_instances_imputed = 10_000
        if profile is not None:
            profile = utils.read_file_to_dataframe(profile)
        else:
            if isinstance(age_start, int) and isinstance(age_interval_length, int):
                profile = pd.DataFrame(data=np.full((num_instances_imputed, snp_names.shape[0]), np.nan))
                print(f"\nNote: You did not provide an 'apply_snp_profile'. "
                      f"iCARE will impute SNPs for {num_instances_imputed:,} individuals.")
                print("If you want more/less, please specify an input to 'apply_snp_profile'.\n")
            else:
                profile = pd.DataFrame(data=np.full((len(age_start), snp_names.shape[0]), np.nan))
                print(f"\nNote: You did not provide an 'apply_snp_profile'. "
                      f"iCARE will impute SNPs for {len(age_start)} individuals, "
                      f"to match the specified number of age intervals.\n")

        family_history = dict()
        family_history["population"] = np.repeat(0, num_instances_imputed)
        family_history["profile"] = np.repeat(0, len(profile))
        family_history["attenuate"] = False
        print("\nNote: You did not provide a 'model_family_history_variable_name', therefore "
              "the model will not adjust the SNP imputations for family history.\n")

    def instantiate_joint_snp_and_covariate_model(self):
        pass

    def extract_snp_info(self, snp_info_path: Union[str, pathlib.Path, None]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        snp_info = utils.read_file_to_dataframe(snp_info_path)
        check_errors.check_snp_info(snp_info)
        snp_names = snp_info["snp_name"].values
        betas = np.log(snp_info["snp_odds_ratio"].values)
        frequencies = snp_info["snp_freq"].values
        return snp_names, betas, frequencies

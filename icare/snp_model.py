import pathlib
from typing import List, Union, Optional, Tuple

import numpy as np
import pandas as pd

from icare import utils, check_errors
from icare.covariate_model import CovariateModel


def extract_snp_info(info_path: Union[str, pathlib.Path, None]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    info = utils.read_file_to_dataframe(info_path)
    check_errors.check_snp_info(info)
    snp_names = info["snp_name"].values
    betas = np.log(info["snp_odds_ratio"].values)
    frequencies = info["snp_freq"].values
    return snp_names, betas, frequencies


def instantiate_snp_profile(profile_path: Union[str, pathlib.Path, None], age_start: Union[int, List[int]],
                            age_interval_length: Union[int, List[int]], snp_names: np.ndarray,
                            frequencies: np.ndarray) -> pd.DataFrame:
    if profile_path is not None:
        profile = utils.read_file_to_dataframe(profile_path)
        check_errors.check_snp_profile(profile, snp_names)


class SnpModel:
    """
    iCARE's special option for specifying a SNP model without the need to provide a
    reference dataset that the general-purpose covariate model requires.
    """
    beta_estimates: np.ndarray
    z_profile: pd.DataFrame
    population_distribution: pd.DataFrame
    population_weights: np.ndarray

    NUM_SAMPLES_IMPUTED = 10_000

    def __init__(
            self,
            info_path: Union[str, pathlib.Path, None],
            profile_path: Union[str, pathlib.Path, None],
            family_history_variable_name: Optional[str],
            age_start: Union[int, List[int]],
            age_interval_length: Union[int, List[int]],
            covariate_model: Optional[CovariateModel]) -> None:
        snp_only_model = covariate_model is None
        snp_names, betas, frequencies = extract_snp_info(info_path)
        z_profile = instantiate_snp_profile(profile_path, age_start, age_interval_length, snp_names, frequencies)

        if profile_path is not None:
            profile = utils.read_file_to_dataframe(profile_path)
        else:
            if isinstance(age_start, int) and isinstance(age_interval_length, int):
                profile = pd.DataFrame(data=np.full((self.NUM_SAMPLES_IMPUTED, snp_names.shape[0]), np.nan))
                print(f"\nNote: You did not provide an 'apply_snp_profile'. "
                      f"iCARE will impute SNPs for {self.NUM_SAMPLES_IMPUTED:,} individuals.")
                print("If you want more/less, please specify an input to 'apply_snp_profile'.\n")
            else:
                profile = pd.DataFrame(data=np.full((len(age_start), snp_names.shape[0]), np.nan))
                print(f"\nNote: You did not provide an 'apply_snp_profile'. "
                      f"iCARE will impute SNPs for {len(age_start)} individuals, "
                      f"to match the specified number of age intervals.\n")

        family_history = dict()
        family_history["population"] = np.repeat(0, self.NUM_SAMPLES_IMPUTED)
        family_history["profile"] = np.repeat(0, len(profile))
        family_history["attenuate"] = False
        print("\nNote: You did not provide a 'model_family_history_variable_name', therefore "
              "the model will not adjust the SNP imputations for family history.\n")

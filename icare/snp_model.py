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


class FamilyHistory:
    """ A data structure to contain family history information. """
    family_history_variable_name: Optional[str]
    population: np.ndarray
    profile: np.ndarray
    attenuate: bool

    def __init__(self, covariate_model: Optional[CovariateModel], family_history_variable_name: Optional[str],
                 num_samples_profile: int, default_num_samples: int) -> None:
        if covariate_model is None:
            self.population = np.repeat(0, default_num_samples)
            self.profile = np.repeat(0, num_samples_profile)
            self.attenuate = False

            print("\nNote: You did not provide a 'model_family_history_variable_name', therefore "
                  "the model will not adjust the SNP imputations for family history.\n")
        else:
            if family_history_variable_name is None:
                self.population = np.repeat(0, len(covariate_model.population_distribution))
                self.profile = np.repeat(0, num_samples_profile)
                self.attenuate = False

                print("\nNote: You did not provide a 'model_family_history_variable_name', therefore "
                      "the model will not adjust the SNP imputations for family history.\n")
            else:
                check_errors.check_family_history_variable_name_type(family_history_variable_name)
                family_history_variable_name = design_matrix.get_design_matrix_column_name_from_data_column_name(
                    covariate_model.population_distribution, family_history_variable_name)

                if family_history_variable_name is None:
                    raise ValueError("The 'model_family_history_variable_name' you provided does not exist in the "
                                     "covariate design matrix. Please provide a valid "
                                     "'model_family_history_variable_name'.")

                check_errors.check_family_history_variable(
                    family_history_variable_name, covariate_model.z_profile, covariate_model.population_distribution)

                self.family_history_variable_name = family_history_variable_name
                self.population = covariate_model.population_distribution[family_history_variable_name].values
                self.profile = covariate_model.z_profile[family_history_variable_name].values
                self.attenuate = True


def simulate_snps(snp_names: List[str], betas: np.ndarray, frequencies: np.ndarray,
                  family_history: np.ndarray, seed: Optional[int] = None) -> pd.DataFrame:
    """
    Simulate SNPs using the given SNP allele frequencies, log odds ratios, and family history information.

    The method assumes that SNPs are independent of each other and the genotype of each SNP follows the
    Hardy-Weinberg equilibrium in the reference population. If family history is provided, it is accounted
    for under the assumption of rare disease and that the SNPs have a multiplicative effect on the risk of
    the disease using Mendel's laws of inheritance.
    """
    simulated_snps = pd.DataFrame(data=np.zeros((len(family_history), len(snp_names))),
                                  columns=snp_names, dtype=np.float64)

    # probability distribution of genotypes for individuals with no family history of the disease
    genotype_distribution_fh_no = np.array([(1 - frequencies) ** 2,  # p**2
                                            2 * frequencies * (1 - frequencies),  # 2pq
                                            frequencies ** 2]).T  # q**2

    # probability distribution of genotypes for individuals with family history of the disease
    betas_matrix = np.repeat(betas.reshape(-1, 1), 3, axis=1)
    genotype_matrix = np.array([[0.0, 1.0, 2.0]] * len(betas))
    numerator = np.exp(betas_matrix * 0.5 * genotype_matrix) * genotype_distribution_fh_no
    denominator = np.repeat(np.sum(numerator, axis=1).reshape(-1, 1), 3, axis=1)
    genotype_distribution_fh_yes = numerator / denominator

    # sample genotypes
    rng = np.random.default_rng(seed)
    fh_yes = family_history.astype(int) == 1
    fh_no = ~fh_yes

    if np.any(fh_no):
        simulated_snps.loc[fh_no, :] = sample_genotype_from_distribution(
            np.array(fh_no).astype(int).sum(), genotype_distribution_fh_no, len(snp_names), rng)

    if np.any(fh_yes):
        simulated_snps.loc[fh_yes, :] = sample_genotype_from_distribution(
            np.array(fh_yes).astype(int).sum(), genotype_distribution_fh_yes, len(snp_names), rng)

    return simulated_snps


def sample_genotype_from_distribution(num_rows: int, genotype_distribution: np.ndarray, num_snps: int,
                                      rng: np.random.Generator) -> np.ndarray:
    uniform = rng.random((num_rows, num_snps))
    prob_genotype_0 = genotype_distribution[:, 0].reshape(-1, 1).T.repeat(num_rows, axis=0)
    prob_genotype_0_or_1 = genotype_distribution[:, 0:2].sum(axis=1).reshape(-1, 1).T.repeat(num_rows, axis=0)

    return np.array(uniform > prob_genotype_0).astype(int) + np.array(uniform > prob_genotype_0_or_1).astype(int)


class SnpModel:
    """
    iCARE's special option for specifying a SNP model without the need to provide a
    reference dataset that the general-purpose covariate model requires.

    Caution: this method mutates the parameters of the covariate model passed into it.
    """
    age_start: Union[int, List[int]]
    age_interval_length: Union[int, List[int]]
    beta_estimates: np.ndarray
    profile: pd.DataFrame
    z_profile: pd.DataFrame  # z_profile is the design matrix generated from profile; for SNP model, they are the same
    population_distribution: pd.DataFrame
    population_weights: np.ndarray
    family_history: FamilyHistory

    DEFAULT_NUM_SAMPLES_IMPUTED: int = 10_000

    def __init__(self,
                 info_path: Union[str, pathlib.Path, None],
                 profile_path: Union[str, pathlib.Path, None],
                 family_history_variable_name: Optional[str],
                 age_start: Union[int, List[int]],
                 age_interval_length: Union[int, List[int]],
                 num_imputations: int,
                 covariate_model: Optional[CovariateModel],
                 seed: Optional[int] = None) -> None:
        snp_names, betas, frequencies = extract_snp_info(info_path)

        self._set_z_profile(profile_path, age_start, snp_names, covariate_model)
        self._set_profile()
        self.age_start, self.age_interval_length = utils.set_age_intervals(
            age_start, age_interval_length, len(self.z_profile), "apply_snp_profile_path")
        self._set_family_history(covariate_model, family_history_variable_name)

        self._set_population_distribution(covariate_model, snp_names, betas, frequencies, num_imputations, seed)
        self._set_population_weights(covariate_model, num_imputations)
        self._set_beta_estimates(covariate_model, betas, frequencies)

    def _set_z_profile(self, profile_path: Union[str, pathlib.Path, None], age_start: Union[int, List[int]],
                       snp_names: List[str], covariate_model: Optional[CovariateModel]) -> None:
        if profile_path is not None:
            profile = utils.read_file_to_dataframe_given_dtype(profile_path, dtype=float)
            profile = profile[snp_names]
            check_errors.check_snp_profile(profile, snp_names)
            if covariate_model is not None:
                if len(profile) != len(covariate_model.z_profile):
                    raise ValueError(f"ERROR: The number of individuals in the 'apply_snp_profile_path' ({len(profile)})"
                                     f" does not match the number of individuals in the 'apply_covariate_profile'"
                                     f"({len(covariate_model.z_profile)}).")
            self.z_profile = profile
            return

        def create_empty_snp_profile(num_rows: int, columns: List[str]) -> pd.DataFrame:
            return pd.DataFrame(data=np.full((num_rows, len(columns)), np.nan), columns=columns)

        if covariate_model is None:
            if isinstance(age_start, int):
                num_samples = self.DEFAULT_NUM_SAMPLES_IMPUTED
                print(f"\nNote: You included 'model_snp_info' but did not provide an 'apply_snp_profile_path'. "
                      f"iCARE will impute SNPs for {num_samples} individuals. If you require more, "
                      f"please provide an input to 'apply_snp_profile_path' input.\n")
                self.z_profile = create_empty_snp_profile(num_samples, snp_names)
            else:
                num_samples = len(age_start)
                print(f"\nNote: You included 'model_snp_info' but did not provide an 'apply_snp_profile_path'. "
                      f"iCARE will impute SNPs for {num_samples} individuals, matching the number of"
                      f" age intervals specified.\n")
                self.z_profile = create_empty_snp_profile(num_samples, snp_names)
        else:
            num_samples = len(covariate_model.z_profile)
            print(f"\nNote: You included 'model_snp_info' but did not provide an 'apply_snp_profile_path'. "
                  f"iCARE will impute SNPs for {num_samples} individuals, matching the number of"
                  f" individuals in the specified 'apply_covariate_profile'.\n")
            self.z_profile = create_empty_snp_profile(num_samples, snp_names)

    def _set_family_history(self, covariate_model: Optional[CovariateModel],
                            family_history_variable_name: Optional[str]) -> None:
        self.family_history = FamilyHistory(covariate_model, family_history_variable_name, len(self.z_profile),
                                            self.DEFAULT_NUM_SAMPLES_IMPUTED)

    def _set_population_distribution(self, covariate_model: CovariateModel, snp_names: List[str], betas: np.ndarray,
                                     frequencies: np.ndarray, num_imputations: int, seed: Optional[int] = None) -> None:
        population_fh = np.tile(self.family_history.population, num_imputations)
        self.population_distribution = simulate_snps(snp_names, betas, frequencies, population_fh, seed)

        if covariate_model is not None:
            stacked_population_distribution = np.tile(covariate_model.population_distribution, (num_imputations, 1))
            covariate_model.population_distribution = pd.DataFrame(
                data=stacked_population_distribution, columns=covariate_model.population_distribution.columns)

    def _set_population_weights(self, covariate_model: CovariateModel, num_imputations: int) -> None:
        if covariate_model is None:
            self.population_weights = np.ones(len(self.population_distribution)) / len(self.population_distribution)
        else:
            stacked_weights = np.tile(covariate_model.population_weights, num_imputations)
            covariate_model.population_weights = stacked_weights
            self.population_weights = stacked_weights

    def _set_beta_estimates(self, covariate_model: CovariateModel, betas: np.ndarray, frequencies: np.ndarray) -> None:
        self.beta_estimates = betas

        if self.family_history.attenuate:
            def adjust_beta_for_family_history(betas: np.ndarray, frequencies: np.ndarray,
                                               beta_family_history: float) -> float:
                return beta_family_history - (0.5 * np.sum(betas ** 2 * 2 * frequencies * (1 - frequencies)))

            family_history_column_index = covariate_model.population_distribution.columns.get_loc(
                self.family_history.family_history_variable_name)
            beta_family_history = covariate_model.beta_estimates[family_history_column_index]
            covariate_model.beta_estimates[family_history_column_index] = adjust_beta_for_family_history(
                betas, frequencies, beta_family_history)

    def _set_profile(self):
        self.profile = self.z_profile.copy()

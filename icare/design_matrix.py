import tokenize
from collections import OrderedDict
from typing import Optional

import numpy as np
import pandas as pd
from patsy import dmatrix
from patsy.tokens import python_tokenize


def get_arbitrary_column_values(df: pd.DataFrame) -> pd.Series:
    return df.apply(lambda x: x.value_counts().index[0])


def create_missing_values_mask(df: pd.DataFrame) -> pd.DataFrame:
    return df.isna()


def impute_dataframe(df: pd.DataFrame, values: pd.Series) -> pd.DataFrame:
    return df.fillna(values)


def get_python_name_tokens(factor_name: str) -> str:
    return_string = False
    for token in python_tokenize(factor_name):
        token_type, token_name = token[0], token[1]
        if token_type == tokenize.NAME and token_name == "Q":
            return_string = True
        elif token_type == tokenize.STRING and return_string:
            yield token_name[1:-1]
            return_string = False
        if token_type == tokenize.NAME:
            yield token_name


def get_design_matrix_missing_pattern(design_matrix: pd.DataFrame, missing_mask: pd.DataFrame) -> pd.DataFrame:
    design_matrix_missing_pattern = pd.DataFrame(
        data=np.zeros(shape=design_matrix.shape, dtype=bool),
        index=design_matrix.index,
        columns=design_matrix.columns
    )

    for term, term_slice in design_matrix.design_info.term_slices.items():
        num_columns = term_slice.stop - term_slice.start
        term_missing_pattern = np.zeros(shape=(len(design_matrix),), dtype=bool)

        for factor in term.factors:
            data_columns_in_factor = [token for token in get_python_name_tokens(factor.name()) if
                                      token in missing_mask.columns]
            factor_missing_pattern = missing_mask[data_columns_in_factor]
            if len(data_columns_in_factor) > 1:
                factor_missing_pattern = factor_missing_pattern.any(axis=0)
            term_missing_pattern = term_missing_pattern | factor_missing_pattern.values.ravel()
        term_missing_pattern = np.stack([term_missing_pattern] * num_columns, axis=1)
        design_matrix_missing_pattern.iloc[:, term_slice] = term_missing_pattern

    return design_matrix_missing_pattern


def reintroduce_missing_values(design_matrix: pd.DataFrame, missing_pattern: pd.DataFrame):
    design_matrix[missing_pattern] = np.nan


def get_design_matrix_column_name_from_data_column_name(design_matrix: pd.DataFrame,
                                                        data_column_name: str) -> Optional[str]:
    all_data_columns = design_matrix.design_info.original_column_names

    for term, term_slice in design_matrix.design_info.term_slices.items():
        num_columns = term_slice.stop - term_slice.start
        if num_columns > 1 or len(term.factors) > 1:
            continue

        for factor in term.factors:
            data_columns_in_factor = [token for token in get_python_name_tokens(factor.name()) if
                                      token in all_data_columns]
            if len(data_columns_in_factor) > 1:
                continue

            if data_column_name in data_columns_in_factor:
                return design_matrix.columns[term_slice][0]
    return None


class DesignInfo():
    column_names: list
    column_name_indexes: OrderedDict
    term_names: list
    term_name_slices: OrderedDict
    terms: list
    term_slices: OrderedDict
    term_codings: OrderedDict

    def __init__(self, design_matrix: pd.DataFrame):
        self.column_names = design_matrix.design_info.column_names
        self.column_name_indexes = design_matrix.design_info.column_name_indexes
        self.term_names = design_matrix.design_info.term_names
        self.term_name_slices = design_matrix.design_info.term_name_slices
        self.terms = design_matrix.design_info.terms
        self.term_slices = design_matrix.design_info.term_slices
        self.term_codings = design_matrix.design_info.term_codings


def remove_intercept(design_matrix: pd.DataFrame) -> pd.DataFrame:
    if "Intercept" != design_matrix.columns[0]:
        if "Intercept" in design_matrix.columns:
            raise ValueError("ERROR: Please remove the intercept term from the model formula supplied in"
                             " 'model_covariate_formula_path'")
        return design_matrix

    design_matrix.pop("Intercept")

    design_info = DesignInfo(design_matrix)
    design_info.column_names = [column_name for column_name in design_info.column_names if column_name != "Intercept"]
    design_info.column_name_indexes = OrderedDict(
        [(key, value - 1) for key, value in design_info.column_name_indexes.items() if key != "Intercept"])
    design_info.term_names = [term_name for term_name in design_info.term_names if term_name != "Intercept"]
    design_info.term_name_slices = OrderedDict([(key, slice(value.start - 1, value.stop - 1)) for key, value in
                                                design_info.term_name_slices.items() if key != "Intercept"])
    design_info.terms = [term for term in design_info.terms if len(term.factors) != 0]
    design_info.term_slices = OrderedDict(
        [(key, slice(value.start - 1, value.stop - 1)) for key, value in design_info.term_slices.items() if
         len(key.factors) != 0])
    design_info.term_codings = OrderedDict(
        [(key, value) for key, value in design_info.term_codings.items() if len(key.factors) != 0])

    design_matrix.design_info = design_info

    return design_matrix


def build_design_matrix(formula: str, dataset: pd.DataFrame) -> pd.DataFrame:
    design_matrix = dmatrix(formula, dataset, return_type="dataframe")
    design_matrix = remove_intercept(design_matrix)
    design_matrix.design_info.original_column_names = list(dataset.columns)

    return design_matrix


def build_design_matrix_with_missing_values(formula: str, dataset: pd.DataFrame,
                                            dataset_complete: pd.DataFrame) -> pd.DataFrame:
    missing_mask = create_missing_values_mask(dataset)
    dataset_imputed = impute_dataframe(dataset, get_arbitrary_column_values(dataset_complete))
    design_matrix = build_design_matrix(formula, dataset_imputed)
    reintroduce_missing_values(design_matrix, get_design_matrix_missing_pattern(design_matrix, missing_mask))

    return design_matrix

import tokenize

import numpy as np
import pandas as pd
from patsy.tokens import python_tokenize


def get_arbitrary_column_values(df: pd.DataFrame) -> pd.Series:
    return df.apply(lambda x: x.value_counts().index[0])


def create_missing_values_mask(df: pd.DataFrame) -> pd.DataFrame:
    return df.isna()


def impute_dataframe(df: pd.DataFrame, values: pd.Series) -> pd.DataFrame:
    return df.fillna(values)


def get_python_name_tokens(factor_name: str) -> str:
    for token in python_tokenize(factor_name):
        token_type, token_name = token[0], token[1]
        if token_type == tokenize.NAME:
            yield token_name


def get_design_matrix_missing_pattern(design_matrix: pd.DataFrame, missing_mask: pd.DataFrame) -> pd.DataFrame:
    design_matrix_missing_pattern = pd.DataFrame(
        data=np.zeros(shape=design_matrix.shape, dtype=bool),
        columns=design_matrix.columns
    )

    for term, term_slice in design_matrix.design_info.term_slices.items():
        num_columns = term_slice.stop - term_slice.start
        term_missing_pattern = np.zeros(shape=(len(design_matrix),), dtype=bool)

        for factor in term.factors:
            data_columns_in_factor = [token for token in get_python_name_tokens(factor.name()) if token in missing_mask.columns]
            factor_missing_pattern = missing_mask[data_columns_in_factor]
            if len(data_columns_in_factor) > 1:
                factor_missing_pattern = factor_missing_pattern.any(axis=0)
            term_missing_pattern = term_missing_pattern | factor_missing_pattern.values.ravel()
        term_missing_pattern = np.stack([term_missing_pattern] * num_columns, axis=1)
        design_matrix_missing_pattern.iloc[:, term_slice] = term_missing_pattern

    return design_matrix_missing_pattern

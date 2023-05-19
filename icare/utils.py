import json
import pathlib
from typing import Union, List, Tuple

import pandas as pd

from icare import check_errors


def read_file_to_string(file: Union[str, pathlib.Path]) -> str:
    with open(file, mode='r') as f:
        return ' '.join(f.read().splitlines())


def read_file_to_dict(file: Union[str, pathlib.Path]) -> dict:
    with open(file, mode='r') as f:
        return json.load(f)


def read_file_to_dataframe(file: Union[str, pathlib.Path], allow_integers: bool = True) -> pd.DataFrame:
    df = pd.read_csv(file)

    if 'id' in df.columns:
        df.set_index('id', inplace=True)

    if not allow_integers:  # to support nullable integer types
        numeric_columns = df.select_dtypes(include=['number']).columns
        df[numeric_columns] = df[numeric_columns].astype(float)

    return df


def read_file_to_dataframe_given_dtype(file, dtype) -> pd.DataFrame:
    header = pd.read_csv(file, nrows=1).columns
    if 'id' in header:
        if isinstance(dtype, dict) and 'id' not in dtype:
            dtype = {'id': str, **dtype}
        else:
            dtype = {'id': str, **{col: dtype for col in header if col != 'id'}}

    df = pd.read_csv(file, dtype=dtype)

    if 'id' in df.columns:
        df.set_index('id', inplace=True)

    return df


def set_age_intervals(age_start: Union[int, List[int]], age_interval_length: Union[int, List[int]],
                      num_samples_profile: int, profile_name: str) -> Tuple[List[int], List[int]]:
    if isinstance(age_start, int):
        age_start = [age_start] * num_samples_profile

    if isinstance(age_interval_length, int):
        age_interval_length = [age_interval_length] * num_samples_profile

    if len(age_start) != num_samples_profile or len(age_interval_length) != num_samples_profile:
        raise ValueError(f"ERROR: the number of values in 'apply_age_start' and 'apply_age_interval_length', "
                         f"and the number of rows in '{profile_name}' must match.")

    check_errors.check_age_intervals(age_start, age_interval_length)

    return age_start, age_interval_length

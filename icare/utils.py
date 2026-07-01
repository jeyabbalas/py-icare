import json
import os
import pathlib
from typing import Union, List, Tuple

import pandas as pd

from icare import check_errors


def read_file_to_string(file: Union[str, pathlib.Path]) -> str:
    """Read a Patsy formula from a file, or accept one inline.

    A ``pathlib.Path`` (or a ``str`` naming an existing file) is read from disk; any other ``str`` is
    treated as an inline formula. In both cases line breaks are collapsed to single spaces.
    """
    if isinstance(file, pathlib.Path):
        with open(file, mode='r') as f:
            return ' '.join(f.read().splitlines())
    if isinstance(file, str) and os.path.exists(file):
        with open(file, mode='r') as f:
            return ' '.join(f.read().splitlines())
    return ' '.join(str(file).splitlines())


def read_file_to_dict(file: Union[str, pathlib.Path, dict]) -> dict:
    if isinstance(file, dict):
        return file
    with open(file, mode='r') as f:
        return json.load(f)


def read_file_to_dataframe(file: Union[str, pathlib.Path, pd.DataFrame],
                           allow_integers: bool = True) -> pd.DataFrame:
    df = file.copy() if isinstance(file, pd.DataFrame) else pd.read_csv(file)

    if 'id' in df.columns:
        df.set_index('id', inplace=True)

    if not allow_integers:  # to support nullable integer types
        numeric_columns = df.select_dtypes(include=['number']).columns
        df[numeric_columns] = df[numeric_columns].astype(float)

    return df


def read_file_to_dataframe_given_dtype(file, dtype) -> pd.DataFrame:
    if isinstance(file, pd.DataFrame):
        df = file.copy()
        columns = df.columns
        if 'id' in columns:
            if isinstance(dtype, dict) and 'id' not in dtype:
                dtype = {'id': str, **dtype}
            else:
                dtype = {'id': str, **{col: dtype for col in columns if col != 'id'}}
        if isinstance(dtype, dict):
            # astype() raises on keys absent from the frame; read_csv(dtype=...) silently tolerates them.
            dtype = {col: col_dtype for col, col_dtype in dtype.items() if col in df.columns}
        df = df.astype(dtype)
        if 'id' in df.columns:
            df.set_index('id', inplace=True)
        elif df.index.name == 'id':
            df.index = df.index.astype(str)
        return df

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


def read_file_to_dataframe_raw(file: Union[str, pathlib.Path, pd.DataFrame]) -> pd.DataFrame:
    """Read tabular data without normalization (no id-indexing, no dtype coercion).

    A DataFrame is copied so callers that pass one in memory are not mutated by downstream in-place edits.
    """
    return file.copy() if isinstance(file, pd.DataFrame) else pd.read_csv(file)


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

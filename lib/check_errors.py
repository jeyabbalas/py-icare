import numpy as np
import pandas as pd


def check_age_lengths(apply_age_start, apply_age_interval_length, match, match_name):
    if isinstance(apply_age_start, int):
        apply_age_start = np.full((match.shape[0],), apply_age_start)

    if isinstance(apply_age_interval_length, int):
        apply_age_interval_length = np.full((match.shape[0],), apply_age_interval_length)

    if apply_age_start.shape[0] != apply_age_interval_length.shape[0]:
        raise ValueError("ERROR: 'apply_age_start and 'apply_age_interval_length' must have the same length.")

    if apply_age_start.shape[0] != match.shape[0]:
        raise ValueError(f"ERROR: 'apply_age_start' and number of rows in '{match_name}' must match.")

    if (sum(np.isnan(apply_age_start)) + sum(np.isnan(apply_age_interval_length))) > 0:
        raise ValueError("ERROR: 'apply_age_start' and 'apply_age_interval_length' must not contain missing values.")

    if ((sum(apply_age_start < 0)) + sum(apply_age_interval_length < 0)) > 0:
        raise ValueError("ERROR: 'apply_age_start' and 'apply_age_interval_length' must contain positive values.")

    return apply_age_start, apply_age_interval_length


def check_snp_info(model_snp_info):
    if not isinstance(model_snp_info, pd.DataFrame):
        raise ValueError("ERROR: If specified, the argument 'model_snp_info' requires a Pandas dataframe (that "
                         "contains the information on SNP names, odds ratios, and allele frequencies.")

    if "snp_name" not in model_snp_info.columns or \
            "snp_odds_ratio" not in model_snp_info.columns or \
            "snp_freq" not in model_snp_info.columns:
        raise ValueError("ERROR: If specified, the argument 'model_snp_info' must be a Pandas dataframe with at least "
                         "3 columns named: 'snp_name', 'snp_odds_ratio', and 'snp_freq'.")


def format_flexible_rate_inputs(data):
    if len(data.columns) == 3:
        integer_ages = np.array(range(np.min(data["start_age"]), np.min(data["end_age"])))
        data_formatted = pd.DataFrame(columns=["age", "rate"], index=np.array(range(integer_ages.shape[0])))
        data_formatted["age"] = integer_ages

        for i in range(data.shape[0]):
            idxs = np.where((data_formatted["age"] >= data.loc[i, "start_age"]) &
                            (data_formatted["age"] <= data.loc[i, "end_age"]))
            data_formatted.loc[idxs, "rate"] = data_formatted.loc[i, "rate"]/idxs[0].shape[0]

        return data_formatted
    else:
        return data


def check_flexible_rate_inputs(data, data_name):
    if not isinstance(data, pd.DataFrame):
        raise ValueError(f"ERROR: argument '{data_name}' requires a Pandas DataFrame as its input.")

    if len(data.columns) != 2 and len(data.columns) != 3:
        raise ValueError(f"ERROR: argument '{data_name}' requires a Pandas DataFrame with either 2 ('age', 'rate') or "
                         f"3 ('start_age', 'end_age', 'rate') columns.")

    if len(data.columns) == 2:
        if "age" not in data.columns or "rate" not in data.columns:
            raise ValueError(f"ERROR: argument '{data_name}' requires a Pandas DataFrame with columns:"
                             f" ['age', 'rate'].")

        if sum(data["age"] % 1) != 0:
            raise ValueError(f"ERROR: The 'age' column in the Pandas DataFrame, passed into argument {data_name}, "
                             f"should be integers.")

    if len(data.columns) == 3:
        if "start_age" not in data.columns or "end_age" not in data.columns or "rate" not in data.columns:
            raise ValueError(f"ERROR: argument '{data_name}' requires a Pandas DataFrame with columns:"
                             f" ['start_age', 'end_age', 'rate'].")

        if data.shape[0] > 1 and (sum(data.loc[1:, "start_age"] - data.loc[:data.shape[0]-1, "end_age"]) != 0):
            raise ValueError(f"ERROR: The rates provided in that Pandas DataFrame in the argument '{data_name}' must "
                             f"cover sequential age intervals (i.e. if an interval ends at age 30, the next interval "
                             f"must start at age 31).")

    if (sum(data["rate"] < 0.0) + sum(data["rate"] > 1.0)) > 0:
        raise ValueError("ERROR: The rates should be probabilities between 0 and 1.")

    return format_flexible_rate_inputs(data)


def check_rates(model_competing_incidence_rates, model_disease_incidence_rates, apply_age_start,
                apply_age_interval_length):
    lambda_vals = check_flexible_rate_inputs(model_disease_incidence_rates, "model_disease_incidence_rates")

    if model_competing_incidence_rates is None:
        model_competing_incidence_rates = pd.DataFrame(data=np.vstack((lambda_vals["age"],
                                                                       np.zeros(lambda_vals.shape[0]))).transpose(),
                                                       columns=["age", "rate"])

    model_competing_incidence_rates = check_flexible_rate_inputs(model_competing_incidence_rates,
                                                                 "model_competing_incidence_rates")

    if sum([x not in lambda_vals["age"] for
            x in range(np.min(apply_age_start), np.max(apply_age_start + apply_age_interval_length))]) > 0:
        raise ValueError("ERROR: The 'model_disease_incidence_rates' input must have age-specific rates for each "
                         "integer age covered by the prediction intervals defined by 'apply_age_start' and "
                         "'apply_age_interval_length'. You must make these inputs consistent with one "
                         "another to proceed.")

    if sum([x not in model_competing_incidence_rates["age"] for
            x in range(np.min(apply_age_start), np.max(apply_age_start + apply_age_interval_length))]) > 0:
        raise ValueError("ERROR: The 'model_competing_incidence_rates' input must have age-specific rates for each "
                         "integer age covered by the prediction intervals defined by 'apply_age_start' and "
                         "'apply_age_interval_length'. You must make these inputs consistent with one "
                         "another to proceed.")
    
    return lambda_vals, model_competing_incidence_rates

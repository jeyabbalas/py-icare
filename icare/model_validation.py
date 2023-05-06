import pathlib
from typing import Union, List, Optional

import numpy as np
import pandas as pd

from icare import check_errors


class ModelValidation:
    study_data: pd.DataFrame
    timeframe: str
    predicted_time_interval: np.array
    dataset_name: str
    model_name: str

    def __init__(self,
                 study_data_path: Union[str, pathlib.Path],
                 predicted_risk_interval: Union[str, int, List[int]],
                 icare_model_parameters: Optional[dict],
                 predicted_risk_variable_name: Optional[str],
                 linear_predictor_variable_name: Optional[str],
                 reference_entry_age: Union[int, List[int], None],
                 reference_exit_age: Union[int, List[int], None],
                 reference_predicted_risks: Optional[List[float]],
                 reference_linear_predictors: Optional[List[float]],
                 number_of_percentiles: int,
                 linear_predictor_cutoffs: Optional[List[float]],
                 dataset_name: str,
                 model_name: str) -> None:
        self._set_study_data(study_data_path)
        self._set_predicted_time_interval(predicted_risk_interval)
        self._calculate_followup_period()
        # self._calculate_risk(icare_model_parameters, predicted_risk_variable_name, linear_predictor_variable_name)
        # self._calculate_reference_risk
        self.dataset_name = dataset_name
        self.model_name = model_name

    def _set_study_data(self, study_data_path: Union[str, pathlib.Path]) -> None:
        # load study data and set data types
        self.study_data = pd.read_csv(study_data_path)

        mandatory_columns = ["observed_outcome", "study_entry_age", "study_exit_age", "time_of_onset"]
        check_errors.check_data_columns(self.study_data, mandatory_columns)
        integer_columns = ["observed_outcome", "study_entry_age", "study_exit_age"]
        self.study_data[integer_columns] = self.study_data[integer_columns].astype(int)
        float_columns = ["time_of_onset"]
        self.study_data[float_columns] = self.study_data[float_columns].astype(float)

        if "sampling_weights" in self.study_data.columns:
            check_errors.check_data_columns(self.study_data, ["sampling_weights"])
            self.study_data["sampling_weights"] = self.study_data["sampling_weights"].astype(float)

        if "id" in self.study_data.columns:
            self.study_data.set_index("id", inplace=True)

        # check data
        check_errors.check_study_data(self.study_data)
        self.study_data["observed_followup"] = self.study_data["study_exit_age"] - self.study_data["study_entry_age"]

        # censor cases where the time of onset is after the observed follow-up period
        case_onset_after_followup = (self.study_data["observed_outcome"] == 1) & \
                                    (self.study_data["time_of_onset"] > self.study_data["observed_followup"])
        self.study_data.loc[case_onset_after_followup, "observed_outcome"] = 0
        self.study_data.loc[case_onset_after_followup, "time_of_onset"] = float("inf")

    def _set_predicted_time_interval(self, predicted_risk_interval: Union[str, int, List[int]]) -> None:
        check_errors.check_validation_time_interval_type(predicted_risk_interval, self.study_data)

        if isinstance(predicted_risk_interval, str):
            self.timeframe = "Observed follow-up"
            self.predicted_risk_interval = self.study_data["observed_followup"].values
        elif isinstance(predicted_risk_interval, int):
            if predicted_risk_interval == 1:
                self.timeframe = "1 year"
            else:
                self.timeframe = f"{predicted_risk_interval} years"
            self.predicted_risk_interval = np.array([predicted_risk_interval] * len(self.study_data))
        else:
            self.timeframe = "Varies across individuals"
            self.predicted_risk_interval = np.array(predicted_risk_interval)

    def _calculate_followup_period(self):
        self.study_data["followup"] = self.study_data["observed_followup"]

        # follow-up period is the minimum of the predicted risk interval and the observed follow-up period
        onset_within_interval = (self.study_data["time_of_onset"] <= self.predicted_risk_interval)
        interval_ends_before_followup = (self.predicted_risk_interval <= self.study_data["observed_followup"])
        self.study_data.loc[onset_within_interval & interval_ends_before_followup, "followup"] = \
            self.predicted_risk_interval[onset_within_interval & interval_ends_before_followup]

        # censor cases when the time of onset is after the predicted risk interval
        onset_after_interval = (self.study_data["time_of_onset"] > self.predicted_risk_interval)
        onset_before_followup = (self.study_data["time_of_onset"] <= self.study_data["observed_followup"])
        self.study_data.loc[onset_after_interval & onset_before_followup, "observed_outcome"] = 0
        self.study_data.loc[onset_after_interval & onset_before_followup, "followup"] = \
            self.predicted_risk_interval[onset_after_interval & onset_before_followup]

        # censor cases when onset is after the observed follow-up period
        observed_longer_than_interval = (self.study_data["observed_followup"] >= self.predicted_risk_interval)
        onset_after_followup = (self.study_data["time_of_onset"] > self.study_data["observed_followup"])
        self.study_data.loc[observed_longer_than_interval & onset_after_followup, "followup"] = \
            self.predicted_risk_interval[observed_longer_than_interval & onset_after_followup]

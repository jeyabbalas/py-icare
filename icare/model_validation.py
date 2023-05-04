import pathlib
from typing import Union, List, Optional

import pandas as pd

from icare import check_errors


class ModelValidation:
    study_data: pd.DataFrame
    timeframe: str
    predicted_time_interval: List[int]
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
        self.dataset_name = dataset_name
        self.model_name = model_name

    def _set_study_data(self, study_data_path: Union[str, pathlib.Path]) -> None:
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

        check_errors.check_study_data(self.study_data)

    def _set_predicted_time_interval(self, predicted_risk_interval: Union[str, int, List[int]]) -> None:
        check_errors.check_validation_time_interval_type(predicted_risk_interval)

        if isinstance(predicted_risk_interval, str):
            self.timeframe = "Observed follow-up"

        self.timeframe = "Observed follow-up" if predicted_risk_interval == "total-followup" else \
            f"{predicted_risk_interval} year(s)"

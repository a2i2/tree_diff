"""
Defines the configuration schema for the project.
"""

from dataclasses import dataclass
from typing import List, Callable

from surround import BaseConfig, config


@dataclass
class DataSet:
    ml_input_data_folder: str
    training_input_file_name: str
    validate_input_file_name: str
    y_column: str
    features: List[str]
    string_columns: List[str]


DATASETS = {
    "adult": DataSet(
        ml_input_data_folder="adult/batch_1",
        training_input_file_name="adult1.data",
        validate_input_file_name="adult1.test",
        features = ["age","workclass","fnlwgt","education","education-num","marital","occupation","relationship","race","sex","capital-gain","capital-loss","hours","native"],
        y_column = "income",
        string_columns = ["workclass","education","marital","occupation","relationship","race","sex","native"])
}


@config(name="config")
@dataclass
class Config(BaseConfig):
    # Docker image configuration
    company: str = "yourcompany"
    image: str = "simple_example"
    version: str = "latest"

    # Pipeline configuration
    runner: str = "0"
    assembler: str = "baseline"

    # Dataset to use
    dataset: str = "adult"

    # Pipeline run configuration
    mode: str = "predict"
    status: bool = False

    # Training and Validation config
    cv: int = 10
    test_percentage: float = 0.2
    validation_percentage: float = 0.1

"""
This module is responsible for loading the data and running the pipeline.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from surround import Runner, RunMode

from .config import Config, DATASETS
from .stages import AssemblerState

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

class FileSystemRunner(Runner):

    def load_data(self, mode, config: Config):
        state = AssemblerState()
        if mode == RunMode.BATCH_PREDICT:
            #state.input_data = pd.read_csv(Path(config.input_file_name))
            LOGGER.info("Batch predict mode not implemented")
            sys.exit(1)
        elif mode == RunMode.TRAIN:
            LOGGER.info("Running training mode")

            dataset = DATASETS[config.dataset]
            input_path = Path(dataset.ml_input_data_folder)


            if input_path.is_absolute():
                directory_path = input_path
            else:
                # TODO: Fix config.input_path, convoluted with output_path
                directory_path = Path(config.input_path, dataset.ml_input_data_folder)
            p =directory_path.absolute()
            LOGGER.info(f"Training dataset: {config.dataset}")

            # Load read other records
            def load_data(data_set_file_name):
                temp_df = pd.read_csv(Path(directory_path, data_set_file_name),
                                   delimiter=",\\s",
                                   engine="python")
                temp_df.columns = dataset.features + [dataset.y_column]
                return temp_df

            training_observations = load_data(dataset.training_input_file_name)
            validation_observations = load_data(dataset.validate_input_file_name)

            # While developing algorithm use cross validation on the training observations only
            data = training_observations

            #data = pd.concat([training_observations, validation_observations])
            state.training_records = data[dataset.features]
            state.labels = data[dataset.y_column]
            state.string_columns = dataset.string_columns

        else:
            LOGGER.info("No prediction pipeline")
            sys.exit(1)
        return state

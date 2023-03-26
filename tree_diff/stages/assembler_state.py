"""
This module defines the state object that is passed between each stage
in the pipeline.
"""
from dataclasses import dataclass

from surround import State

class AssemblerState(State):

    def __init__(self):
        super().__init__()


    training_records = None
    labels = None
    string_columns = None

    input_data = None
    training_observations = None
    validation_observations = None
    output_data = None
    validation_predictions = None
    missed_by_model = None
    missed_by_system = None
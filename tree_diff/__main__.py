"""
Main entry-point for the Surround project.
Runners and assemblies are defined in here.
"""

import os
from pathlib import Path

import hydra
from surround import Surround, Assembler

from .config import Config
from .stages import BaselineModels
from .file_system_runner import FileSystemRunner

from hydra.core.hydra_config import HydraConfig

RUNNERS = [
    FileSystemRunner()
]

ASSEMBLIES = [

    # Assembler("baseline")
    #     .set_stages([Baseline(), MissedPredictions()]),

    Assembler("baseline")
    .set_stages([BaselineModels()])


]

@hydra.main(config_name="config")
def main(config: Config):
    # TODO: Surround needs to set the same path for Hydra and output folders
    config.output_path = Path(config.package_path, HydraConfig.get().run.dir).resolve()

    surround = Surround(
        RUNNERS,
        ASSEMBLIES,
        config,
        "simple_example",
        "Simple example ",
        os.path.dirname(os.path.dirname(__file__))
    )

    if config.status:
        surround.show_info()
    else:
        surround.run(config.runner, config.assembler, config.mode)

if __name__ == "__main__":
    main(None)

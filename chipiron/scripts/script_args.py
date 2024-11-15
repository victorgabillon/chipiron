from dataclasses import dataclass
from datetime import datetime

from chipiron.utils import path


@dataclass
class BaseScriptArgs:
    """
    Dataclass representing the arguments for the Script class.
    """

    # whether the script is profiling computation usage
    profiling: bool = False

    # whether the script is testing the code (using pytest for instance)
    testing: bool = False

    # the folder where to output the results
    experiment_output_folder: path | None = None

    def __post_init__(self) -> None:
        if self.experiment_output_folder is None:
            now = datetime.now()  # current date and time

            self.experiment_output_folder = now.strftime(
                "%A-%m-%d-%Y--%H:%M:%S:%f")

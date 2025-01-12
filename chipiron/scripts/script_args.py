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

    # whether the script should force the behavior of the algorithm be only dependent on the seed and number
    # and on no other parameter ( could happen that using some external functions/modules add some randomness)
    universal_behavior: bool = False

    # if experiment_output_folder is specified it is used as is. If not it is build from default path plus the
    # relative_script_instance_experiment_output_folder

    # the folder where to output the results
    experiment_output_folder: path | None = None

    # specific folder name  where to output the results (if none it is set to time and day in post init)
    relative_script_instance_experiment_output_folder: path | None = None

    # the seed
    seed: int = 0

    def __post_init__(self) -> None:

        # if relative_script_instance_experiment_output_folde is not set, it gets time and day
        if self.relative_script_instance_experiment_output_folder is None:
            now = datetime.now()  # current date and time
            self.relative_script_instance_experiment_output_folder = now.strftime(
                "%A-%m-%d-%Y--%H:%M:%S:%f"
            )

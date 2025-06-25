"""
This module contains the Script class which is responsible for launching scripts.
It handles computing execution time, profiling, and parsing arguments.
"""

import cProfile
import io
import os
import pprint
import pstats
import time
from pstats import SortKey
from typing import Any, Protocol, runtime_checkable

from parsley_coco import Parsley

from chipiron.scripts.script_args import BaseScriptArgs
from chipiron.utils import path
from chipiron.utils.dataclass import IsDataclass
from chipiron.utils.logger import chipiron_logger, suppress_logging
from chipiron.utils.small_tools import mkdir_if_not_existing


@runtime_checkable
class HasBaseScriptArgs(Protocol):
    """
    Protocol of generic ScriptArgs that contains the BaseScriptArgs
    """

    base_script_args: BaseScriptArgs


class Script[T_Dataclass: IsDataclass]:
    """
    The core Script class to launch scripts.
    Takes care of computing execution time, profiling, and parsing arguments.
    """

    start_time: float
    parser: Parsley[T_Dataclass]
    gui_args: dict[str, Any] | None
    profile: cProfile.Profile | None
    experiment_script_type_output_folder: path | None = None
    base_experiment_output_folder: path = "chipiron/scripts/"
    default_experiment_output_folder: path = "chipiron/scripts/default_output_folder"
    config_file_name: str | None

    def __init__(
        self,
        parser: Parsley[T_Dataclass],
        extra_args: dict[str, Any] | None = None,
        config_file_name: str | None = None,
    ) -> None:
        """
        Initializes the Script object.
        Starts the clock, the profiling, and parses arguments.

        Args:
            parser: An instance of MyParser used for parsing arguments.
            extra_args: Additional arguments to be passed to the parser.
        """
        self.start_time = time.time()  # start the clock
        self.parser = parser
        self.experiment_script_type_output_folder = None
        self.config_file_name = config_file_name
        self.extra_args = extra_args

        self.profile = None
        self.args: IsDataclass | None = None

    def initiate(
        self,
        experiment_output_folder: str | None = None,
    ) -> T_Dataclass:
        """
        Initiates the script by parsing arguments and converting them into a standardized dataclass.

        Args:
            args_dataclass_name: The type of the dataclass to convert the arguments into.
            experiment_output_folder: The base folder for experiment output. If None, uses the default value.

        Returns:
            The converted arguments as a dataclass.
        """

        # checking that the dataclass that will contain the script args contains BaseScriptArgs
        # assert issubclass(args_dataclass_name, HasBaseScriptArgs)

        if experiment_output_folder is not None:
            self.experiment_script_type_output_folder = experiment_output_folder
        else:
            self.experiment_script_type_output_folder = (
                self.default_experiment_output_folder
            )

        final_args: T_Dataclass = self.parser.parse_arguments(
            extra_args=self.extra_args, config_file_path=self.config_file_name
        )

        assert hasattr(final_args, "base_script_args")

        final_args.base_script_args.experiment_output_folder = os.path.join(
            self.experiment_script_type_output_folder,
            final_args.base_script_args.relative_script_instance_experiment_output_folder,
        )
        mkdir_if_not_existing(final_args.base_script_args.experiment_output_folder)
        mkdir_if_not_existing(
            os.path.join(
                final_args.base_script_args.experiment_output_folder,
                "inputs_and_parsing",
            )
        )

        self.parser.log_parser_info(
            final_args.base_script_args.experiment_output_folder
        )

        # activate profiling is if needed
        if final_args.base_script_args.profiling:
            self.profile = cProfile.Profile()
            self.profile.enable()

        self.args = final_args
        chipiron_logger.info(
            f"The args of the script are:\n{pprint.pformat(self.args)}"
        )

        return final_args

    def terminate(self) -> None:
        """
        Finishes the script by printing execution time and profiling information (if enabled).
        """
        chipiron_logger.info("terminate")
        if self.profile is not None:
            chipiron_logger.info(f"--- {time.time() - self.start_time} seconds ---")
            self.profile.disable()
            string_io = io.StringIO()
            sort_by = SortKey.CUMULATIVE
            stats = pstats.Stats(self.profile, stream=string_io).sort_stats(sort_by)
            stats.print_stats()
            chipiron_logger.info(string_io.getvalue())
            assert self.args is not None
            assert hasattr(self.args, "base_script_args")

            path_to_profiling_stats: path = os.path.join(
                self.args.base_script_args.experiment_output_folder,
                "profiling_output.stats",
            )
            path_to_profiling_stats_png: path = os.path.join(
                self.args.base_script_args.experiment_output_folder,
                "profiling_output.png",
            )
            stats.dump_stats(path_to_profiling_stats)
            os.popen(
                f"gprof2dot -f pstats {path_to_profiling_stats} | dot -Tpng -o {path_to_profiling_stats_png}"
            )

            end_time = time.time()

            chipiron_logger.info("The args of the script were:\n")
            pprint.pprint(self.args)
            chipiron_logger.info("Execution time", end_time - self.start_time)

    def run(self) -> None:
        """
        Runs the script.
        """

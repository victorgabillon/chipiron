"""Document the module contains the Script class which is responsible for launching scripts.

It handles computing execution time, profiling, and parsing arguments.
"""

import cProfile
import io
import os
import pstats
import time
from pstats import SortKey
from typing import Any, cast

from parsley import Parsley
from rich.pretty import pretty_repr

from chipiron.scripts.default_script_args import DataClassWithBaseScriptArgs
from chipiron.utils import MyPath
from chipiron.utils.dataclass import IsDataclass
from chipiron.utils.logger import chipiron_logger, set_chipiron_logger_level
from chipiron.utils.path_runtime import output_root_path_str
from chipiron.utils.small_tools import mkdir_if_not_existing, resolve_package_path


class Script[DataclassT: DataClassWithBaseScriptArgs = DataClassWithBaseScriptArgs]:
    """The core Script class to launch scripts.

    Takes care of computing execution time, profiling, and parsing arguments.
    """

    start_time: float
    parser: Parsley[DataclassT]
    gui_args: dict[str, Any] | None
    profile: cProfile.Profile | None
    experiment_script_type_output_folder: MyPath | None = None
    base_experiment_output_folder: MyPath = output_root_path_str()
    default_experiment_output_folder: MyPath = os.path.join(
        base_experiment_output_folder, "default_output_folder"
    )
    config_file_name: str | None

    def __init__(
        self,
        parser: Parsley[DataclassT],
        extra_args: IsDataclass | None = None,
        config_file_name: str | None = None,
    ) -> None:
        """Initialize the Script object.

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
    ) -> DataclassT:
        """Initiate the script by parsing arguments and converting them into a standardized dataclass.

        Args:
            experiment_output_folder: The base folder for experiment output. If None, uses the default value.

        Returns:
            The converted arguments as a dataclass.

        """
        if experiment_output_folder is not None:
            self.experiment_script_type_output_folder = experiment_output_folder
        else:
            self.experiment_script_type_output_folder = (
                self.default_experiment_output_folder
            )
        assert self.experiment_script_type_output_folder is not None
        mkdir_if_not_existing(self.experiment_script_type_output_folder)

        final_args: DataclassT = self.parser.parse_arguments(
            extra_args=self.extra_args, config_file_path=resolve_package_path(self.config_file_name)
        )

        final_args_with_base = final_args

        set_chipiron_logger_level(
            level=final_args_with_base.base_script_args.logging_levels.chipiron
        )

        # Ensure paths are not None before using them
        assert self.experiment_script_type_output_folder is not None
        assert (
            final_args_with_base.base_script_args.relative_script_instance_experiment_output_folder
            is not None
        )

        final_args_with_base.base_script_args.experiment_output_folder = os.path.join(
            self.experiment_script_type_output_folder,
            final_args_with_base.base_script_args.relative_script_instance_experiment_output_folder,
        )

        assert (
            final_args_with_base.base_script_args.experiment_output_folder is not None
        )
        mkdir_if_not_existing(
            final_args_with_base.base_script_args.experiment_output_folder
        )
        mkdir_if_not_existing(
            os.path.join(
                final_args_with_base.base_script_args.experiment_output_folder,
                "inputs_and_parsing",
            )
        )

        self.parser.log_parser_info(
            str(final_args_with_base.base_script_args.experiment_output_folder)
        )

        # activate profiling is if needed
        if final_args_with_base.base_script_args.profiling:
            self.profile = cProfile.Profile()
            self.profile.enable()

        self.args = final_args
        chipiron_logger.info("The args of the script are:\n%s", pretty_repr(self.args))

        return final_args

    def terminate(self) -> None:
        """Finishes the script by printing execution time and profiling information (if enabled)."""
        chipiron_logger.info("terminate")
        if self.profile is not None:
            chipiron_logger.info("--- %s seconds ---", time.time() - self.start_time)
            self.profile.disable()
            string_io = io.StringIO()
            sort_by = SortKey.CUMULATIVE
            stats = pstats.Stats(self.profile, stream=string_io).sort_stats(sort_by)
            stats.print_stats()
            chipiron_logger.info(string_io.getvalue())
            assert self.args is not None
            assert hasattr(self.args, "base_script_args")

            # Type assertion for self.args as well
            args_with_base = cast("DataClassWithBaseScriptArgs", self.args)

            # Ensure experiment_output_folder is not None
            assert args_with_base.base_script_args.experiment_output_folder is not None

            path_to_profiling_stats: MyPath = os.path.join(
                args_with_base.base_script_args.experiment_output_folder,
                "profiling_output.stats",
            )
            path_to_profiling_stats_png: MyPath = os.path.join(
                args_with_base.base_script_args.experiment_output_folder,
                "profiling_output.png",
            )
            stats.dump_stats(path_to_profiling_stats)
            os.popen(
                f"gprof2dot -f pstats {path_to_profiling_stats} | dot -Tpng -o {path_to_profiling_stats_png}"
            )

            end_time = time.time()

            chipiron_logger.info(
                "The args of the script were:\n%s", pretty_repr(self.args)
            )
            chipiron_logger.info("Execution time: %s", end_time - self.start_time)

    def run(self) -> None:
        """Run the script."""


# Type alias to handle Script generic variance issues
type AnyScript = Script[Any]

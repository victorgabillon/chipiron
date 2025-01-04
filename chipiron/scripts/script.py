"""
This module contains the Script class which is responsible for launching scripts.
It handles computing execution time, profiling, and parsing arguments.
"""

import cProfile
import io
import os.path
import pprint
import pstats
import time
from enum import Enum
from pstats import SortKey
from typing import Any
from typing import Protocol, runtime_checkable

import dacite

from chipiron.scripts.parsers.parser import MyParser
from chipiron.scripts.script_args import BaseScriptArgs
from chipiron.utils.dataclass import IsDataclass
from chipiron.utils.small_tools import mkdir


@runtime_checkable
class HasBaseScriptArgs(Protocol):
    """
    Protocol of generic ScriptArgs that contains the BaseScriptArgs
    """
    base_script_args: BaseScriptArgs


class Script:
    """
    The core Script class to launch scripts.
    Takes care of computing execution time, profiling, and parsing arguments.
    """

    start_time: float
    parser: MyParser
    gui_args: dict[str, Any] | None
    profile: cProfile.Profile | None
    base_experiment_output_folder: str = 'chipiron/scripts'

    def __init__(
            self,
            parser: MyParser,
            extra_args: dict[str, Any] | None = None
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
        self.experiment_output_folder = None
        self.extra_args = extra_args
        self.profile = None
        self.args: IsDataclass | None = None

    def initiate[_T_co:IsDataclass](
            self,
            args_dataclass_name: type[_T_co],
            base_experiment_output_folder: str | None = None
    ) -> _T_co:
        """
        Initiates the script by parsing arguments and converting them into a standardized dataclass.

        Args:
            args_dataclass_name: The type of the dataclass to convert the arguments into.
            base_experiment_output_folder: The base folder for experiment output. If None, uses the default value.

        Returns:
            The converted arguments as a dataclass.
        """

        # checking that the dataclass that will contain the script args contains BaseScriptArgs
        # assert issubclass(args_dataclass_name, HasBaseScriptArgs)

        if base_experiment_output_folder is None:
            base_experiment_output_folder = self.base_experiment_output_folder

        # parse the arguments
        args_dict: dict[str, Any] = self.parser.parse_arguments(
            extra_args=self.extra_args
        )

        # Converting the args in the standardized dataclass
        final_args: _T_co = dacite.from_dict(
            data_class=args_dataclass_name,
            data=args_dict,
            config=dacite.Config(cast=[Enum])
        )
        assert hasattr(final_args, 'base_script_args')

        final_args.base_script_args.experiment_output_folder = os.path.join(
            base_experiment_output_folder,
            final_args.base_script_args.experiment_output_folder
        )
        mkdir(final_args.base_script_args.experiment_output_folder)
        mkdir(os.path.join(final_args.base_script_args.experiment_output_folder, 'inputs_and_parsing'))

        self.parser.log_parser_info(final_args.base_script_args.experiment_output_folder)

        # activate profiling is if needed
        if final_args.base_script_args.profiling:
            self.profile = cProfile.Profile()
            self.profile.enable()

        self.args = final_args
        print('the args of the script are:\n')
        pprint.pprint(self.args)
        return final_args

    def terminate(self) -> None:
        """
        Finishes the script by printing execution time and profiling information (if enabled).
        """
        print('terminate')
        if self.profile is not None:
            print(f'--- {time.time() - self.start_time} seconds ---')
            self.profile.disable()
            string_io = io.StringIO()
            sort_by = SortKey.CUMULATIVE
            stats = pstats.Stats(self.profile, stream=string_io).sort_stats(sort_by)
            stats.print_stats()
            print(string_io.getvalue())

        end_time = time.time()

        print('the args of the script were:\n')
        pprint.pprint(self.args)
        print('execution time', end_time - self.start_time)

    def run(self) -> None:
        """
        Runs the script.
        """

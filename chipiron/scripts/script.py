"""
This module contains the Script class which is responsible for launching scripts.
It handles computing execution time, profiling, and parsing arguments.
"""

import cProfile
import io
import os.path
import pstats
import time
from dataclasses import dataclass
from pstats import SortKey
from typing import Any
from typing import TypeVar

import dacite

from chipiron.scripts.parsers.parser import MyParser
from chipiron.utils.is_dataclass import IsDataclass
from chipiron.utils.small_tools import mkdir


@dataclass
class ScriptArgs:
    """
    Dataclass representing the arguments for the Script class.
    """

    profiling: bool = False
    testing: bool = False


_T_co = TypeVar("_T_co", covariant=True, bound=IsDataclass)


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
        self.start_time = time.time()
        self.parser = parser
        self.experiment_output_folder = None
        self.extra_args = extra_args
        self.profile = None

    def initiate(
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
        if base_experiment_output_folder is None:
            base_experiment_output_folder = self.base_experiment_output_folder

        args_dict: dict[str, Any] = self.parser.parse_arguments(
            base_experiment_output_folder=base_experiment_output_folder,
            extra_args=self.extra_args
        )

        args: ScriptArgs = dacite.from_dict(
            data_class=ScriptArgs,
            data=args_dict
        )

        mkdir(args_dict['experiment_output_folder'])
        mkdir(os.path.join(args_dict['experiment_output_folder'], 'inputs_and_parsing'))

        self.parser.log_parser_info(args_dict['experiment_output_folder'])

        if args.profiling:
            self.profile = cProfile.Profile()
            self.profile.enable()

        final_args: _T_co = dacite.from_dict(
            data_class=args_dataclass_name,
            data=args_dict
        )
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
        print('execution time', end_time - self.start_time)

    def run(self) -> None:
        """ 
        Runs the script.
        """

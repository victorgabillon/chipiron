"""
The base script
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
    # whether the script is profiling computation usage
    profiling: bool = False

    # whether the script is testing the code (using pytest for instance)
    testing: bool = False


_T_co = TypeVar("_T_co", covariant=True, bound=IsDataclass)


class Script:
    """
    The core Script class to launch scripts.
    Takes care of computing execution time, profiling, ang parsing arguments
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
        Building the Script object, starts the clock,
        the profiling and parse arguments and deals with global variables
        """
        # start the clock
        self.start_time = time.time()
        self.parser = parser
        self.experiment_output_folder = None
        self.extra_args = extra_args
        self.profile = None

    def initiate(
            self,
            args_dataclass_name: type[_T_co],
            base_experiment_output_folder=None

    ) -> _T_co:

        if base_experiment_output_folder is None:
            base_experiment_output_folder = self.base_experiment_output_folder

        # parse the arguments
        args_dict: dict[str, Any] = self.parser.parse_arguments(
            base_experiment_output_folder=base_experiment_output_folder,
            extra_args=self.extra_args
        )

        # Converting the args in the standardized dataclass
        args: ScriptArgs = dacite.from_dict(
            data_class=ScriptArgs,
            data=args_dict
        )

        mkdir(args_dict['experiment_output_folder'])
        mkdir(os.path.join(args_dict['experiment_output_folder'], 'inputs_and_parsing'))

        self.parser.log_parser_info(args_dict['experiment_output_folder'])

        # activate profiling is if needed
        if args.profiling:
            self.profile = cProfile.Profile()
            self.profile.enable()

        # Converting the args in the standardized dataclass
        final_args: _T_co = dacite.from_dict(
            data_class=args_dataclass_name,
            data=args_dict
        )
        return final_args

    def terminate(self) -> None:
        """
        Finishing the script. Profiling or timing.
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
        """ Running the script"""

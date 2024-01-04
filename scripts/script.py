"""
The base script
"""

import cProfile
import pstats
import io
from pstats import SortKey
import time
from chipiron.utils.small_tools import mkdir
from scripts.parsers.parser import MyParser
from dataclasses import dataclass
import dacite


@dataclass
class ScriptArgs:
    profiling: bool = False


class Script:
    """
    The core Script class to launch scripts.
    Takes care of computing execution time, profiling, ang parsing arguments
    """

    start_time: float
    parser: MyParser
    gui_args: dict | None
    base_experiment_output_folder: str = 'scripts'

    def __init__(
            self,
            parser: MyParser,
            gui_args: dict | None = None
    ) -> None:
        """
        Building the Script object, starts the clock,
        the profiling and parse arguments and deals with global variables
        """
        # start the clock
        self.start_time = time.time()
        self.parser = parser
        self.experiment_output_folder = None
        self.gui_args = gui_args
        self.profile = None

    def initiate(self,
                 base_experiment_output_folder=None) -> dict:

        if base_experiment_output_folder is None:
            base_experiment_output_folder = self.base_experiment_output_folder

        # parse the arguments
        args_dict: dict = self.parser.parse_arguments(base_experiment_output_folder=base_experiment_output_folder,
                                                      gui_args=self.gui_args)

        # Converting the args in the standardized dataclass
        args: ScriptArgs = dacite.from_dict(data_class=ScriptArgs,
                                            data=args_dict)

        mkdir(args_dict['experiment_output_folder'])
        self.parser.log_parser_info(args_dict['experiment_output_folder'])

        # activate profiling is if needed
        if args.profiling:
            self.profile = cProfile.Profile()
            self.profile.enable()

        return args_dict

    def terminate(self) -> None:
        """
        Finishing the script. Profiling or timing.
        """
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

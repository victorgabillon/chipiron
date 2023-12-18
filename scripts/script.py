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


class Script:
    """
    The core Script class to launch scripts.
    Takes care of computing execution time, profiling, ang parsing arguments
    """

    default_param_dict: dict = {'profiling': False}
    base_experiment_output_folder: str = 'scripts/'
    start_time: float
    parser: MyParser
    gui_args: dict | None

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

    def parse(self,
              default_param_dict: dict
              ) -> dict:

        # parse the arguments
        args = self.parser.parse_arguments(default_param_dict=default_param_dict | self.default_param_dict,
                                           base_experiment_output_folder=self.base_experiment_output_folder,
                                           gui_args=self.gui_args)

        # activate profiling is if needed
        if args['profiling']:
            self.profile = cProfile.Profile()
            self.profile.enable()

        return args

    def initiate(self,
                 default_param_dict: dict
                 ) -> dict:
        args: dict = self.parse(default_param_dict=default_param_dict)
        mkdir(args['experiment_output_folder'])
        self.parser.log_parser_info(args['experiment_output_folder'])
        return args

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

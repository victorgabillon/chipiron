"""
The base script
"""

import cProfile
import pstats
import io
from pstats import SortKey
import time
from datetime import datetime
from chipiron.extra_tools.small_tools import mkdir
from scripts.parsers.parser import create_parser


class Script:
    """
    The core Script class to launch scripts.
    Takes care of computing execution time, profiling, ang parsing arguments
    """

    default_param_dict = {'profiling': False}
    base_experiment_output_folder = 'scripts/'

    def __init__(self, gui_args=None):
        """
        Building the Script object, starts the clock,
        the profiling and parse arguments and deals with global variables
        """
        # start the clock
        self.start_time = time.time()

        # parse the arguments
        parser = create_parser(default_param_dict=self.default_param_dict)
        self.args = parser.parse_arguments(gui_args)
        self.experiment_output_folder = None
        self.set_experiment_output_folder()
        mkdir(self.experiment_output_folder)
        parser.log_parser_info(self.experiment_output_folder)

        # activate profiling is if needed
        if self.args['profiling']:
            self.profile = cProfile.Profile()
            self.profile.enable()

    def set_experiment_output_folder(self) -> None:
        """
            computes the path to the experiment output folder
        """
        if 'output_folder' not in self.args:
            now = datetime.now()  # current date and time
            self.experiment_output_folder = self.base_experiment_output_folder + now.strftime(
                "%A-%m-%d-%Y--%H:%M:%S:%f")
        else:
            self.experiment_output_folder = self.base_experiment_output_folder + self.args['output_folder']

    def terminate(self) -> None:
        """
        Finishing the script. Profiling or timing.
        """
        if self.args['profiling']:
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

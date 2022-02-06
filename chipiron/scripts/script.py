import cProfile, pstats, io
from pstats import SortKey
import time
from scripts.parsers.parser import create_parser
from datetime import datetime
from src.extra_tools.small_tools import mkdir


class Script:
    """
    The core Script class to launch scripts.
    Takes care of computing execution time, profiling, ang parsing arguments
    """

    default_param_dict = {'profiling': False}
    base_experiment_output_folder = 'chipiron/scripts/'

    def __init__(self):
        """
        Building the Script object, starts the clock, the profiling and parse arguments and deals with global variables
        """
        # start the clock
        self.start_time = time.time()

        # parse the arguments
        parser = create_parser(default_param_dict=self.default_param_dict)
        self.args = parser.parse_arguments()
        self.experiment_output_folder = None
        self.set_experiment_output_folder()
        mkdir(self.experiment_output_folder)
        parser.log_parser_info(self.experiment_output_folder)

        # init global variables
        # global_variables.init(self.args)

        # activate profiling is if needed
        if self.args['profiling']:
            self.pr = cProfile.Profile()
            self.pr.enable()

    def set_experiment_output_folder(self):
        if 'output_folder' not in self.args:
            now = datetime.now()  # current date and time
            self.experiment_output_folder = self.base_experiment_output_folder + now.strftime(
                "%A-%m-%d-%Y--%H:%M:%S:%f")
        else:
            self.experiment_output_folder = self.base_experiment_output_folder + self.args['output_folder']

    def terminate(self):
        if self.args['profiling']:
            print("--- %s seconds ---" % (time.time() - self.start_time))
            self.pr.disable()
            s = io.StringIO()
            sortby = SortKey.CUMULATIVE
            ps = pstats.Stats(self.pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print(s.getvalue())

        end_time = time.time()
        print('execution time', end_time - self.start_time)

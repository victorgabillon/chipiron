import global_variables
import cProfile, pstats, io
from pstats import SortKey
import time


class Script:

    def __init__(self):
        self.start_time = time.time()

        if global_variables.profiling_bool:
            self.pr = cProfile.Profile()
            self.pr.enable()

    def terminate(self):
        if global_variables.profiling_bool:
            print("--- %s seconds ---" % (time.time() - self.start_time))
            self.pr.disable()
            s = io.StringIO()
            sortby = SortKey.CUMULATIVE
            ps = pstats.Stats(self.pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print(s.getvalue())

        end_time = time.time()
        print('execution time', end_time - self.start_time)





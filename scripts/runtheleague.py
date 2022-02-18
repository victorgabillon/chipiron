from scripts.script import Script
from src.league import League
import os
import pickle
from src.players.boardevaluators.table_base.syzygy import SyzygyTable


class RuntheLeagueScript(Script):

    default_param_dict = Script.default_param_dict | \
                         {'config_file_name': None,
                          'deterministic_behavior': False,
                          'deterministic_mode': 'SEED_FIXED_EVERY_MOVE',
                          'seed_fixing_type': 'FIX_SEED_WITH_CONSTANT'
                          }
    base_experiment_output_folder = Script.base_experiment_output_folder + '/league_outputs/'

    def __init__(self):
        super().__init__()
        print(os.getcwd())
        self.folder_league = 'chipiron/runs/league/league_10000'

        try:
            with (open(self.folder_league + '/players.pickle', "rb")) as openfile:
                #TODO not use pickle make somehting human readbale and moidifiable
                self.league = pickle.load(openfile)
        except:
            self.league = League(self.folder_league)

        self.league.print_info()

    def run(self):
        syzygy_table = SyzygyTable('')
        while True:
            self.league.run(syzygy_table)
            # save
            pickle.dump(self.league, open(self.folder_league + '/players.pickle', "wb"))

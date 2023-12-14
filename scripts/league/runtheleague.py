"""
RunTheLeagueScript
"""
from scripts.script import Script
from chipiron.league.league import League
import os
import pickle
from chipiron.players.boardevaluators.table_base.syzygy import SyzygyTable
import random

class RunTheLeagueScript(Script):
    """
    Running a league playing games between
    ers in the league and computing ELOs
    """
    default_param_dict: dict = Script.default_param_dict | \
                               {'config_file_name': None,
                                'deterministic_behavior': False,
                                'deterministic_mode': 'SEED_FIXED_EVERY_MOVE',
                                'seed_fixing_type': 'FIX_SEED_WITH_CONSTANT'
                                }
    base_experiment_output_folder: str = os.path.join(Script.base_experiment_output_folder, 'league/outputs/')

    folder_league: str = os.path.join(Script.base_experiment_output_folder, 'league/league_data/league_10_000')

    def __init__(
            self
    ) -> None:
        super().__init__()

        try:
            with (open(self.folder_league + '/players.pickle', "rb")) as openfile:
                # TODO not use pickle make somehting human readbale and moidifiable
                self.league = pickle.load(openfile)
        except:
            random_generator: random.Random = random.Random(0)
            self.league: League = League(foldername=self.folder_league,
                                         random_generator=random_generator)

        self.league.print_info()

    def run(self):
        while True:
            self.league.run()
            # save
            pickle.dump(self.league, open(self.folder_league + '/players.pickle', "wb"))

"""
RunTheLeagueScript
"""
from scripts.script import Script
from chipiron.league.league import League
import os
import pickle
import random


class RunTheLeagueScript:
    """
    Running a league playing games between
    ers in the league and computing ELOs
    """
    default_param_dict: dict = {'config_file_name': None,
                                }
    base_experiment_output_folder: str = os.path.join(Script.base_experiment_output_folder, 'league/outputs/')

    folder_league: str = os.path.join(Script.base_experiment_output_folder, 'league/league_data/league_10_001')
    base_script: Script

    def __init__(
            self,
            base_script: Script
    ) -> None:

        self.base_script = base_script

        try:
            with (open(self.folder_league + '/players.pickle', "rb")) as openfile:
                # TODO not use pickle make somehting human readbale and moidifiable
                self.league = pickle.load(openfile)
            print('Previous league loaded successfully')
        except:
            print('Creation of a new league')
            new_random = random.Random(x=None)
            seed_run: int = new_random.randrange(0, 10 ** 10)
            self.league: League = League(
                folder_league=self.folder_league,
                seed=seed_run
            )

        self.league.print_info()

    def run(self):
        print('run the league')
        while True:
            self.league.run()
            # save
            pickle.dump(self.league, open(self.folder_league + '/players.pickle', "wb"))

    def terminate(self) -> None:
        self.base_script.terminate()

from scripts.script import Script
from src.league.league import League
import os


class RuntheLeagueScript(Script):

    def __init__(self):
        super().__init__()
        print(os.getcwd())
        self.league = League('data/league/league_10000')

    def run(self):
        self.league.run()

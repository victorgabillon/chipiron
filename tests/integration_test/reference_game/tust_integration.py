"""Module for tust integration."""

import os
import pickle

import dacite
import yaml

from chipiron.games.match.match_args import MatchArgs
from chipiron.games.match.match_factories import create_match_manager_from_args
from chipiron.games.match.match_manager import MatchManager
from chipiron.games.match.match_results import MatchReport
from chipiron.utils import path

os.chdir("../../")
print(os.getcwd())

configs = ["Uniform"]

config: str
for config in configs:
    with open(
        os.path.join(
            "data/reference_games",
            config,
            "inputs_and_parsing/one_match_script_merge.yaml",
        )
    ) as file:
        args_dict = yaml.safe_load(file)

    # Converting the args in the standardized dataclass
    args: MatchArgs = dacite.from_dict(data_class=MatchArgs, data=args_dict)

    match_manager: MatchManager = create_match_manager_from_args(args=args)
    match_report: MatchReport = match_manager.play_one_match()

    path_file: path = os.path.join("data/reference_games", config, "match_report.obj")
    print("path_file", path_file)

    with open(path_file, "rb") as file:
        match_report_reference: MatchReport = pickle.load(file)

    # checking if two matches launched with the same fixed seed returns the same game moves.
    print(
        "match_report.match_move_history, match_report_base.match_move_history",
        match_report.match_move_history,
        match_report_reference.match_move_history,
    )
    assert match_report.match_move_history == match_report_reference.match_move_history

print(match_report)
print("ALL OK for REFERENCE GAME")

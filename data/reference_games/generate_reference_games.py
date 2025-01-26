import os
from dataclasses import asdict

import yaml

import chipiron as ch
from chipiron.games.match.match_args import MatchArgs
from chipiron.games.match.match_factories import create_match_manager_from_args
from chipiron.games.match.match_results import MatchReport
from chipiron.utils import mkdir_if_not_existing, path

algos: list[str] = ["Uniform"]

algo: str
for algo in algos:
    path_folder: path = os.path.join("data/reference_games", algo)
    mkdir_if_not_existing(folder_path=path_folder)
    args: MatchArgs = MatchArgs()
    args.seed = 11
    args.file_name_player_one = f"players_for_test_purposes/{algo}.yaml"
    args.file_name_player_two = "players_for_test_purposes/Uniform.yaml"
    args.file_name_match_setting = "setting_tron.yaml"
    args.experiment_output_folder = path_folder

    mkdir_if_not_existing(folder_path=os.path.join(path_folder, "inputs_and_parsing"))

    # saving the arguments of the script
    with open(
        os.path.join(
            args.experiment_output_folder,
            "inputs_and_parsing/one_match_script_merge.yaml",
        ),
        "w",
    ) as one_match_script:
        yaml.dump(asdict(args), one_match_script, default_flow_style=False)

    match_manager: ch.game.MatchManager = create_match_manager_from_args(args=args)
    match_report_base: MatchReport = match_manager.play_one_match()

    # with open(os.path.join('data/reference_games', algo, 'inputs_and_parsing/one_match_script_merge.yaml'),
    #          'rb') as file:
    #    pickle.dump(match_report_base, file)

# import os
# os.system('git add data/reference_games/\*')

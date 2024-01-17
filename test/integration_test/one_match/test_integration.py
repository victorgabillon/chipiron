import scripts
import os

os.chdir('../../')
print(os.getcwd())
configs = [

    # checking gui
    {'seed': 11, 'gui': True, 'file_name_player_one': 'Sequool.yaml', 'file_name_player_two': 'Random.yaml',
     'file_name_match_setting': 'setting_jime.yaml', 'profiling': False},

    # random Player first to have a fast game
    {'seed': 11, 'gui': False, 'file_name_player_one': 'Sequool.yaml', 'file_name_player_two': 'Random.yaml',
     'file_name_match_setting': 'setting_jime.yaml', 'profiling': False},

    {'seed': 11, 'gui': False, 'file_name_player_one': 'Sequool.yaml', 'file_name_player_two': 'RecurZipfBase3.yaml',
     'file_name_match_setting': 'setting_jime.yaml', 'profiling': False},
    {'seed': 11, 'gui': False, 'file_name_player_one': 'RecurZipfBase4.yaml',
     'file_name_player_two': 'RecurZipfBase3.yaml', 'file_name_match_setting': 'setting_jime.yaml', 'profiling': False},
    {'seed': 11, 'gui': False, 'file_name_player_one': 'Uniform.yaml', 'file_name_player_two': 'RecurZipfBase3.yaml',
     'file_name_match_setting': 'setting_jime.yaml', 'profiling': False},
    {'seed': 11, 'gui': False, 'file_name_player_one': 'RecurZipfBase3.yaml',
     'file_name_player_two': 'RecurZipfBase3.yaml', 'file_name_match_setting': 'setting_jime.yaml', 'profiling': False},

    # checking profiling
    {'seed': 11, 'gui': False, 'file_name_player_one': 'Sequool.yaml', 'file_name_player_two': 'Random.yaml',
     'file_name_match_setting': 'setting_jime.yaml', 'profiling': True},

    # checking another seed haha
    {'seed': 12, 'gui': False, 'file_name_player_one': 'Sequool.yaml', 'file_name_player_two': 'Random.yaml',
     'file_name_match_setting': 'setting_jime.yaml', 'profiling': False},

    # need a check with two games
    {'seed': 11, 'gui': False, 'file_name_player_one': 'Sequool.yaml', 'file_name_player_two': 'Random.yaml',
     'file_name_match_setting': 'setting_tron.yaml', 'profiling': False},

]

for config in configs:
    script_object: scripts.Script = scripts.create_script(
        script_type=scripts.ScriptType.OneMatch,
        extra_args=config
    )

    # run the script
    script_object.run()

    # terminate the script
    script_object.terminate()

# test randomness
import chipiron as ch

from chipiron.players.factory import PlayerArgs
from chipiron.players.utils import fetch_two_players_args_convert_and_save
from chipiron.games.match.utils import fetch_match_games_args_convert_and_save
import chipiron.games.match as match
import chipiron.games.game as game
from chipiron.games.match.observable_match_result import MatchReport
from scripts.one_match.one_match import OneMatchScriptArgs

args = OneMatchScriptArgs()
args.seed = 0
args.file_name_player_one = 'players_for_test_purposes/RecurZipfSequool.yaml'
args.file_name_player_two = 'players_for_test_purposes/Sequool.yaml'
args.file_name_match_setting = 'setting_tron.yaml'

player_one_args: PlayerArgs
player_two_args: PlayerArgs
player_one_args, player_two_args = fetch_two_players_args_convert_and_save(
    file_name_player_one=args.file_name_player_one,
    file_name_player_two=args.file_name_player_two,
    modification_player_one=args.player_one,
    modification_player_two=args.player_two,
    experiment_output_folder=args.experiment_output_folder
)

# Recovering args from yaml file for match and game and merging with extra args and converting
# to standardized dataclass
match_args: match.MatchArgs
game_args: game.GameArgs
match_args, game_args = fetch_match_games_args_convert_and_save(
    profiling=args.profiling,
    file_name_match_setting=args.file_name_match_setting,
    modification=args.match,
    experiment_output_folder=args.experiment_output_folder
)

# taking care of random
ch.set_seeds(seed=args.seed)

print('self.args.experiment_output_folder', args.experiment_output_folder)
match_manager: ch.game.MatchManager = match.create_match_manager(
    args_match=match_args,
    args_player_one=player_one_args,
    args_player_two=player_two_args,
    output_folder_path=args.experiment_output_folder,
    seed=args.seed,
    args_game=game_args,
    gui=args.gui
)
match_report_base: MatchReport = match_manager.play_one_match()

test_passed: bool = True
number_test: int = 1

for ind in range(number_test):
    match_manager: ch.game.MatchManager = match.create_match_manager(
        args_match=match_args,
        args_player_one=player_one_args,
        args_player_two=player_two_args,
        output_folder_path=args.experiment_output_folder,
        seed=args.seed,
        args_game=game_args,
        gui=args.gui
    )
    match_report: MatchReport = match_manager.play_one_match()

    # checking if two matches launched with the same fixed seed returns the same game moves.
    print('match_report.match_move_history, match_report_base.match_move_history',
          match_report.match_move_history, match_report_base.match_move_history)
    assert (match_report.match_move_history == match_report_base.match_move_history)

print(match_report)
print('ALL OK for ONE MATCH')

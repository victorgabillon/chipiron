import scripts

configs = [

    # checking gui
    {'seed': 11, 'gui': True, 'file_name_player_one': 'players_for_test_purposes/Sequool.yaml',
     'file_name_player_two': 'Random.yaml',
     'file_name_match_setting': 'setting_jime.yaml', 'profiling': False},


    # random Player first to have a fast game
    {'seed': 11, 'gui': False, 'file_name_player_one': 'players_for_test_purposes/Sequool.yaml',
     'file_name_player_two': 'Random.yaml',
     'file_name_match_setting': 'setting_jime.yaml', 'profiling': False},

    {'seed': 11, 'gui': False, 'file_name_player_one': 'Stockfish.yaml',
     'file_name_player_two': 'Random.yaml',
     'file_name_match_setting': 'setting_jime.yaml', 'profiling': False},

    {'seed': 11, 'gui': False, 'file_name_player_one': 'players_for_test_purposes/Sequool.yaml',
     'file_name_player_two': 'players_for_test_purposes/RecurZipfSequool.yaml',
     'file_name_match_setting': 'setting_jime.yaml', 'profiling': False},
    {'seed': 11, 'gui': False, 'file_name_player_one': 'players_for_test_purposes/Sequool.yaml',
     'file_name_player_two': 'players_for_test_purposes/RecurZipfBase3.yaml',
     'file_name_match_setting': 'setting_jime.yaml', 'profiling': False},
    {'seed': 11, 'gui': False, 'file_name_player_one': 'players_for_test_purposes/RecurZipfBase4.yaml',
     'file_name_player_two': 'players_for_test_purposes/RecurZipfBase3.yaml',
     'file_name_match_setting': 'setting_jime.yaml', 'profiling': False},
    {'seed': 11, 'gui': False, 'file_name_player_one': 'players_for_test_purposes/Uniform.yaml',
     'file_name_player_two': 'players_for_test_purposes/RecurZipfBase3.yaml',
     'file_name_match_setting': 'setting_jime.yaml', 'profiling': False},
    {'seed': 11, 'gui': False, 'file_name_player_one': 'players_for_test_purposes/RecurZipfBase3.yaml',
     'file_name_player_two': 'players_for_test_purposes/RecurZipfBase3.yaml',
     'file_name_match_setting': 'setting_jime.yaml', 'profiling': False},

    # checking profiling
    {'seed': 11, 'gui': False, 'file_name_player_one': 'players_for_test_purposes/Sequool.yaml',
     'file_name_player_two': 'Random.yaml',
     'file_name_match_setting': 'setting_jime.yaml', 'profiling': True},

    # checking another seed haha
    {'seed': 12, 'gui': False, 'file_name_player_one': 'players_for_test_purposes/Sequool.yaml',
     'file_name_player_two': 'Random.yaml',
     'file_name_match_setting': 'setting_jime.yaml', 'profiling': False},

    # need a check with two games
    {'seed': 11, 'gui': False, 'file_name_player_one': 'players_for_test_purposes/Sequool.yaml',
     'file_name_player_two': 'Random.yaml',
     'file_name_match_setting': 'setting_tron.yaml', 'profiling': False},


]

for config in configs:
    print(f'Running the SCRIPT with config {config}')
    script_object: scripts.Script = scripts.create_script(
        script_type=scripts.ScriptType.OneMatch,
        extra_args=config
    )

    # run the script
    script_object.run()

    # terminate the script
    script_object.terminate()

print('test randomness')
# test randomness
import chipiron as ch
from chipiron.games.match.match_factories import create_match_manager_from_args
from chipiron.games.match.match_results import MatchReport
from chipiron.games.match.match_args import MatchArgs

args: MatchArgs = MatchArgs()
args.seed = 0
args.file_name_player_one = 'players_for_test_purposes/RecurZipfSequool.yaml'
args.file_name_player_two = 'players_for_test_purposes/Sequool.yaml'
args.file_name_player_two = 'Random.yaml'
args.file_name_match_setting = 'setting_tron.yaml'

match_manager: ch.game.MatchManager = create_match_manager_from_args(args=args)
match_report_base: MatchReport = match_manager.play_one_match()

test_passed: bool = True
number_test: int = 1

for ind in range(number_test):
    match_manager: ch.game.MatchManager = create_match_manager_from_args(args=args)
    match_report: MatchReport = match_manager.play_one_match()

    # checking if two matches launched with the same fixed seed returns the same game moves.
    print('match_report.match_move_history, match_report_base.match_move_history',
          match_report.match_move_history, match_report_base.match_move_history)
    assert (match_report.match_move_history == match_report_base.match_move_history)

print(match_report)
print('ALL OK for ONE MATCH')

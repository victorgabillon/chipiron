import sys
import time

import chipiron.scripts as scripts

# we need to not use multiprocessing to be able to use pytest therefore use setting_cubo  and not setting_jime

start_time = time.time()
configs_base = [

    # random Player first to have a fast game
    {
        'seed': 11, 'gui': False,
        'file_name_player_one': 'players_for_test_purposes/Sequool.yaml',
        'file_name_player_two': 'Random.yaml',
        'file_name_match_setting': 'setting_cubo.yaml',
        'profiling': False,
        'testing': True
    },

    {
        'seed': 11,
        'gui': False,
        'file_name_player_one':
            'players_for_test_purposes/Sequool.yaml',
        'file_name_player_two': 'players_for_test_purposes/RecurZipfSequool.yaml',
        'file_name_match_setting': 'setting_cubo.yaml',
        'profiling': False,
        'testing': True
    },

    {
        'seed': 11, 'gui': False,
        'file_name_player_one': 'players_for_test_purposes/Sequool.yaml',
        'file_name_player_two': 'players_for_test_purposes/RecurZipfBase3.yaml',
        'file_name_match_setting': 'setting_cubo.yaml',
        'profiling': False,
        'testing': True
    },

    {
        'seed': 11,
        'gui': False,
        'file_name_player_one': 'players_for_test_purposes/RecurZipfBase4.yaml',
        'file_name_player_two': 'players_for_test_purposes/RecurZipfBase3.yaml',
        'file_name_match_setting': 'setting_cubo.yaml',
        'profiling': False,
        'testing': True
    },

    {
        'seed': 11,
        'gui': False,
        'file_name_player_one': 'players_for_test_purposes/Uniform.yaml',
        'file_name_player_two': 'players_for_test_purposes/RecurZipfBase3.yaml',
        'file_name_match_setting': 'setting_cubo.yaml',
        'profiling': False,
        'testing': True
    },

    {
        'seed': 11,
        'gui': False,
        'file_name_player_one': 'players_for_test_purposes/RecurZipfBase3.yaml',
        'file_name_player_two': 'players_for_test_purposes/RecurZipfBase3.yaml',
        'file_name_match_setting': 'setting_cubo.yaml',
        'profiling': False,
        'testing': True
    },

    # checking profiling
    {
        'seed': 11,
        'gui': False,
        'file_name_player_one': 'players_for_test_purposes/Sequool.yaml',
        'file_name_player_two': 'Random.yaml',
        'file_name_match_setting': 'setting_cubo.yaml',
        'profiling': True,
        'testing': True
    },

    # checking another seed haha
    {
        'seed': 12,
        'gui': False,
        'file_name_player_one': 'players_for_test_purposes/Sequool.yaml',
        'file_name_player_two': 'Random.yaml',
        'file_name_match_setting': 'setting_cubo.yaml',
        'profiling': False,
        'testing': True
    },

    # need a check with two games
    {
        'seed': 11,
        'gui': False,
        'file_name_player_one': 'players_for_test_purposes/Sequool.yaml',
        'file_name_player_two': 'Random.yaml',
        'file_name_match_setting': 'setting_tron.yaml',
        'profiling': False,
        'testing': True
    },

    # todo add basic eval (no neural nets)
]

configs_with_stock = [
                         # stockfish
                         {'seed': 11, 'gui': False, 'file_name_player_one': 'Stockfish.yaml',
                          'file_name_player_two': 'Random.yaml',
                          'file_name_match_setting': 'setting_jime.yaml', 'profiling': False},
                     ] + configs_base

configs_with_stock_and_gui = [
                                 # checking gui
                                 {'seed': 11, 'gui': True,
                                  'file_name_player_one': 'players_for_test_purposes/Sequool.yaml',
                                  'file_name_player_two': 'Random.yaml',
                                  'file_name_match_setting': 'setting_jime.yaml', 'profiling': False},

                             ] + configs_with_stock


def test_one_matches(
        configs=configs_base
):
    for config in configs:
        print(f'Running the SCRIPT with config {config}')
        script_object: scripts.IScript = scripts.create_script(
            script_type=scripts.ScriptType.OneMatch,
            extra_args=config
        )

        # run the script
        script_object.run()

        # terminate the script
        script_object.terminate()


def test_randomness():
    print('test randomness')
    # test randomness
    import chipiron as ch
    from chipiron.games.match.match_factories import create_match_manager_from_args
    from chipiron.games.match.match_results import MatchReport
    from chipiron.games.match.match_args import MatchArgs

    args: MatchArgs = MatchArgs()
    args.seed = 0
    args.file_name_player_one = 'players_for_test_purposes/Uniform.yaml'
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

    end_time = time.time()
    print(f'time: {end_time - start_time}')


if __name__ == "__main__":

    try:
        which_test = sys.argv[1]
        if which_test == 'gui_stock':
            print('gui_stock')
            test_configs = configs_with_stock_and_gui
        elif which_test == 'stock':
            print('stock')
            test_configs = configs_with_stock
    except Exception:
        test_configs = configs_base

    test_one_matches(configs=test_configs)

    test_randomness()
    print('ALL OK for ONE MATCH')

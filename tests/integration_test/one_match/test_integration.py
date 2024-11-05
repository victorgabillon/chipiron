import sys
import time

import chipiron.scripts as scripts
from chipiron.scripts.chipiron_args import ImplementationArgs
from chipiron.scripts.factory import create_script
from chipiron.scripts.script_args import BaseScriptArgs

# we need to not use multiprocessing to be able to use pytest therefore use setting_cubo  and not setting_jime

# todo check that the game is the same with or without rust
# todo check that the game is the same with or without board mofificaiton
# todo have a game that starts with an endgame to force to check the syzygy

start_time = time.time()
configs_base = [

    # random Player first to have a fast game
    {

        'gui': False,
        'match_args': {
            'seed': 11,
            'file_name_player_one': 'players_for_test_purposes/Sequool.yaml',
            'file_name_player_two': 'Random.yaml',
            'file_name_match_setting': 'setting_cubo.yaml'
        },
        'base_script_args': {
            'profiling': False,
            'testing': True
        }
    },

    {
        'gui': False,
        'match_args': {
            'seed': 11,
            'file_name_player_one':
                'players_for_test_purposes/Sequool.yaml',
            'file_name_player_two': 'players_for_test_purposes/RecurZipfSequool.yaml',
            'file_name_match_setting': 'setting_cubo.yaml'
        },
        'base_script_args': {
            'profiling': False,
            'testing': True
        }
    },

    {
        'gui': False,
        'match_args': {
            'seed': 11,
            'file_name_player_one': 'players_for_test_purposes/Sequool.yaml',
            'file_name_player_two': 'players_for_test_purposes/RecurZipfBase3.yaml',
            'file_name_match_setting': 'setting_cubo.yaml'
        },
        'base_script_args': {
            'profiling': False,
            'testing': True
        }
    },

    {
        'gui': False,

        'match_args': {
            'seed': 11,
            'file_name_player_one': 'players_for_test_purposes/RecurZipfBase4.yaml',
            'file_name_player_two': 'players_for_test_purposes/RecurZipfBase3.yaml',
            'file_name_match_setting': 'setting_cubo.yaml'
        },
        'base_script_args': {
            'profiling': False,
            'testing': True
        }
    },

    {
        'gui': False,

        'match_args': {
            'seed': 11,
            'file_name_player_one': 'players_for_test_purposes/Uniform.yaml',
            'file_name_player_two': 'players_for_test_purposes/RecurZipfBase3.yaml',
            'file_name_match_setting': 'setting_cubo.yaml'
        },
        'base_script_args': {
            'profiling': False,
            'testing': True
        }
    },

    {
        'gui': False,
        'match_args': {
            'seed': 11,

            'file_name_player_one': 'players_for_test_purposes/RecurZipfBase3.yaml',
            'file_name_player_two': 'players_for_test_purposes/RecurZipfBase3.yaml',
            'file_name_match_setting': 'setting_cubo.yaml'
        },
        'base_script_args': {
            'profiling': False,
            'testing': True
        }
    },

    # checking profiling
    {
        'gui': False,
        'match_args': {
            'seed': 11,

            'file_name_player_one': 'players_for_test_purposes/Sequool.yaml',
            'file_name_player_two': 'Random.yaml',
            'file_name_match_setting': 'setting_cubo.yaml'
        },
        'base_script_args': {
            'profiling': True,
            'testing': True
        }
    },

    # checking another seed haha
    {
        'gui': False,
        'match_args': {
            'seed': 12,

            'file_name_player_one': 'players_for_test_purposes/Sequool.yaml',
            'file_name_player_two': 'Random.yaml',
            'file_name_match_setting': 'setting_cubo.yaml'
        },
        'base_script_args': {
            'profiling': False,
            'testing': True
        }
    },

    # need a check with two games
    {
        'gui': False,

        'match_args': {
            'seed': 11,
            'file_name_player_one': 'players_for_test_purposes/Sequool.yaml',
            'file_name_player_two': 'Random.yaml',
            'file_name_match_setting': 'setting_tron.yaml'
        },
        'base_script_args': {
            'profiling': False,
            'testing': True
        }
    },

    # todo add basic eval (no neural nets)
]

configs_implementation = [
    {
        'implementation_args': {
            'use_board_modification': False,
            'use_rust_boards': False
        }
    },
    {
        'implementation_args': {
            'use_board_modification': True,
            'use_rust_boards': False
        }
    },
    {
        'implementation_args': {
            'use_board_modification': False,
            'use_rust_boards': True
        }
    },
    {
        'implementation_args': {
            'use_board_modification': True,
            'use_rust_boards': True
        }
    },
]

configs_with_stock = [
                         # stockfish
                         {
                             'gui': False,
                             'match_args': {
                                 'seed': 11, 'file_name_player_one': 'Stockfish.yaml',
                                 'file_name_player_two': 'Random.yaml',
                                 'file_name_match_setting': 'setting_jime.yaml'
                             },
                             'base_script_args': {
                                 'profiling': False}
                         }
                     ] + configs_base

configs_with_stock_and_gui = [
                                 # checking gui
                                 {
                                     'gui': True,
                                     'match_args': {
                                         'seed': 11,
                                         'file_name_player_one': 'players_for_test_purposes/Sequool.yaml',
                                         'file_name_player_two': 'Random.yaml',
                                         'file_name_match_setting': 'setting_jime.yaml'
                                     },
                                     'base_script_args': {'profiling': False}
                                 }

                             ] + configs_with_stock


def test_one_matches(
        configs=configs_base
):
    for config in configs:
        for config_implementation in configs_implementation:
            total_config = config | config_implementation
            print(f'Running the SCRIPT with config {total_config}')
            script_object: scripts.IScript = create_script(
                script_type=scripts.ScriptType.OneMatch,
                extra_args=total_config,
                should_parse_command_line_arguments=False
            )

            # run the script
            script_object.run()

            # terminate the script
            script_object.terminate()


def test_randomness():
    print('test randomness')

    # test randomness: just two time the same script anf nothing should change!! the seed is fixed to the same on both
    # cases

    import chipiron as ch
    from chipiron.games.match.match_factories import create_match_manager_from_args
    from chipiron.games.match.match_results import MatchReport
    from chipiron.games.match.match_args import MatchArgs

    match_args: MatchArgs = MatchArgs()
    match_args.seed = 0
    match_args.file_name_player_one = 'players_for_test_purposes/Uniform.yaml'
    match_args.file_name_player_two = 'players_for_test_purposes/Sequool.yaml'
    match_args.file_name_player_two = 'Random.yaml'
    match_args.file_name_match_setting = 'setting_tron.yaml'

    implementation_args: ImplementationArgs = ImplementationArgs()
    base_script_args: BaseScriptArgs = BaseScriptArgs()
    base_script_args.experiment_output_folder = None

    match_manager: ch.game.MatchManager = create_match_manager_from_args(
        match_args=match_args,
        implementation_args=implementation_args,
        base_script_args=base_script_args
    )
    match_report_base: MatchReport = match_manager.play_one_match()

    test_passed: bool = True
    number_test: int = 1

    for ind in range(number_test):
        match_manager: ch.game.MatchManager = create_match_manager_from_args(
            match_args=match_args,
            implementation_args=implementation_args,
            base_script_args=base_script_args
        )
        match_report: MatchReport = match_manager.play_one_match()

        # checking if two matches launched with the same fixed seed returns the same game moves.
        print('match_report.match_move_history, match_report_base.match_move_history',
              match_report.match_move_history, match_report_base.match_move_history)
        assert (match_report.match_move_history == match_report_base.match_move_history)

    print(match_report)

    end_time = time.time()
    print(f'time: {end_time - start_time}')


def test_same_game_with_or_without_rust():


    import chipiron as ch
    from chipiron.games.match.match_factories import create_match_manager_from_args
    from chipiron.games.match.match_results import MatchReport
    from chipiron.games.match.match_args import MatchArgs

    match_args: MatchArgs = MatchArgs()
    match_args.seed = 0
    match_args.file_name_player_one = 'players_for_test_purposes/Uniform.yaml'
    match_args.file_name_player_two = 'players_for_test_purposes/Sequool.yaml'
    match_args.file_name_player_two = 'Random.yaml'
    match_args.file_name_match_setting = 'setting_tron.yaml'

    implementation_args: ImplementationArgs = ImplementationArgs(use_rust_boards=False)
    base_script_args: BaseScriptArgs = BaseScriptArgs()
    base_script_args.experiment_output_folder = None

    match_manager: ch.game.MatchManager = create_match_manager_from_args(
        match_args=match_args,
        implementation_args=implementation_args,
        base_script_args=base_script_args
    )
    match_report_base: MatchReport = match_manager.play_one_match()

    test_passed: bool = True
    number_test: int = 1

    for ind in range(number_test):
        implementation_args: ImplementationArgs = ImplementationArgs(use_rust_boards=True)

        match_manager: ch.game.MatchManager = create_match_manager_from_args(
            match_args=match_args,
            implementation_args=implementation_args,
            base_script_args=base_script_args
        )
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

    #test_one_matches(configs=test_configs)

    # todo maybe merege the test that verifiy if two game are unchanged when variying some parameters like ranfom and rust atm
    test_same_game_with_or_without_rust()
    test_randomness()
    print('ALL OK for ONE MATCH')

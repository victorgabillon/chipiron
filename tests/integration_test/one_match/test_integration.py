import copy
import logging
import time
from typing import Any

from parsley_coco import make_partial_dataclass_with_optional_paths

import chipiron.scripts as scripts
from chipiron.games.match.match_args import MatchArgs
from chipiron.games.match.MatchTag import MatchConfigTag
from chipiron.players.player_ids import PlayerConfigTag
from chipiron.scripts.chipiron_args import ImplementationArgs
from chipiron.scripts.factory import create_script
from chipiron.scripts.one_match.one_match import MatchScriptArgs
from chipiron.scripts.script_args import BaseScriptArgs
from chipiron.utils.logger import chipiron_logger, suppress_logging

# we need to not use multiprocessing to be able to use pytest therefore use setting_cubo  and not setting_jime

# todo check that the game is the same with or without rust
# todo check that the game is the same with or without board mofificaiton
# todo have a game that starts with an endgame to force to check the syzygy

start_time = time.time()

PartialOpMatchScriptArgs = make_partial_dataclass_with_optional_paths(
    cls=MatchScriptArgs
)
PartialOpMatchArgs = make_partial_dataclass_with_optional_paths(cls=MatchArgs)
PartialOpBaseScriptArgs = make_partial_dataclass_with_optional_paths(cls=BaseScriptArgs)

configs_base: list[Any] = [
    # random Player first to have a fast game
    PartialOpMatchScriptArgs(
        gui=False,
        match_args=PartialOpMatchArgs(
            player_one=PlayerConfigTag.Test_Sequool,
            player_two=PlayerConfigTag.RANDOM,
            match_setting=MatchConfigTag.Cubo,
        ),
        base_script_args=PartialOpBaseScriptArgs(
            profiling=False, testing=True, seed=11
        ),
    ),
    PartialOpMatchScriptArgs(
        gui=False,
        match_args=PartialOpMatchArgs(
            player_one=PlayerConfigTag.Test_Sequool,
            player_two=PlayerConfigTag.Test_RecurZipfSequool,
            match_setting=MatchConfigTag.Cubo,
        ),
        base_script_args=PartialOpBaseScriptArgs(
            profiling=False, testing=True, seed=11
        ),
    ),
    PartialOpMatchScriptArgs(
        gui=False,
        match_args=PartialOpMatchArgs(
            player_one=PlayerConfigTag.Test_Sequool,
            player_two=PlayerConfigTag.Test_RecurZipfBase3,
            match_setting=MatchConfigTag.Cubo,
        ),
        base_script_args=PartialOpBaseScriptArgs(
            profiling=False, testing=True, seed=11
        ),
    ),
    PartialOpMatchScriptArgs(
        gui=False,
        match_args=PartialOpMatchArgs(
            player_one=PlayerConfigTag.Test_RecurZipfBase4,
            player_two=PlayerConfigTag.Test_RecurZipfBase3,
            match_setting=MatchConfigTag.Cubo,
        ),
        base_script_args=PartialOpBaseScriptArgs(
            profiling=False, testing=True, seed=11
        ),
    ),
    PartialOpMatchScriptArgs(
        gui=False,
        match_args=PartialOpMatchArgs(
            player_one=PlayerConfigTag.Test_Uniform,
            player_two=PlayerConfigTag.Test_RecurZipfBase3,
            match_setting=MatchConfigTag.Cubo,
        ),
        base_script_args=PartialOpBaseScriptArgs(
            profiling=False, testing=True, seed=11
        ),
    ),
    PartialOpMatchScriptArgs(
        gui=False,
        match_args=PartialOpMatchArgs(
            player_one=PlayerConfigTag.Test_RecurZipfBase3,
            player_two=PlayerConfigTag.Test_RecurZipfBase3,
            match_setting=MatchConfigTag.Cubo,
        ),
        base_script_args=PartialOpBaseScriptArgs(
            profiling=False, testing=True, seed=11
        ),
    ),
    # checking profiling
    PartialOpMatchScriptArgs(
        gui=False,
        match_args=PartialOpMatchArgs(
            player_one=PlayerConfigTag.Test_Sequool,
            player_two=PlayerConfigTag.RANDOM,
            match_setting=MatchConfigTag.Cubo,
        ),
        base_script_args=PartialOpBaseScriptArgs(profiling=True, testing=True, seed=11),
    ),
    # checking another seed haha
    PartialOpMatchScriptArgs(
        gui=False,
        match_args=PartialOpMatchArgs(
            player_one=PlayerConfigTag.Test_Sequool,
            player_two=PlayerConfigTag.RANDOM,
            match_setting=MatchConfigTag.Cubo,
        ),
        base_script_args=PartialOpBaseScriptArgs(
            profiling=False, testing=True, seed=12
        ),
    ),
    # need a check with two games
    PartialOpMatchScriptArgs(
        gui=False,
        match_args=PartialOpMatchArgs(
            player_one=PlayerConfigTag.Test_Sequool,
            player_two=PlayerConfigTag.RANDOM,
            match_setting=MatchConfigTag.Tron,
        ),
        base_script_args=PartialOpBaseScriptArgs(
            profiling=False, testing=True, seed=11
        ),
    ),
    # todo add basic eval (no neural nets)
]

PartialOpImplementationArgs = make_partial_dataclass_with_optional_paths(
    cls=ImplementationArgs
)
configs_implementation = [
    PartialOpImplementationArgs(use_board_modification=False, use_rust_boards=False),
    PartialOpImplementationArgs(use_board_modification=True, use_rust_boards=False),
    PartialOpImplementationArgs(use_board_modification=False, use_rust_boards=True),
    PartialOpImplementationArgs(use_board_modification=True, use_rust_boards=True),
]


def test_one_matches(configs=configs_base):
    index: int = 0
    for config in configs:
        for config_implementation in configs_implementation:
            index += 1
            total_config = copy.copy(config)
            total_config.implementation_args = config_implementation
            chipiron_logger.info(
                f"Running the test #{index}/{len(configs) * len(configs_implementation)} with SCRIPT with config {total_config}"
            )

            with suppress_logging(
                chipiron_logger, level=logging.ERROR
            ):  # only warning and error logging during script run

                script_object: scripts.IScript = create_script(
                    script_type=scripts.ScriptType.OneMatch,
                    extra_args=total_config,
                    should_parse_command_line_arguments=False,
                )

                # run the script
                script_object.run()

                # terminate the script
                script_object.terminate()


def test_randomness():
    chipiron_logger.info("test randomness")

    # test randomness: just two time the same script anf nothing should change!! the seed is fixed to the same on both
    # cases

    import chipiron as ch
    from chipiron.games.match.match_args import MatchArgs
    from chipiron.games.match.match_factories import create_match_manager_from_args
    from chipiron.games.match.match_results import MatchReport

    match_args: MatchArgs = MatchArgs()
    match_args.seed = 0
    match_args.player_one = PlayerConfigTag.Test_Uniform.get_players_args()
    match_args.player_two = PlayerConfigTag.Test_Sequool.get_players_args()
    match_args.player_two = PlayerConfigTag.RANDOM.get_players_args()
    match_args.match_setting = MatchConfigTag.Tron.get_match_settings_args()

    implementation_args: ImplementationArgs = ImplementationArgs()
    base_script_args: BaseScriptArgs = BaseScriptArgs()
    base_script_args.relative_script_instance_experiment_output_folder = None

    with suppress_logging(
        chipiron_logger, level=logging.ERROR
    ):  # only warning and error logging during script run

        match_manager: ch.game.MatchManager = create_match_manager_from_args(
            match_args=match_args,
            implementation_args=implementation_args,
            base_script_args=base_script_args,
        )
        match_report_base: MatchReport = match_manager.play_one_match()

    test_passed: bool = True
    number_test: int = 1

    for ind in range(number_test):
        with suppress_logging(
            chipiron_logger, level=logging.ERROR
        ):  # only warning and error logging during script run

            match_manager: ch.game.MatchManager = create_match_manager_from_args(
                match_args=match_args,
                implementation_args=implementation_args,
                base_script_args=base_script_args,
            )
            match_report: MatchReport = match_manager.play_one_match()

        # checking if two matches launched with the same fixed seed returns the same game moves.
        chipiron_logger.debug(
            f"match_report.match_move_history, match_report_base.match_move_history {match_report.match_move_history} {match_report_base.match_move_history}"
        )
        assert match_report.match_move_history == match_report_base.match_move_history

    chipiron_logger.debug(match_report)

    end_time = time.time()
    chipiron_logger.info(f"time: {end_time - start_time}")


def test_same_game_with_or_without_rust():
    import chipiron as ch
    from chipiron.games.match.match_args import MatchArgs
    from chipiron.games.match.match_factories import create_match_manager_from_args
    from chipiron.games.match.match_results import MatchReport

    match_args: MatchArgs = MatchArgs()
    match_args.seed = 0
    match_args.player_one = PlayerConfigTag.Test_Uniform.get_players_args()
    match_args.player_two = PlayerConfigTag.Test_Sequool.get_players_args()
    match_args.player_two = PlayerConfigTag.RANDOM.get_players_args()
    match_args.match_setting = MatchConfigTag.Tron.get_match_settings_args()

    implementation_args: ImplementationArgs = ImplementationArgs(use_rust_boards=False)

    # important to set universal_behavior=True to obtain a standard behavior in test (based on sorting wrt uci)
    base_script_args: BaseScriptArgs = BaseScriptArgs(universal_behavior=True)
    base_script_args.relative_script_instance_experiment_output_folder = None

    with suppress_logging(
        chipiron_logger, level=logging.ERROR
    ):  # only warning and error logging during script run

        match_manager: ch.game.MatchManager = create_match_manager_from_args(
            match_args=match_args,
            implementation_args=implementation_args,
            base_script_args=base_script_args,
        )
        match_report_base: MatchReport = match_manager.play_one_match()

    number_test: int = 1

    for ind in range(number_test):
        implementation_args: ImplementationArgs = ImplementationArgs(
            use_rust_boards=True
        )

        with suppress_logging(
            chipiron_logger, level=logging.ERROR
        ):  # only warning and error logging during script run

            match_manager: ch.game.MatchManager = create_match_manager_from_args(
                match_args=match_args,
                implementation_args=implementation_args,
                base_script_args=base_script_args,
            )

            match_report: MatchReport = match_manager.play_one_match()

        # checking if two matches launched with the same fixed seed returns the same game moves.
        chipiron_logger.debug(
            f"match_report.match_move_history, match_report_base.match_move_history {match_report.match_move_history} {match_report_base.match_move_history}",
        )
        assert match_report.match_move_history == match_report_base.match_move_history

    chipiron_logger.debug(match_report)

    end_time = time.time()
    chipiron_logger.info(f"time: {end_time - start_time}")


if __name__ == "__main__":
    chipiron_logger.setLevel(logging.INFO)

    test_configs = configs_base

    # todo maybe merege the test that verifiy if two game are unchanged when variying some parameters like ranfom and rust atm
    test_same_game_with_or_without_rust()
    test_randomness()
    test_one_matches(configs=test_configs)

    chipiron_logger.info("ALL OK for ONE MATCH")

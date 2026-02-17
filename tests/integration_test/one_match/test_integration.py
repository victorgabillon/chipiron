"""Document the module contains integration tests for the one match script in the Chipiron project."""

import argparse
import copy
import logging
import time
from typing import TYPE_CHECKING, Any

from anemone import TreeAndValuePlayerArgs
from anemone.progress_monitor.progress_monitor import (
    StoppingCriterionTypes,
    TreeBranchLimitArgs,
)
from parsley import (
    make_partial_dataclass_with_optional_paths,
    resolve_extended_object,
)

import chipiron as ch
import chipiron.scripts as scripts
from chipiron.games.match.match_args import MatchArgs
from chipiron.games.match.match_factories import create_match_manager_from_args
from chipiron.games.match.match_tag import MatchConfigTag
from chipiron.players import PlayerArgs
from chipiron.players.move_selector.move_selector_types import MoveSelectorTypes
from chipiron.players.move_selector.stockfish_selector import StockfishSelector
from chipiron.players.move_selector.tree_and_value_args import TreeAndValueAppArgs
from chipiron.players.player_ids import PlayerConfigTag
from chipiron.scripts.chipiron_args import ImplementationArgs
from chipiron.scripts.factory import create_script
from chipiron.scripts.one_match.one_match import MatchScriptArgs
from chipiron.scripts.script_args import BaseScriptArgs
from chipiron.utils.logger import chipiron_logger, suppress_logging

if TYPE_CHECKING:
    from chipiron.games.match.match_results import MatchReport

# we need to not use multiprocessing to be able to use pytest therefore use setting_cubo  and not setting_jime

# TODO: check that the game is the same with or without rust
# TODO: check that the game is the same with or without board mofificaiton
# TODO: have a game that starts with an endgame to force to check the syzygy

start_time = time.time()

PartialOpMatchScriptArgs = make_partial_dataclass_with_optional_paths(
    cls=MatchScriptArgs
)
PartialOpMatchArgs = make_partial_dataclass_with_optional_paths(cls=MatchArgs)
PartialOpBaseScriptArgs = make_partial_dataclass_with_optional_paths(cls=BaseScriptArgs)
PartialOpPlayerArgs = make_partial_dataclass_with_optional_paths(cls=PlayerArgs)
PartialOpTreeAndValuePlayerArgs = make_partial_dataclass_with_optional_paths(
    cls=TreeAndValuePlayerArgs
)
PartialOpTreeAndValueAppArgs = make_partial_dataclass_with_optional_paths(
    cls=TreeAndValueAppArgs
)
PartialOpTreeBranchLimitArgs = make_partial_dataclass_with_optional_paths(
    cls=TreeBranchLimitArgs
)


# Create a common overwrite for test players with tree branch limit of 100
TEST_TREE_BRANCH_LIMIT = 100
test_player_overwrite = PartialOpPlayerArgs(
    main_move_selector=PartialOpTreeAndValueAppArgs(
        type=MoveSelectorTypes.TREE_AND_VALUE,
        anemone_args=PartialOpTreeAndValuePlayerArgs(
            type=MoveSelectorTypes.TREE_AND_VALUE,
            stopping_criterion=PartialOpTreeBranchLimitArgs(
                type=StoppingCriterionTypes.TREE_BRANCH_LIMIT,
                tree_branch_limit=TEST_TREE_BRANCH_LIMIT,
            ),
        ),
    )
)


def _build_base_configs() -> list[Any]:
    """Build the base configurations, including Stockfish test only if available."""
    configs = [
        # random Player first to have a fast game
        PartialOpMatchScriptArgs(
            gui=False,
            match_args=PartialOpMatchArgs(
                player_one=PlayerConfigTag.SEQUOOL,
                player_two=PlayerConfigTag.RANDOM,
                match_setting=MatchConfigTag.CUBO,
                player_one_overwrite=test_player_overwrite,
            ),
            base_script_args=PartialOpBaseScriptArgs(
                profiling=False, testing=True, seed=11
            ),
        ),
        PartialOpMatchScriptArgs(
            gui=False,
            match_args=PartialOpMatchArgs(
                player_one=PlayerConfigTag.SEQUOOL,
                player_two=PlayerConfigTag.RECUR_ZIPF_BASE_3,
                match_setting=MatchConfigTag.CUBO,
                player_one_overwrite=test_player_overwrite,
                player_two_overwrite=test_player_overwrite,
            ),
            base_script_args=PartialOpBaseScriptArgs(
                profiling=False, testing=True, seed=11
            ),
        ),
        PartialOpMatchScriptArgs(
            gui=False,
            match_args=PartialOpMatchArgs(
                player_one=PlayerConfigTag.RECUR_ZIPF_BASE_4,
                player_two=PlayerConfigTag.RECUR_ZIPF_BASE_3,
                match_setting=MatchConfigTag.CUBO,
                player_one_overwrite=test_player_overwrite,
                player_two_overwrite=test_player_overwrite,
            ),
            base_script_args=PartialOpBaseScriptArgs(
                profiling=False, testing=True, seed=11
            ),
        ),
        PartialOpMatchScriptArgs(
            gui=False,
            match_args=PartialOpMatchArgs(
                player_one=PlayerConfigTag.UNIFORM,
                player_two=PlayerConfigTag.RECUR_ZIPF_BASE_3,
                match_setting=MatchConfigTag.CUBO,
                player_one_overwrite=test_player_overwrite,
                player_two_overwrite=test_player_overwrite,
            ),
            base_script_args=PartialOpBaseScriptArgs(
                profiling=False, testing=True, seed=11
            ),
        ),
        PartialOpMatchScriptArgs(
            gui=False,
            match_args=PartialOpMatchArgs(
                player_one=PlayerConfigTag.RECUR_ZIPF_BASE_3,
                player_two=PlayerConfigTag.RECUR_ZIPF_BASE_3,
                match_setting=MatchConfigTag.CUBO,
                player_one_overwrite=test_player_overwrite,
                player_two_overwrite=test_player_overwrite,
            ),
            base_script_args=PartialOpBaseScriptArgs(
                profiling=False, testing=True, seed=11
            ),
        ),
        # checking profiling
        PartialOpMatchScriptArgs(
            gui=False,
            match_args=PartialOpMatchArgs(
                player_one=PlayerConfigTag.SEQUOOL,
                player_two=PlayerConfigTag.RANDOM,
                match_setting=MatchConfigTag.CUBO,
                player_one_overwrite=test_player_overwrite,
            ),
            base_script_args=PartialOpBaseScriptArgs(
                profiling=True, testing=True, seed=11
            ),
        ),
        # checking another seed haha
        PartialOpMatchScriptArgs(
            gui=False,
            match_args=PartialOpMatchArgs(
                player_one=PlayerConfigTag.SEQUOOL,
                player_two=PlayerConfigTag.RANDOM,
                match_setting=MatchConfigTag.CUBO,
                player_one_overwrite=test_player_overwrite,
            ),
            base_script_args=PartialOpBaseScriptArgs(
                profiling=False, testing=True, seed=12
            ),
        ),
    ]

    # Only add Stockfish test if it's properly installed
    if StockfishSelector.is_stockfish_available():
        configs.append(
            # checking stockfish
            PartialOpMatchScriptArgs(
                gui=False,
                match_args=PartialOpMatchArgs(
                    player_one=PlayerConfigTag.RANDOM,
                    player_two=PlayerConfigTag.STOCKFISH,
                    match_setting=MatchConfigTag.CUBO,
                ),
                base_script_args=PartialOpBaseScriptArgs(
                    profiling=False, testing=True, seed=12
                ),
            )
        )

    # Add remaining configs
    configs.extend(
        [
            # need a check with two games
            PartialOpMatchScriptArgs(
                gui=False,
                match_args=PartialOpMatchArgs(
                    player_one=PlayerConfigTag.SEQUOOL,
                    player_two=PlayerConfigTag.RANDOM,
                    match_setting=MatchConfigTag.TRON,
                    player_one_overwrite=test_player_overwrite,
                ),
                base_script_args=PartialOpBaseScriptArgs(
                    profiling=False, testing=True, seed=11
                ),
            ),
            # TODO: add basic eval (no neural nets)
        ]
    )

    return configs


configs_base: list[Any] = _build_base_configs()

PartialOpImplementationArgs = make_partial_dataclass_with_optional_paths(
    cls=ImplementationArgs
)
configs_implementation = [
    PartialOpImplementationArgs(use_board_modification=False, use_rust_boards=False),
    PartialOpImplementationArgs(use_board_modification=True, use_rust_boards=False),
    PartialOpImplementationArgs(use_board_modification=False, use_rust_boards=True),
    PartialOpImplementationArgs(use_board_modification=True, use_rust_boards=True),
]


def test_one_matches(
    configs: list[Any] | None = None, log_level: int = logging.ERROR
) -> None:
    """Run integration tests for one match using provided configurations.

    If no configs are provided, uses the default configs_base.

    Args:
        configs: List of test configurations to run
        log_level: Logging level to use during test execution (ERROR for pytest, INFO for main)

    """
    if configs is None:
        configs = configs_base

    # Log Stockfish availability status
    if StockfishSelector.is_stockfish_available():
        chipiron_logger.info("Stockfish is available - including Stockfish tests")
    else:
        chipiron_logger.info(
            "Stockfish not available - skipping Stockfish tests (run 'make stockfish' to enable)"
        )

    index: int = 0
    for config in configs:
        for config_implementation in configs_implementation:
            index += 1
            total_config = copy.copy(config)
            total_config.implementation_args = config_implementation
            chipiron_logger.info(
                "Running the test #%d/%d with SCRIPT with config %s",
                index,
                len(configs) * len(configs_implementation),
                total_config,
            )

            with suppress_logging(
                chipiron_logger, level=log_level
            ):  # logging level depends on how the function is called
                script_object: scripts.IScript = create_script(
                    script_type=scripts.ScriptType.ONE_MATCH,
                    extra_args=total_config,
                    should_parse_command_line_arguments=False,
                )

                # run the script
                script_object.run()

                # terminate the script
                script_object.terminate()


def test_randomness(log_level: int = logging.ERROR) -> None:
    """Test that running the same match twice with a fixed seed produces identical results.

    Args:
        log_level: Logging level to use during test execution (ERROR for pytest, INFO for main)

    """
    chipiron_logger.info("test randomness")

    # test randomness: just two time the same script anf nothing should change!! the seed is fixed to the same on both
    # cases

    match_args: MatchArgs = MatchArgs()
    match_args.player_one = PlayerConfigTag.UNIFORM.get_players_args()
    match_args.player_two = PlayerConfigTag.RANDOM.get_players_args()
    match_args.match_setting = MatchConfigTag.TRON.get_match_settings_args()
    match_args.player_one = PlayerConfigTag.UNIFORM.get_players_args()
    match_args.player_two = PlayerConfigTag.RANDOM.get_players_args()
    match_args.match_setting = MatchConfigTag.TRON.get_match_settings_args()

    # Override player two with test tree move limit using parsley_coco
    match_args.player_one_overwrite = test_player_overwrite
    match_args = resolve_extended_object(extended_obj=match_args, base_cls=MatchArgs)

    assert isinstance(match_args.player_one, PlayerArgs)
    print(
        f"match_args.player_one.main_move_selector{match_args.player_one.main_move_selector}, type: {type(match_args.player_one.main_move_selector)}  "
    )
    assert isinstance(match_args.player_one.main_move_selector, TreeAndValueAppArgs)
    assert isinstance(
        match_args.player_one.main_move_selector.anemone_args.stopping_criterion,
        TreeBranchLimitArgs,
    )

    print(
        f" args_tree_branch_limit: {match_args.player_one.main_move_selector.anemone_args.stopping_criterion.tree_branch_limit} , Target branch limit{TEST_TREE_BRANCH_LIMIT}"
    )
    assert (
        match_args.player_one.main_move_selector.anemone_args.stopping_criterion.tree_branch_limit
        == TEST_TREE_BRANCH_LIMIT
    )

    implementation_args: ImplementationArgs = ImplementationArgs()
    base_script_args: BaseScriptArgs = BaseScriptArgs()
    base_script_args.relative_script_instance_experiment_output_folder = None

    with suppress_logging(
        chipiron_logger, level=log_level
    ):  # logging level depends on how the function is called
        match_manager: ch.game.MatchManager = create_match_manager_from_args(
            match_args=match_args,
            implementation_args=implementation_args,
            base_script_args=base_script_args,
        )
        match_report_base: MatchReport = match_manager.play_one_match()

    number_test: int = 1

    for _ in range(number_test):
        with suppress_logging(
            chipiron_logger, level=log_level
        ):  # logging level depends on how the function is called
            match_manager: ch.game.MatchManager = create_match_manager_from_args(
                match_args=match_args,
                implementation_args=implementation_args,
                base_script_args=base_script_args,
            )
            match_report: MatchReport = match_manager.play_one_match()

        # checking if two matches launched with the same fixed seed returns the same game moves.
        chipiron_logger.debug(
            "match_report.match_move_history, match_report_base.match_move_history %s %s",
            match_report.match_move_history,
            match_report_base.match_move_history,
        )
        assert match_report.match_move_history == match_report_base.match_move_history

    chipiron_logger.debug(match_report)

    end_time = time.time()
    chipiron_logger.info("time: %s", end_time - start_time)


def test_same_game_with_or_without_rust(log_level=logging.ERROR) -> None:
    """Test that running the same match with and without Rust boards produces identical results.

    Args:
        log_level: Logging level to use during test execution (ERROR for pytest, INFO for main)

    """
    match_args: MatchArgs = MatchArgs()

    match_args.player_one = PlayerConfigTag.UNIFORM.get_players_args()

    match_args.player_two = PlayerConfigTag.RANDOM.get_players_args()
    match_args.match_setting = MatchConfigTag.TRON.get_match_settings_args()
    match_args.player_one_overwrite = (
        test_player_overwrite  # Override player one with test tree move limit
    )

    # Override player two with test tree move limit using parsley_coco
    match_args = resolve_extended_object(extended_obj=match_args, base_cls=MatchArgs)

    assert isinstance(match_args.player_one, PlayerArgs)
    print(
        f"match_args.player_one.main_move_selector{match_args.player_one.main_move_selector}, type: {type(match_args.player_one.main_move_selector)}  "
    )
    assert isinstance(match_args.player_one.main_move_selector, TreeAndValueAppArgs)
    assert isinstance(
        match_args.player_one.main_move_selector.anemone_args.stopping_criterion,
        TreeBranchLimitArgs,
    )

    print(
        f" args_tree_branch_limit: {match_args.player_one.main_move_selector.anemone_args.stopping_criterion.tree_branch_limit} , Target branch limit{TEST_TREE_BRANCH_LIMIT}"
    )
    assert (
        match_args.player_one.main_move_selector.anemone_args.stopping_criterion.tree_branch_limit
        == TEST_TREE_BRANCH_LIMIT
    )

    implementation_args: ImplementationArgs = ImplementationArgs(use_rust_boards=False)

    # important to set universal_behavior=True to obtain a standard behavior in test (based on sorting wrt uci)
    base_script_args: BaseScriptArgs = BaseScriptArgs(universal_behavior=True)
    base_script_args.relative_script_instance_experiment_output_folder = None

    with suppress_logging(
        chipiron_logger, level=log_level
    ):  # logging level depends on how the function is called
        match_manager: ch.game.MatchManager = create_match_manager_from_args(
            match_args=match_args,
            implementation_args=implementation_args,
            base_script_args=base_script_args,
        )
        match_report_base: MatchReport = match_manager.play_one_match()

    number_test: int = 1

    for _ in range(number_test):
        implementation_args: ImplementationArgs = ImplementationArgs(
            use_rust_boards=True
        )

        with suppress_logging(
            chipiron_logger, level=log_level
        ):  # logging level depends on how the function is called
            match_manager: ch.game.MatchManager = create_match_manager_from_args(
                match_args=match_args,
                implementation_args=implementation_args,
                base_script_args=base_script_args,
            )

            match_report: MatchReport = match_manager.play_one_match()

        # checking if two matches launched with the same fixed seed returns the same game moves.
        chipiron_logger.debug(
            "match_report.match_move_history, match_report_base.match_move_history %s %s",
            match_report.match_move_history,
            match_report_base.match_move_history,
        )
        assert match_report.match_move_history == match_report_base.match_move_history

    chipiron_logger.debug(match_report)

    end_time = time.time()
    chipiron_logger.info("time: %s", end_time - start_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run one_match integration tests with optional log level."
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    args = parser.parse_args()

    # Convert string log level to logging constant
    level = getattr(logging, args.log_level.upper(), logging.INFO)
    chipiron_logger.setLevel(level)

    test_configs = configs_base

    # Pass chosen level when running from main for more verbose output
    test_same_game_with_or_without_rust(log_level=level)
    test_randomness(log_level=level)
    test_one_matches(configs=test_configs, log_level=level)

    chipiron_logger.info("ALL OK for ONE MATCH")

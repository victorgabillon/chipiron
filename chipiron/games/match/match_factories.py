"""
This module contains functions for creating match managers in the Chipiron game framework.
"""

import multiprocessing
import queue

import chipiron as ch
import chipiron.games.game as game
import chipiron.players as players
from chipiron.environments.chess.board.factory import create_board_factory
from chipiron.games.game.game_manager_factory import GameManagerFactory
from chipiron.games.match.match_args import MatchArgs
from chipiron.games.match.match_manager import MatchManager
from chipiron.games.match.match_results_factory import MatchResultsFactory
from chipiron.games.match.utils import fetch_match_games_args_convert_and_save
from chipiron.players.boardevaluators.board_evaluator import IGameBoardEvaluator
from chipiron.players.boardevaluators.factory import create_game_board_evaluator
from chipiron.players.boardevaluators.table_base.factory import create_syzygy
from chipiron.players.boardevaluators.table_base.syzygy import SyzygyTable
from chipiron.players.utils import fetch_two_players_args_convert_and_save
from chipiron.scripts.chipiron_args import ImplementationArgs
from chipiron.scripts.script_args import BaseScriptArgs
from chipiron.utils import path
from chipiron.utils.is_dataclass import IsDataclass
from .match_settings_args import MatchSettingsArgs


def create_match_manager(
        args_match: MatchSettingsArgs,
        args_player_one: players.PlayerArgs,
        args_player_two: players.PlayerArgs,
        args_game: game.GameArgs,
        implementation_args: ImplementationArgs,
        seed: int | None = None,
        output_folder_path: path | None = None,
        gui: bool = False
) -> MatchManager:
    """
    Create a match manager for running matches between two players.

    Args:
        implementation_args: (ImplementationArgs) the implementation args
        args_match (MatchSettingsArgs): The match settings arguments.
        args_player_one (players.PlayerArgs): The arguments for player one.
        args_player_two (players.PlayerArgs): The arguments for player two.
        args_game (game.GameArgs): The game arguments.
        seed (int | None, optional): The seed for random number generation. Defaults to None.
        output_folder_path (path | None, optional): The output folder path. Defaults to None.
        gui (bool, optional): Flag indicating whether to enable GUI. Defaults to False.

    Returns:
        MatchManager: The created match manager.
    """
    main_thread_mailbox: queue.Queue[IsDataclass] = multiprocessing.Manager().Queue()

    # Creation of the Syzygy table for perfect play in low pieces cases, needed by the GameManager
    # and can also be used by the players
    syzygy_mailbox: SyzygyTable | None = create_syzygy()

    player_one_name: str = args_player_one.name
    player_two_name: str = args_player_two.name

    game_board_evaluator: IGameBoardEvaluator = create_game_board_evaluator(gui=gui)

    board_factory: BoardFactory = create_board_factory(
        use_rust_boards=implementation_args.use_rust_boards,
        use_board_modification=implementation_args.use_board_modification
    )

    game_manager_factory: GameManagerFactory = GameManagerFactory(
        syzygy_table=syzygy_mailbox,
        game_manager_board_evaluator=game_board_evaluator,
        output_folder_path=output_folder_path,
        main_thread_mailbox=main_thread_mailbox,
    )

    match_results_factory: MatchResultsFactory = MatchResultsFactory(
        player_one_name=player_one_name,
        player_two_name=player_two_name
    )

    game_args_factory: game.GameArgsFactory = game.GameArgsFactory(
        args_match=args_match,
        args_player_one=args_player_one,
        args_player_two=args_player_two,
        seed_=seed,
        args_game=args_game
    )

    match_manager: MatchManager = MatchManager(
        player_one_id=player_one_name,
        player_two_id=player_two_name,
        game_manager_factory=game_manager_factory,
        game_args_factory=game_args_factory,
        match_results_factory=match_results_factory,
        output_folder_path=output_folder_path
    )
    return match_manager


def create_match_manager_from_args(
        match_args: MatchArgs,
        base_script_args: BaseScriptArgs,
        implementation_args: ImplementationArgs,
        gui: bool = False
) -> MatchManager:
    """
    Create a match manager from the given arguments.

    Args:
        implementation_args(ImplementationArgs): The implementation args
        match_args (MatchArgs): The match arguments.
        base_script_args (ScriptArgs) The script arguments.
        gui (bool, optional): Flag indicating whether to enable GUI. Defaults to False.

    Returns:
        MatchManager: The created match manager.
    """
    player_one_args: players.PlayerArgs
    player_two_args: players.PlayerArgs
    player_one_args, player_two_args = fetch_two_players_args_convert_and_save(
        file_name_player_one=match_args.file_name_player_one,
        file_name_player_two=match_args.file_name_player_two,
        modification_player_one=match_args.player_one,
        modification_player_two=match_args.player_two,
        experiment_output_folder=base_script_args.experiment_output_folder
    )

    # Recovering args from yaml file for match and game and merging with extra args and converting
    # to standardized dataclass
    match_setting_args: MatchSettingsArgs
    game_args: game.GameArgs
    match_setting_args, game_args = fetch_match_games_args_convert_and_save(
        profiling=base_script_args.profiling,
        testing=base_script_args.testing,
        file_name_match_setting=match_args.file_name_match_setting,
        modification=match_args.match,
        experiment_output_folder=base_script_args.experiment_output_folder
    )

    assert player_one_args.name != 'Command_Line_Human.yaml' or not game_args.each_player_has_its_own_thread
    assert player_two_args.name != 'Command_Line_Human.yaml' or not game_args.each_player_has_its_own_thread

    # taking care of random
    ch.set_seeds(seed=match_args.seed)

    print('self.args.experiment_output_folder', base_script_args.experiment_output_folder)
    match_manager: MatchManager = create_match_manager(
        args_match=match_setting_args,
        args_player_one=player_one_args,
        args_player_two=player_two_args,
        output_folder_path=base_script_args.experiment_output_folder,
        seed=match_args.seed,
        args_game=game_args,
        gui=gui,
        implementation_args=implementation_args
    )

    return match_manager

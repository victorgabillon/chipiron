"""
Module for creating players.
"""

import queue
import random
from typing import Any

import chess

import chipiron.players.boardevaluators.table_base as table_base
from chipiron.players.boardevaluators.table_base.factory import create_syzygy
from chipiron.players.boardevaluators.table_base.syzygy_table import SyzygyTable
from chipiron.players.player_args import PlayerArgs
from chipiron.players.utils import fetch_player_args_convert_and_save
from . import move_selector
from .game_player import GamePlayer
from .player import Player
from .player_args import PlayerFactoryArgs
from ..utils.dataclass import IsDataclass


def create_chipiron_player(
        depth: int,
        use_rusty_board: bool,
        random_generator: random.Random,
        queue_progress_player: queue.Queue[IsDataclass] | None = None,

) -> Player:
    """
    Creates the chipiron champion/representative/standard/default player

    Args:
        depth: int, the depth at which computation should be made.

    Returns: the player

    """
    syzygy_table: table_base.SyzygyTable[Any] | None = create_syzygy(use_rust=use_rusty_board)

    args_player: PlayerArgs = fetch_player_args_convert_and_save(
        file_name_player='data/players/player_config/chipiron/chipiron.yaml',
        from_data_folder=False)

    main_move_selector: move_selector.MoveSelector | None = move_selector.create_main_move_selector(
        move_selector_instance_or_args=args_player.main_move_selector,
        syzygy=syzygy_table,
        random_generator=random_generator,
        queue_progress_player=queue_progress_player
    )

    assert main_move_selector is not None

    return Player(
        name='chipiron',
        syzygy=syzygy_table,
        main_move_selector=main_move_selector
    )


def create_player_from_file(
        player_args_file: str,
        random_generator: random.Random,
        use_rusty_board: bool,
        queue_progress_player: queue.Queue[IsDataclass] | None = None

) -> Player:
    """Create a player object from a file.

    Args:
        player_args_file (path): The path to the player arguments file.
        random_generator (random.Random): The random number generator.

    Returns:
        Player: The created player object.
    """
    args: PlayerArgs = fetch_player_args_convert_and_save(
        file_name_player=player_args_file
    )

    syzygy_table: table_base.SyzygyTable[Any] | None = create_syzygy(use_rust=use_rusty_board)

    print('create player from file')
    main_move_selector: move_selector.MoveSelector = move_selector.create_main_move_selector(
        move_selector_instance_or_args=args.main_move_selector,
        syzygy=syzygy_table,
        random_generator=random_generator,
        queue_progress_player=queue_progress_player
    )

    return Player(
        name=args.name,
        syzygy=syzygy_table,
        main_move_selector=main_move_selector
    )


def create_player(
        args: PlayerArgs,
        syzygy: SyzygyTable[Any] | None,
        random_generator: random.Random,
        queue_progress_player: queue.Queue[IsDataclass] | None = None
) -> Player:
    """Create a player object.

    This function creates a player object based on the provided arguments.

    Args:
        args (PlayerArgs): The arguments for creating the player.
        syzygy (SyzygyTable | None): The Syzygy table to be used by the player, or None if not available.
        random_generator (random.Random): The random number generator to be used by the player.

    Returns:
        Player: The created player object.
    """
    print('create player')
    main_move_selector: move_selector.MoveSelector = move_selector.create_main_move_selector(
        move_selector_instance_or_args=args.main_move_selector,
        syzygy=syzygy,
        random_generator=random_generator,
        queue_progress_player=queue_progress_player
    )

    player: Player = Player(
        name=args.name,
        syzygy=syzygy,
        main_move_selector=main_move_selector
    )

    return player


def create_game_player(
        player_factory_args: PlayerFactoryArgs,
        player_color: chess.Color,
        syzygy_table: table_base.SyzygyTable[Any] | None,
        queue_progress_player: queue.Queue[IsDataclass] | None
) -> GamePlayer:
    """Create a game player

    This function creates a game player using the provided player factory arguments and player color.

    Args:
        player_factory_args (PlayerFactoryArgs): The arguments for creating the player.
        player_color (chess.Color): The color of the player.

    Returns:
        GamePlayer: The created game player.
    """
    random_generator = random.Random(player_factory_args.seed)
    player: Player = create_player(
        args=player_factory_args.player_args,
        syzygy=syzygy_table,
        random_generator=random_generator,
        queue_progress_player=queue_progress_player
    )
    game_player: GamePlayer = GamePlayer(player, player_color)
    return game_player

"""
Module for creating players.
"""

import queue
import random
from typing import Any

import chess

import chipiron.players.move_selector.treevalue as treevalue
from chipiron.players.boardevaluators import table_base
from chipiron.players.boardevaluators.table_base.factory import create_syzygy
from chipiron.players.boardevaluators.table_base.syzygy_table import SyzygyTable
from chipiron.players.move_selector.treevalue.progress_monitor.progress_monitor import (
    StoppingCriterionTypes,
    TreeMoveLimitArgs,
)
from chipiron.players.player_args import PlayerArgs
from chipiron.players.player_ids import PlayerConfigTag
from chipiron.utils.logger import chipiron_logger

from ..environments.chess.board import BoardFactory, create_board_factory
from ..scripts.chipiron_args import ImplementationArgs
from ..utils.dataclass import IsDataclass
from . import move_selector
from .game_player import GamePlayer
from .player import Player
from .player_args import PlayerFactoryArgs


def create_chipiron_player(
    implementation_args: ImplementationArgs,
    universal_behavior: bool,
    random_generator: random.Random,
    queue_progress_player: queue.Queue[IsDataclass] | None = None,
    tree_move_limit: int | None = None,
) -> Player:
    """
    Creates the chipiron champion/representative/standard/default player

    Args:
        depth: int, the depth at which computation should be made.

    Returns: the player

    """
    syzygy_table: table_base.SyzygyTable[Any] | None = create_syzygy(
        use_rust=implementation_args.use_rust_boards
    )

    args_player: PlayerArgs = PlayerConfigTag.CHIPIRON.get_players_args()

    if tree_move_limit is not None:
        # todo find a prettier way to do this
        assert isinstance(
            args_player.main_move_selector, treevalue.TreeAndValuePlayerArgs
        )
        assert isinstance(
            args_player.main_move_selector.stopping_criterion, TreeMoveLimitArgs
        )

        args_player.main_move_selector.stopping_criterion.tree_move_limit = (
            tree_move_limit
        )

    main_move_selector: move_selector.MoveSelector | None = (
        move_selector.create_main_move_selector(
            move_selector_instance_or_args=args_player.main_move_selector,
            syzygy=syzygy_table,
            random_generator=random_generator,
            queue_progress_player=queue_progress_player,
        )
    )

    assert main_move_selector is not None

    board_factory: BoardFactory = create_board_factory(
        use_rust_boards=implementation_args.use_rust_boards,
        use_board_modification=implementation_args.use_board_modification,
        sort_legal_moves=universal_behavior,
    )

    return Player(
        name="chipiron",
        syzygy=syzygy_table,
        main_move_selector=main_move_selector,
        board_factory=board_factory,
    )


def create_player(
    args: PlayerArgs,
    syzygy: SyzygyTable[Any] | None,
    random_generator: random.Random,
    implementation_args: ImplementationArgs,
    universal_behavior: bool,
    queue_progress_player: queue.Queue[IsDataclass] | None = None,
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
    chipiron_logger.info("Create player")
    main_move_selector: move_selector.MoveSelector = (
        move_selector.create_main_move_selector(
            move_selector_instance_or_args=args.main_move_selector,
            syzygy=syzygy,
            random_generator=random_generator,
            queue_progress_player=queue_progress_player,
        )
    )

    board_factory: BoardFactory = create_board_factory(
        use_rust_boards=implementation_args.use_rust_boards,
        use_board_modification=implementation_args.use_board_modification,
        sort_legal_moves=universal_behavior,
    )

    player: Player = Player(
        name=args.name,
        syzygy=syzygy,
        main_move_selector=main_move_selector,
        board_factory=board_factory,
    )

    return player


def create_game_player(
    player_factory_args: PlayerFactoryArgs,
    player_color: chess.Color,
    syzygy_table: table_base.SyzygyTable[Any] | None,
    queue_progress_player: queue.Queue[IsDataclass] | None,
    implementation_args: ImplementationArgs,
    universal_behavior: bool,
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
        queue_progress_player=queue_progress_player,
        implementation_args=implementation_args,
        universal_behavior=universal_behavior,
    )
    game_player: GamePlayer = GamePlayer(player, player_color)
    return game_player

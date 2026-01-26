"""
Module for creating players.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from anemone import TreeAndValuePlayerArgs
from anemone.progress_monitor.progress_monitor import (
    TreeBranchLimitArgs,
)
from atomheart.board import BoardFactory, create_board_factory
from atomheart.board.utils import FenPlusHistory
from valanga import Color

from chipiron.environments.chess.types import ChessState
from chipiron.players.adapters.chess_adapter import ChessAdapter
from chipiron.players.adapters.chess_syzygy_oracle import ChessSyzygyOracle
from chipiron.players.boardevaluators.master_board_evaluator import (
    create_master_board_evaluator,
)
from chipiron.players.boardevaluators.table_base.factory import (
    AnySyzygyTable,
    create_syzygy,
)
from chipiron.players.player_args import PlayerArgs
from chipiron.players.player_ids import PlayerConfigTag
from chipiron.utils.logger import chipiron_logger

from ..scripts.chipiron_args import ImplementationArgs
from ..utils.dataclass import IsDataclass
from ..utils.queue_protocols import PutQueue
from . import move_selector
from .game_player import GamePlayer
from .player import Player
from .player_args import PlayerFactoryArgs

if TYPE_CHECKING:
    from valanga.policy import BranchSelector


@dataclass
class PlayerCreationArgs:
    random_generator: random.Random
    implementation_args: ImplementationArgs
    universal_behavior: bool
    queue_progress_player: PutQueue[IsDataclass] | None = None
    syzygy: AnySyzygyTable | None = None


def create_chipiron_player(
    implementation_args: ImplementationArgs,
    universal_behavior: bool,
    random_generator: random.Random,
    queue_progress_player: PutQueue[IsDataclass] | None = None,
    tree_branch_limit: int | None = None,
) -> Player[FenPlusHistory, ChessState]:
    """
    Creates the chipiron champion/representative/standard/default player

    Args:
        depth: int, the depth at which computation should be made.

    Returns: the player

    """
    syzygy_table: AnySyzygyTable | None = create_syzygy(
        use_rust=implementation_args.use_rust_boards
    )

    args_player: PlayerArgs = PlayerConfigTag.CHIPIRON.get_players_args()

    if tree_branch_limit is not None:
        # todo find a prettier way to do this
        assert isinstance(args_player.main_move_selector, TreeAndValuePlayerArgs)
        assert isinstance(
            args_player.main_move_selector.stopping_criterion, TreeBranchLimitArgs
        )

        args_player.main_move_selector.stopping_criterion.tree_branch_limit = (
            tree_branch_limit
        )

    if isinstance(args_player.main_move_selector, TreeAndValuePlayerArgs):
        master_state_evaluator = create_master_board_evaluator(
            board_evaluator=args_player.main_move_selector.board_evaluator,
            syzygy=syzygy_table,
            evaluation_scale=args_player.main_move_selector.evaluation_scale,
        )
        main_move_selector: BranchSelector[ChessState] | None = (
            move_selector.create_tree_and_value_move_selector(
                args_player.main_move_selector,
                state_type=ChessState,
                master_state_evaluator=master_state_evaluator,
                state_representation_factory=None,
                random_generator=random_generator,
                queue_progress_player=queue_progress_player,
            )
        )
    else:
        main_move_selector = move_selector.create_main_move_selector(
            args_player.main_move_selector,
            random_generator=random_generator,
        )

    assert main_move_selector is not None

    board_factory: BoardFactory = create_board_factory(
        use_rust_boards=implementation_args.use_rust_boards,
        use_board_modification=implementation_args.use_board_modification,
        sort_legal_moves=universal_behavior,
    )

    oracle = ChessSyzygyOracle(syzygy_table) if syzygy_table is not None else None
    adapter = ChessAdapter(
        board_factory=board_factory,
        main_move_selector=main_move_selector,
        oracle=oracle,
    )
    return Player[FenPlusHistory, ChessState](name="chipiron", adapter=adapter)


def create_player(
    args: PlayerArgs,
    syzygy: AnySyzygyTable | None,
    random_generator: random.Random,
    implementation_args: ImplementationArgs,
    universal_behavior: bool,
    queue_progress_player: PutQueue[IsDataclass] | None = None,
) -> Player[FenPlusHistory, ChessState]:
    """Create a player object.

    This function creates a player object based on the provided arguments.

    Args:
        args (PlayerArgs): The arguments for creating the player.
        syzygy (AnySyzygyTable | None): The Syzygy table to be used by the player, or None if not available.
        random_generator (random.Random): The random number generator to be used by the player.

    Returns:
        Player: The created player object.
    """
    chipiron_logger.debug("Create player")
    if isinstance(args.main_move_selector, TreeAndValuePlayerArgs):
        master_state_evaluator = create_master_board_evaluator(
            board_evaluator=args.main_move_selector.board_evaluator,
            syzygy=syzygy,
            evaluation_scale=args.main_move_selector.evaluation_scale,
        )
        main_move_selector: BranchSelector[ChessState] = (
            move_selector.create_tree_and_value_move_selector(
                args.main_move_selector,
                state_type=ChessState,
                master_state_evaluator=master_state_evaluator,
                state_representation_factory=None,
                random_generator=random_generator,
                queue_progress_player=queue_progress_player,
            )
        )
    else:
        main_move_selector = move_selector.create_main_move_selector(
            args.main_move_selector,
            random_generator=random_generator,
        )

    board_factory: BoardFactory = create_board_factory(
        use_rust_boards=implementation_args.use_rust_boards,
        use_board_modification=implementation_args.use_board_modification,
        sort_legal_moves=universal_behavior,
    )

    oracle = ChessSyzygyOracle(syzygy) if syzygy is not None else None
    adapter = ChessAdapter(
        board_factory=board_factory,
        main_move_selector=main_move_selector,
        oracle=oracle,
    )

    return Player[FenPlusHistory, ChessState](name=args.name, adapter=adapter)


def create_game_player(
    player_factory_args: PlayerFactoryArgs,
    player_color: Color,
    syzygy_table: AnySyzygyTable | None,
    queue_progress_player: PutQueue[IsDataclass] | None,
    implementation_args: ImplementationArgs,
    universal_behavior: bool,
) -> GamePlayer[FenPlusHistory, ChessState]:
    """Create a game player

    This function creates a game player using the provided player factory arguments and player color.

    Args:
        player_factory_args (PlayerFactoryArgs): The arguments for creating the player.
        player_color (Color): The color of the player.

    Returns:
        GamePlayer: The created game player.
    """
    random_generator = random.Random(player_factory_args.seed)
    player: Player[FenPlusHistory, ChessState] = create_player(
        args=player_factory_args.player_args,
        syzygy=syzygy_table,
        random_generator=random_generator,
        queue_progress_player=queue_progress_player,
        implementation_args=implementation_args,
        universal_behavior=universal_behavior,
    )
    game_player: GamePlayer[FenPlusHistory, ChessState] = GamePlayer(
        player, player_color
    )
    return game_player

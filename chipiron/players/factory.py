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
from chipiron.players.adapters.chess_syzygy_oracle import (
    ChessSyzygyPolicyOracle,
    ChessSyzygyTerminalOracle,
    ChessSyzygyValueOracle,
)
from chipiron.players.boardevaluators.master_board_evaluator import (
    create_master_state_evaluator,
    create_master_state_evaluator_from_args,
)
from chipiron.players.boardevaluators.table_base.factory import (
    AnySyzygyTable,
    create_syzygy,
)
from chipiron.players.move_selector.tree_and_value_args import TreeAndValueChipironArgs
from chipiron.players.oracles import PolicyOracle, TerminalOracle, ValueOracle
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
    policy_oracle: PolicyOracle[ChessState] | None = None
    value_oracle: ValueOracle[ChessState] | None = None
    terminal_oracle: TerminalOracle[ChessState] | None = None


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
    policy_oracle = (
        ChessSyzygyPolicyOracle(syzygy_table) if syzygy_table is not None else None
    )
    value_oracle = (
        ChessSyzygyValueOracle(syzygy_table) if syzygy_table is not None else None
    )
    terminal_oracle = (
        ChessSyzygyTerminalOracle(syzygy_table) if syzygy_table is not None else None
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

    main_move_selector: BranchSelector[ChessState]
    match args_player.main_move_selector:
        case TreeAndValuePlayerArgs() as tree_args:
            master_state_evaluator = create_master_state_evaluator(
                board_evaluator=tree_args.board_evaluator,
                value_oracle=value_oracle,
                terminal_oracle=terminal_oracle,
                evaluation_scale=tree_args.evaluation_scale,
            )
            main_move_selector = move_selector.create_tree_and_value_move_selector(
                tree_args,
                state_type=ChessState,
                master_state_evaluator=master_state_evaluator,
                state_representation_factory=None,
                random_generator=random_generator,
                queue_progress_player=queue_progress_player,
            )
        case _:
            main_move_selector = move_selector.create_main_move_selector(
                args_player.main_move_selector,
                random_generator=random_generator,
            )

    board_factory: BoardFactory = create_board_factory(
        use_rust_boards=implementation_args.use_rust_boards,
        use_board_modification=implementation_args.use_board_modification,
        sort_legal_moves=universal_behavior,
    )

    adapter = ChessAdapter(
        board_factory=board_factory,
        main_move_selector=main_move_selector,
        oracle=policy_oracle,
    )
    return Player[FenPlusHistory, ChessState](name="chipiron", adapter=adapter)


def create_player(
    args: PlayerArgs,
    policy_oracle: PolicyOracle[ChessState] | None,
    value_oracle: ValueOracle[ChessState] | None,
    terminal_oracle: TerminalOracle[ChessState] | None,
    random_generator: random.Random,
    implementation_args: ImplementationArgs,
    universal_behavior: bool,
    queue_progress_player: PutQueue[IsDataclass] | None = None,
) -> Player[FenPlusHistory, ChessState]:
    """Create a player object.

    This function creates a player object based on the provided arguments.

    Args:
        args (PlayerArgs): The arguments for creating the player.
        policy_oracle (PolicyOracle | None): The policy oracle for best-move shortcuts.
        value_oracle (ValueOracle | None): The value oracle for exact evaluations.
        terminal_oracle (TerminalOracle | None): The terminal oracle for endgame metadata.
        random_generator (random.Random): The random number generator to be used by the player.

    Returns:
        Player: The created player object.
    """
    chipiron_logger.debug("Create player")
    main_move_selector: BranchSelector[ChessState]
    match args.main_move_selector:
        case TreeAndValueChipironArgs() as tree_args:
            master_state_evaluator = create_master_state_evaluator_from_args(
                master_board_evaluator=tree_args.evaluator_args,
                value_oracle=value_oracle,
                terminal_oracle=terminal_oracle,
            )
            main_move_selector = move_selector.create_tree_and_value_move_selector(
                args=tree_args.anemone_args,
                state_type=ChessState,
                master_state_evaluator=master_state_evaluator,
                state_representation_factory=None,
                random_generator=random_generator,
                queue_progress_player=queue_progress_player,
            )
        case _:
            main_move_selector = move_selector.create_main_move_selector(
                args.main_move_selector,
                random_generator=random_generator,
            )

    board_factory: BoardFactory = create_board_factory(
        use_rust_boards=implementation_args.use_rust_boards,
        use_board_modification=implementation_args.use_board_modification,
        sort_legal_moves=universal_behavior,
    )

    adapter = ChessAdapter(
        board_factory=board_factory,
        main_move_selector=main_move_selector,
        oracle=policy_oracle,
    )

    return Player[FenPlusHistory, ChessState](name=args.name, adapter=adapter)


def create_game_player(
    player_factory_args: PlayerFactoryArgs,
    player_color: Color,
    policy_oracle: PolicyOracle[ChessState] | None,
    value_oracle: ValueOracle[ChessState] | None,
    terminal_oracle: TerminalOracle[ChessState] | None,
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
        policy_oracle=policy_oracle,
        value_oracle=value_oracle,
        terminal_oracle=terminal_oracle,
        random_generator=random_generator,
        queue_progress_player=queue_progress_player,
        implementation_args=implementation_args,
        universal_behavior=universal_behavior,
    )
    game_player: GamePlayer[FenPlusHistory, ChessState] = GamePlayer(
        player, player_color
    )
    return game_player

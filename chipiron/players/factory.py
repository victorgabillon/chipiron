"""Module for creating players."""

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
from valanga.policy import BranchSelector

from chipiron.environments.chess.types import ChessState
from chipiron.players.adapters.chess_adapter import ChessAdapter
from chipiron.players.adapters.chess_syzygy_oracle import (
    ChessSyzygyPolicyOracle,
    ChessSyzygyTerminalOracle,
    ChessSyzygyValueOracle,
)
from chipiron.players.boardevaluators.anemone_adapter import (
    MasterBoardEvaluatorAsAnemone,
    MasterBoardOverEventDetector,
)
from chipiron.players.boardevaluators.master_board_evaluator import (
    MasterBoardEvaluatorArgs,
    create_master_state_evaluator_from_args,
)
from chipiron.players.boardevaluators.table_base.factory import (
    AnySyzygyTable,
    create_syzygy,
)
from chipiron.players.chess_player_args import (
    ChessPlayerArgs,
    ChessPlayerFactoryArgs,
)
from chipiron.players.move_selector.tree_and_value_args import TreeAndValueAppArgs
from chipiron.players.oracles import PolicyOracle, TerminalOracle, ValueOracle
from chipiron.players.player_ids import PlayerConfigTag
from chipiron.utils.communication.mailbox import MainMailboxMessage
from chipiron.utils.logger import chipiron_logger

from ..scripts.chipiron_args import ImplementationArgs
from ..utils.dataclass import IsDataclass
from ..utils.queue_protocols import PutQueue
from . import move_selector
from .factory_pipeline import create_player_with_pipeline
from .game_player import GamePlayer
from .player import Player

if TYPE_CHECKING:
    from anemone.node_evaluation.node_direct_evaluation.node_direct_evaluator import (
        MasterStateEvaluator,
    )


@dataclass
class PlayerCreationArgs:
    """Data container for PlayerCreationArgs."""

    random_generator: random.Random
    implementation_args: ImplementationArgs
    universal_behavior: bool
    queue_progress_player: PutQueue[IsDataclass] | None = None
    policy_oracle: PolicyOracle[ChessState] | None = None
    value_oracle: ValueOracle[ChessState] | None = None
    terminal_oracle: TerminalOracle[ChessState] | None = None


def create_tag_player(
    tag: PlayerConfigTag,
    implementation_args: ImplementationArgs,
    universal_behavior: bool,
    random_generator: random.Random,
    queue_progress_player: PutQueue[IsDataclass] | None = None,
    tree_branch_limit: int | None = None,
) -> Player[FenPlusHistory, ChessState]:
    """Create the chipiron champion/representative/standard/default player.

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

    args_player: ChessPlayerArgs = tag.get_players_args()

    if tree_branch_limit is not None:
        # todo find a prettier way to do this
        assert isinstance(args_player.main_move_selector, TreeAndValueAppArgs)
        assert isinstance(
            args_player.main_move_selector.anemone_args, TreeAndValuePlayerArgs
        )
        assert isinstance(
            args_player.main_move_selector.anemone_args.stopping_criterion,
            TreeBranchLimitArgs,
        )

        args_player.main_move_selector.anemone_args.stopping_criterion.tree_branch_limit = tree_branch_limit

    return create_chess_player(
        args=args_player,
        policy_oracle=policy_oracle,
        value_oracle=value_oracle,
        terminal_oracle=terminal_oracle,
        random_generator=random_generator,
        implementation_args=implementation_args,
        universal_behavior=universal_behavior,
        queue_progress_player=queue_progress_player,
    )


def create_chipiron_player(
    implementation_args: ImplementationArgs,
    universal_behavior: bool,
    random_generator: random.Random,
    queue_progress_player: PutQueue[IsDataclass] | None = None,
    tree_branch_limit: int | None = None,
) -> Player[FenPlusHistory, ChessState]:
    """Create the Chipiron champion/representative/default player.

    This function creates the default Chipiron player using the provided
    implementation arguments and random generator.

    Args:
        implementation_args (ImplementationArgs): The implementation arguments.
        universal_behavior (bool): Whether to use universal behavior.
        random_generator (random.Random): The random number generator.

    Returns:
        Player: The created Chipiron player.

    """
    return create_tag_player(
        tag=PlayerConfigTag.CHIPIRON,
        implementation_args=implementation_args,
        universal_behavior=universal_behavior,
        random_generator=random_generator,
        queue_progress_player=queue_progress_player,
        tree_branch_limit=tree_branch_limit,
    )


def create_chess_player(
    args: ChessPlayerArgs,
    policy_oracle: PolicyOracle[ChessState] | None,
    value_oracle: ValueOracle[ChessState] | None,
    terminal_oracle: TerminalOracle[ChessState] | None,
    random_generator: random.Random,
    implementation_args: ImplementationArgs,
    universal_behavior: bool,
    queue_progress_player: PutQueue[MainMailboxMessage] | None = None,
) -> Player[FenPlusHistory, ChessState]:
    """Create a chess player object.

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

    def master_evaluator_from_args(
        evaluator_args: MasterBoardEvaluatorArgs,
        value_oracle_in: ValueOracle[ChessState] | None,
        terminal_oracle_in: TerminalOracle[ChessState] | None,
    ) -> "MasterStateEvaluator":
        master_board_evaluator = create_master_state_evaluator_from_args(
            master_board_evaluator=evaluator_args,
            value_oracle=value_oracle_in,
            terminal_oracle=terminal_oracle_in,
        )
        return MasterBoardEvaluatorAsAnemone(
            inner=master_board_evaluator,
            over=MasterBoardOverEventDetector(master_board_evaluator),
        )

    def adapter_builder(
        selector: BranchSelector[ChessState],
        policy_oracle_in: PolicyOracle[ChessState] | None,
    ) -> ChessAdapter:
        board_factory: BoardFactory = create_board_factory(
            use_rust_boards=implementation_args.use_rust_boards,
            use_board_modification=implementation_args.use_board_modification,
            sort_legal_moves=universal_behavior,
        )
        return ChessAdapter(
            board_factory=board_factory,
            main_move_selector=selector,
            oracle=policy_oracle_in,
        )

    return create_player_with_pipeline(
        name=args.name,
        main_selector_args=args.main_move_selector,
        state_type=ChessState,
        policy_oracle=policy_oracle,
        value_oracle=value_oracle,
        terminal_oracle=terminal_oracle,
        master_evaluator_from_args=master_evaluator_from_args,
        adapter_builder=adapter_builder,
        create_non_tree_selector=lambda selector_args: (
            move_selector.create_main_move_selector(
                selector_args,
                random_generator=random_generator,
            )
        ),
        random_generator=random_generator,
        queue_progress_player=queue_progress_player,
    )


def create_player(
    args: ChessPlayerArgs,
    syzygy: AnySyzygyTable | None,
    random_generator: random.Random,
    implementation_args: ImplementationArgs,
    universal_behavior: bool,
    queue_progress_player: PutQueue[IsDataclass] | None = None,
) -> Player[FenPlusHistory, ChessState]:
    """Compatibility wrapper for chess player creation using a Syzygy table.

    Deprecated: chess-only wrapper, prefer ``create_chess_player``.
    """
    policy_oracle = ChessSyzygyPolicyOracle(syzygy) if syzygy is not None else None
    value_oracle = ChessSyzygyValueOracle(syzygy) if syzygy is not None else None
    terminal_oracle = ChessSyzygyTerminalOracle(syzygy) if syzygy is not None else None
    return create_chess_player(
        args=args,
        policy_oracle=policy_oracle,
        value_oracle=value_oracle,
        terminal_oracle=terminal_oracle,
        random_generator=random_generator,
        implementation_args=implementation_args,
        universal_behavior=universal_behavior,
        queue_progress_player=queue_progress_player,
    )


def create_game_player(
    player_factory_args: ChessPlayerFactoryArgs,
    player_color: Color,
    policy_oracle: PolicyOracle[ChessState] | None,
    value_oracle: ValueOracle[ChessState] | None,
    terminal_oracle: TerminalOracle[ChessState] | None,
    queue_progress_player: PutQueue[MainMailboxMessage] | None,
    implementation_args: ImplementationArgs,
    universal_behavior: bool,
) -> GamePlayer[FenPlusHistory, ChessState]:
    """Create a game player.

    This function creates a game player using the provided player factory arguments and player color.

    Args:
        player_factory_args (PlayerFactoryArgs): The arguments for creating the player.
        player_color (Color): The color of the player.

    Returns:
        GamePlayer: The created game player.

    """
    random_generator = random.Random(player_factory_args.seed)
    player: Player[FenPlusHistory, ChessState] = create_chess_player(
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

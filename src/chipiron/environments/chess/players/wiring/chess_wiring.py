"""Module for chess wiring."""

from dataclasses import dataclass

from atomheart.games.chess.board.utils import FenPlusHistory
from valanga import Color

from chipiron.environments.chess.types import ChessState
from chipiron.environments.chess.players.oracles.chess_syzygy_oracle import (
    ChessSyzygyPolicyOracle,
    ChessSyzygyTerminalOracle,
    ChessSyzygyValueOracle,
)
from chipiron.environments.chess.players.evaluators.boardevaluators.table_base.factory import (
    AnySyzygyTable,
    create_syzygy_factory,
)
from chipiron.environments.chess.players.args.chess_player_args import ChessPlayerFactoryArgs
from chipiron.environments.chess.players.factory.factory import create_game_player
from chipiron.players.game_player import GamePlayer
from chipiron.players.observer_wiring import ObserverWiring
from chipiron.scripts.chipiron_args import ImplementationArgs


@dataclass(frozen=True)
class BuildChessGamePlayerArgs:
    """Buildchessgameplayerargs implementation."""

    player_factory_args: ChessPlayerFactoryArgs
    player_color: Color
    implementation_args: ImplementationArgs
    universal_behavior: bool
    syzygy_table: AnySyzygyTable | None = None


def build_chess_game_player(
    args: BuildChessGamePlayerArgs,
) -> GamePlayer[FenPlusHistory, ChessState]:
    """Build chess game player."""
    policy_oracle = None
    value_oracle = None
    terminal_oracle = None

    if args.player_factory_args.player_args.oracle_play:
        create_syzygy = create_syzygy_factory(
            use_rust=args.implementation_args.use_rust_boards
        )
        syzygy = args.syzygy_table if args.syzygy_table is not None else create_syzygy()
        policy_oracle = ChessSyzygyPolicyOracle(syzygy) if syzygy is not None else None
        value_oracle = ChessSyzygyValueOracle(syzygy) if syzygy is not None else None
        terminal_oracle = (
            ChessSyzygyTerminalOracle(syzygy) if syzygy is not None else None
        )
    return create_game_player(
        player_factory_args=args.player_factory_args,
        player_color=args.player_color,
        policy_oracle=policy_oracle,
        value_oracle=value_oracle,
        terminal_oracle=terminal_oracle,
        implementation_args=args.implementation_args,
        universal_behavior=args.universal_behavior,
    )


CHESS_WIRING: ObserverWiring[FenPlusHistory, ChessState, BuildChessGamePlayerArgs] = (
    ObserverWiring(
        build_game_player=build_chess_game_player,
        build_args_type=BuildChessGamePlayerArgs,
    )
)

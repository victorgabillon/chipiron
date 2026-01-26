from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from chipiron.players.boardevaluators.table_base.factory import (
    AnySyzygyTable,
    create_syzygy_factory,
)
from chipiron.players.adapters.chess_syzygy_oracle import (
    ChessSyzygyPolicyOracle,
    ChessSyzygyTerminalOracle,
    ChessSyzygyValueOracle,
)
from chipiron.players.factory import create_game_player
from chipiron.players.observer_wiring import ObserverWiring

if TYPE_CHECKING:
    from atomheart.board.utils import FenPlusHistory
    from valanga import Color

    from chipiron.players.game_player import GamePlayer
    from chipiron.players.player_args import PlayerFactoryArgs
    from chipiron.environments.chess.types import ChessState
    from chipiron.scripts.chipiron_args import ImplementationArgs
    from chipiron.utils.dataclass import IsDataclass
    from chipiron.utils.queue_protocols import PutQueue


@dataclass(frozen=True)
class BuildChessGamePlayerArgs:
    player_factory_args: PlayerFactoryArgs
    player_color: Color
    implementation_args: ImplementationArgs
    universal_behavior: bool
    syzygy_table: AnySyzygyTable | None = None


def build_chess_game_player(
    args: BuildChessGamePlayerArgs,
    queue_out: PutQueue[IsDataclass],
) -> GamePlayer[FenPlusHistory, ChessState]:
    create_syzygy = create_syzygy_factory(
        use_rust=args.implementation_args.use_rust_boards
    )
    syzygy = args.syzygy_table if args.syzygy_table is not None else create_syzygy()
    policy_oracle = (
        ChessSyzygyPolicyOracle(syzygy) if syzygy is not None else None
    )
    value_oracle = ChessSyzygyValueOracle(syzygy) if syzygy is not None else None
    terminal_oracle = ChessSyzygyTerminalOracle(syzygy) if syzygy is not None else None
    return create_game_player(
        player_factory_args=args.player_factory_args,
        player_color=args.player_color,
        policy_oracle=policy_oracle,
        value_oracle=value_oracle,
        terminal_oracle=terminal_oracle,
        queue_progress_player=queue_out,
        implementation_args=args.implementation_args,
        universal_behavior=args.universal_behavior,
    )


CHESS_WIRING: ObserverWiring[
    FenPlusHistory, ChessState, BuildChessGamePlayerArgs
] = (
    ObserverWiring(
        build_game_player=build_chess_game_player,
        build_args_type=BuildChessGamePlayerArgs,
    )
)

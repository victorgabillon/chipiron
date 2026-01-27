from __future__ import annotations

from typing import TYPE_CHECKING

from atomheart import ValangaChessState
from atomheart.board.utils import FenPlusHistory
from valanga import StateTag

from chipiron.environments.chess.chess_gui_encoder import ChessGuiEncoder
from chipiron.environments.chess.chess_rules import ChessRules
from chipiron.environments.chess.tags import ChessStartTag
from chipiron.environments.chess.types import ChessState
from chipiron.environments.deps import ChessEnvironmentDeps
from chipiron.environments.environment import Environment
from chipiron.environments.types import GameKind
from chipiron.players.communications.player_request_encoder import (
    ChessPlayerRequestEncoder,
)
from chipiron.scripts.chipiron_args import ImplementationArgs

if TYPE_CHECKING:
    from chipiron.players.factory_higher_level import PlayerObserverFactory


def make_chess_environment(
    *,
    deps: ChessEnvironmentDeps,
) -> Environment[ChessState, FenPlusHistory, ChessStartTag]:
    from chipiron.players.factory_higher_level import create_player_observer_factory

    def build_player_observer_factory(
        *,
        each_player_has_its_own_thread: bool,
        implementation_args: ImplementationArgs,
        universal_behavior: bool,
    ) -> "PlayerObserverFactory":
        return create_player_observer_factory(
            game_kind=GameKind.CHESS,
            each_player_has_its_own_thread=each_player_has_its_own_thread,
            implementation_args=implementation_args,
            universal_behavior=universal_behavior,
        )

    def normalize_start_tag(tag: StateTag) -> ChessStartTag:
        if not isinstance(tag, ChessStartTag):
            raise TypeError("Chess environment expects ChessStartTag")
        return tag

    def make_initial_state(tag: ChessStartTag) -> ChessState:
        return ValangaChessState(
            deps.board_factory(fen_with_history=FenPlusHistory(current_fen=tag.fen))
        )

    return Environment(
        game_kind=GameKind.CHESS,
        rules=ChessRules(syzygy=deps.syzygy_table),
        gui_encoder=ChessGuiEncoder(),
        player_encoder=ChessPlayerRequestEncoder(),
        make_player_observer_factory=build_player_observer_factory,
        normalize_start_tag=normalize_start_tag,
        make_initial_state=make_initial_state,
    )

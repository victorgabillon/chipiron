"""Module for environment."""

from typing import TYPE_CHECKING

from atomheart import ChessDynamics
from atomheart.games.chess.board.utils import FenPlusHistory
from valanga import StateTag

from chipiron.environments.base import Environment
from chipiron.environments.chess.chess_gui_encoder import ChessGuiEncoder
from chipiron.environments.chess.chess_rules import ChessRules
from chipiron.environments.chess.tags import ChessStartTag
from chipiron.environments.chess.types import ChessState
from chipiron.environments.deps import ChessEnvironmentDeps
from chipiron.environments.types import GameKind
from chipiron.players.communications.player_request_encoder import (
    ChessPlayerRequestEncoder,
)
from chipiron.players.factory_higher_level import create_player_observer_factory
from chipiron.scripts.chipiron_args import ImplementationArgs

if TYPE_CHECKING:
    from chipiron.players.factory_higher_level import PlayerObserverFactory


class ChessEnvironmentError(TypeError):
    """Base error for chess environment configuration issues."""


class ChessStartTagTypeError(ChessEnvironmentError):
    """Raised when an invalid start tag is provided."""

    def __init__(self, tag: StateTag) -> None:
        """Initialize the error with the invalid tag."""
        super().__init__(f"Chess environment expects ChessStartTag, got {type(tag)!r}")


def make_chess_environment(
    *,
    deps: ChessEnvironmentDeps,
) -> Environment[ChessState, FenPlusHistory, ChessStartTag]:
    """Create chess environment."""

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
            raise ChessStartTagTypeError(tag)
        return tag

    def make_initial_state(tag: ChessStartTag) -> ChessState:
        return ChessState(
            deps.board_factory(fen_with_history=FenPlusHistory(current_fen=tag.fen))
        )

    dynamics = ChessDynamics()

    return Environment(
        game_kind=GameKind.CHESS,
        rules=ChessRules(syzygy=deps.syzygy_table),
        dynamics=dynamics,
        gui_encoder=ChessGuiEncoder(),
        player_encoder=ChessPlayerRequestEncoder(),
        make_player_observer_factory=build_player_observer_factory,
        normalize_start_tag=normalize_start_tag,
        make_initial_state=make_initial_state,
    )

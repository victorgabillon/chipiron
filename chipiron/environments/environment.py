from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast

from atomheart import ValangaChessState
from valanga import StateTag

from chipiron.environments.chess.types import ChessState
from chipiron.environments.types import GameKind
from chipiron.games.game.game_rules import GameRules
from chipiron.players.boardevaluators.table_base.factory import AnySyzygyTable
from chipiron.players.communications.player_request_encoder import PlayerRequestEncoder
from chipiron.utils.communication.gui_encoder import GuiEncoder
from chipiron.scripts.chipiron_args import ImplementationArgs

if TYPE_CHECKING:
    from chipiron.players.factory_higher_level import PlayerObserverFactory
    import atomheart.board as boards
    from atomheart.board.utils import FenPlusHistory

StateT = TypeVar("StateT", covariant=True)
StateSnapT = TypeVar("StateSnapT")

StartTagT = TypeVar("StartTagT", covariant=True)  # produced by normalize_start_tag
StartTagInT = TypeVar(
    "StartTagInT", contravariant=True
)  # consumed by make_initial_state


class TagNormalizer(Protocol[StartTagT]):
    def __call__(self, tag: StateTag) -> StartTagT: ...


class InitialStateFactory(Protocol[StateT, StartTagInT]):
    def __call__(self, tag: StartTagInT) -> StateT: ...


class PlayerObserverFactoryBuilder(Protocol):
    def __call__(
        self,
        *,
        each_player_has_its_own_thread: bool,
        implementation_args: ImplementationArgs,
        universal_behavior: bool,
    ) -> "PlayerObserverFactory": ...


class ChessBoardFactory(Protocol):
    def __call__(self, *, fen_with_history: "FenPlusHistory") -> "boards.IBoard": ...


@dataclass(frozen=True)
class Environment[StateT, StateSnapT, StartTagT]:
    game_kind: GameKind
    rules: GameRules[StateT]
    gui_encoder: GuiEncoder[StateT]
    player_encoder: PlayerRequestEncoder[StateT, StateSnapT]
    make_player_observer_factory: PlayerObserverFactoryBuilder
    normalize_start_tag: TagNormalizer[StartTagT]
    make_initial_state: InitialStateFactory[StateT, StartTagT]


@dataclass(frozen=True)
class EnvironmentDeps:
    board_factory: Any | None = None


def make_environment(
    *,
    game_kind: GameKind,
    syzygy_table: AnySyzygyTable | None,
    deps: EnvironmentDeps,
) -> Environment[Any, Any, Any]:
    match game_kind:
        case GameKind.CHESS:
            if deps.board_factory is None:
                raise ValueError("board_factory is required for chess environments")
            board_factory = cast(ChessBoardFactory, deps.board_factory)

            from atomheart.board.utils import FenPlusHistory

            from chipiron.environments.chess.chess_gui_encoder import ChessGuiEncoder
            from chipiron.environments.chess.chess_rules import ChessRules
            from chipiron.environments.chess.tags import ChessStartTag
            from chipiron.players.communications.player_request_encoder import (
                ChessPlayerRequestEncoder,
            )
            from chipiron.players.factory_higher_level import (
                create_player_observer_factory,
            )

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
                    syzygy_table=syzygy_table,
                )

            def normalize_start_tag(tag: StateTag) -> ChessStartTag:
                if not isinstance(tag, ChessStartTag):
                    raise TypeError(
                        "Chess environment expects a ChessStartTag for initial state."
                    )
                return tag

            def make_initial_state(tag: ChessStartTag) -> ChessState:
                return ValangaChessState(
                    board_factory(fen_with_history=FenPlusHistory(current_fen=tag.fen))
                )

            return Environment[ChessState, FenPlusHistory, ChessStartTag](
                game_kind=game_kind,
                rules=ChessRules(syzygy=syzygy_table),
                gui_encoder=ChessGuiEncoder(),
                player_encoder=ChessPlayerRequestEncoder(),
                make_player_observer_factory=build_player_observer_factory,
                normalize_start_tag=normalize_start_tag,
                make_initial_state=make_initial_state,
            )
        case GameKind.CHECKERS:
            raise NotImplementedError("Environment for CHECKERS is not implemented yet")
        case _:
            raise ValueError(f"No Environment for game_kind={game_kind!r}")

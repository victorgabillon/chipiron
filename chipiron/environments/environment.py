"""Environment wiring and factory helpers."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar, overload

from atomheart.board.utils import FenPlusHistory
from valanga import StateTag

from chipiron.environments.chess.tags import ChessStartTag
from chipiron.environments.chess.types import ChessState
from chipiron.environments.deps import CheckersEnvironmentDeps, ChessEnvironmentDeps
from chipiron.environments.types import GameKind
from chipiron.games.game.game_rules import GameRules
from chipiron.players.communications.player_request_encoder import PlayerRequestEncoder
from chipiron.scripts.chipiron_args import ImplementationArgs
from chipiron.utils.communication.gui_encoder import GuiEncoder

if TYPE_CHECKING:
    from chipiron.players.factory_higher_level import PlayerObserverFactory


class EnvironmentCreationError(ValueError):
    """Base error for environment creation failures."""


class EnvironmentNotFoundError(EnvironmentCreationError):
    """Raised when an environment implementation is missing."""

    def __init__(self, game_kind: GameKind) -> None:
        super().__init__(f"No Environment for game_kind={game_kind!r}")

StateT = TypeVar("StateT")
StateSnapT = TypeVar("StateSnapT")

StartTagT = TypeVar("StartTagT")  # produced by normalize_start_tag
StartTagOutT = TypeVar("StartTagOutT", covariant=True)
StartTagInT = TypeVar(
    "StartTagInT", contravariant=True
)  # consumed by make_initial_state
StateOutT = TypeVar("StateOutT", covariant=True)


class TagNormalizer(Protocol[StartTagOutT]):
    """Protocol for normalizing raw start tags."""

    def __call__(self, tag: StateTag) -> StartTagOutT:
        """Normalize a start tag to the environment's expected type."""
        ...


class InitialStateFactory(Protocol[StateOutT, StartTagInT]):
    """Protocol for creating initial states from start tags."""

    def __call__(self, tag: StartTagInT) -> StateOutT:
        """Create the initial state from a normalized start tag."""
        ...


class PlayerObserverFactoryBuilder(Protocol):
    """Protocol for building player observer factories."""

    def __call__(
        self,
        *,
        each_player_has_its_own_thread: bool,
        implementation_args: ImplementationArgs,
        universal_behavior: bool,
    ) -> "PlayerObserverFactory":
        """Build a player observer factory with the given configuration."""
        ...


@dataclass(frozen=True)
class Environment[StateT, StateSnapT, StartTagT]:
    """Bundle environment wiring and factories for a game kind."""

    game_kind: GameKind
    rules: GameRules[StateT]
    gui_encoder: GuiEncoder[StateT]
    player_encoder: PlayerRequestEncoder[StateT, StateSnapT]
    make_player_observer_factory: PlayerObserverFactoryBuilder
    normalize_start_tag: TagNormalizer[StartTagT]
    make_initial_state: InitialStateFactory[StateT, StartTagT]


EnvDeps = ChessEnvironmentDeps | CheckersEnvironmentDeps


@overload
def make_environment(
    *,
    game_kind: Literal[GameKind.CHESS],
    deps: ChessEnvironmentDeps,
) -> Environment[ChessState, FenPlusHistory, ChessStartTag]: ...
@overload
def make_environment(
    *,
    game_kind: Literal[GameKind.CHECKERS],
    deps: CheckersEnvironmentDeps,
) -> Environment[object, object, object]: ...
@overload
def make_environment(
    *,
    game_kind: GameKind,
    deps: EnvDeps,
) -> Environment[Any, Any, Any]: ...
def make_environment(
    *,
    game_kind: GameKind,
    deps: EnvDeps,
) -> Environment[Any, Any, Any]:
    """Create an environment wiring bundle for the given game kind."""
    match game_kind:
        case GameKind.CHESS:
            from chipiron.environments.chess.environment import make_chess_environment

            assert isinstance(deps, ChessEnvironmentDeps)
            return make_chess_environment(deps=deps)
        case GameKind.CHECKERS:
            assert isinstance(deps, CheckersEnvironmentDeps)
            raise NotImplementedError("Environment for CHECKERS is not implemented yet")
        case _:
            raise EnvironmentNotFoundError(game_kind)

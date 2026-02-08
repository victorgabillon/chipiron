"""Environment wiring and factory helpers."""

from typing import Any, Literal, overload

from atomheart.board.utils import FenPlusHistory

from chipiron.environments.base import Environment
from chipiron.environments.chess.environment import make_chess_environment
from chipiron.environments.chess.tags import ChessStartTag
from chipiron.environments.chess.types import ChessState
from chipiron.environments.deps import CheckersEnvironmentDeps, ChessEnvironmentDeps
from chipiron.environments.types import GameKind


class EnvironmentCreationError(ValueError):
    """Base error for environment creation failures."""


class EnvironmentNotFoundError(EnvironmentCreationError):
    """Raised when an environment implementation is missing."""

    def __init__(self, game_kind: GameKind) -> None:
        """Initialize the error with the missing game kind."""
        super().__init__(f"No Environment for game_kind={game_kind!r}")


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
            assert isinstance(deps, ChessEnvironmentDeps)
            return make_chess_environment(deps=deps)
        case GameKind.CHECKERS:
            assert isinstance(deps, CheckersEnvironmentDeps)
            raise NotImplementedError("Environment for CHECKERS is not implemented yet")
        case _:
            raise EnvironmentNotFoundError(game_kind)

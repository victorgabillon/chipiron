"""Encoders for player move requests and state snapshots."""

from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, cast

from atomheart.games.chess.board.utils import FenPlusHistory
from valanga import Color, SoloRole
from valanga.game import Seed

from chipiron.displays.gui_protocol import Scope
from chipiron.environments.types import GameKind
from chipiron.players.communications.player_message import (
    PlayerRequest,
    TurnStatePlusHistory,
)

StateT_contra = TypeVar("StateT_contra", contravariant=True)
StateSnapT = TypeVar("StateSnapT", default=Any)


class BoardWithFenHistory(Protocol):
    """Protocol for boards that can export FEN+history snapshots."""

    def into_fen_plus_history(self) -> FenPlusHistory:
        """Return a FEN snapshot enriched with history."""
        ...


class HasFenHistoryState(Protocol):
    """Protocol for chess-like runtime states used by move request encoding."""

    @property
    def tag(self) -> Any:
        """Return state tag."""
        ...

    @property
    def turn(self) -> Any:
        """Return the game role to play."""
        ...

    @property
    def board(self) -> BoardWithFenHistory:
        """Return board supporting FEN+history snapshots."""
        ...


class CheckersLikeState(Protocol):
    """Protocol for checkers-like states serializable as text snapshots."""

    @property
    def tag(self) -> Any:
        """Return state tag."""
        ...

    @property
    def turn(self) -> Color:
        """Return the game role to play."""
        ...

    def to_text(self) -> str:
        """Return a text snapshot for transport."""
        ...


class IntegerReductionLikeState(Protocol):
    """Protocol for solo integer-reduction states transported as integers."""

    @property
    def tag(self) -> Any:
        """Return state tag."""
        ...

    @property
    def turn(self) -> SoloRole:
        """Return the acting solo role."""
        ...

    @property
    def value(self) -> int:
        """Return the current integer value."""
        ...


class MorpionLikeState(Protocol):
    """Protocol for solo Morpion states transported as runtime snapshots."""

    @property
    def tag(self) -> Any:
        """Return state tag."""
        ...

    @property
    def turn(self) -> SoloRole:
        """Return the acting solo role."""
        ...


class PlayerRequestEncoderError(ValueError):
    """Raised when a player request encoder is missing for a game kind."""

    def __init__(self, game_kind: GameKind) -> None:
        """Initialize the error with the missing game kind."""
        super().__init__(f"No PlayerRequestEncoder for game_kind={game_kind!r}")


class PlayerRequestEncoder(Protocol[StateT_contra, StateSnapT]):
    """Protocol for encoding player move requests."""

    game_kind: GameKind

    def make_move_request(
        self,
        *,
        state: StateT_contra,
        seed: Seed,
        scope: Scope,
    ) -> PlayerRequest[StateSnapT]:
        """Build a player request from the current state."""
        ...


@dataclass(frozen=True, slots=True)
class ChessPlayerRequestEncoder(
    PlayerRequestEncoder[HasFenHistoryState, FenPlusHistory]
):
    """Chess-specific move request encoder."""

    game_kind: GameKind = GameKind.CHESS

    def make_move_request(
        self, *, state: HasFenHistoryState, seed: Seed, scope: Scope
    ) -> PlayerRequest[FenPlusHistory]:
        """Encode a chess move request using FEN history."""
        fen_plus_history: FenPlusHistory = state.board.into_fen_plus_history()

        return PlayerRequest(
            schema_version=1,
            scope=scope,
            seed=seed,
            state=TurnStatePlusHistory(
                current_state_tag=state.tag,
                role_to_play=state.turn,
                snapshot=fen_plus_history,
                historical_actions=None,
            ),
        )


@dataclass(frozen=True, slots=True)
class CheckersPlayerRequestEncoder(PlayerRequestEncoder[CheckersLikeState, str]):
    """Checkers-specific move request encoder."""

    game_kind: GameKind = GameKind.CHECKERS

    def make_move_request(
        self, *, state: CheckersLikeState, seed: Seed, scope: Scope
    ) -> PlayerRequest[str]:
        """Encode a checkers move request using text state serialization."""
        return PlayerRequest(
            schema_version=1,
            scope=scope,
            seed=seed,
            state=TurnStatePlusHistory(
                current_state_tag=state.tag,
                role_to_play=state.turn,
                snapshot=state.to_text(),
                historical_actions=None,
            ),
        )


@dataclass(frozen=True, slots=True)
class IntegerReductionPlayerRequestEncoder(
    PlayerRequestEncoder[IntegerReductionLikeState, int]
):
    """Integer-reduction move request encoder."""

    game_kind: GameKind = GameKind.INTEGER_REDUCTION

    def make_move_request(
        self, *, state: IntegerReductionLikeState, seed: Seed, scope: Scope
    ) -> PlayerRequest[int]:
        """Encode an integer-reduction move request using the current integer value."""
        return PlayerRequest(
            schema_version=1,
            scope=scope,
            seed=seed,
            state=TurnStatePlusHistory(
                current_state_tag=state.tag,
                role_to_play=state.turn,
                snapshot=state.value,
                historical_actions=None,
            ),
        )


@dataclass(frozen=True, slots=True)
class MorpionPlayerRequestEncoder(
    PlayerRequestEncoder[MorpionLikeState, MorpionLikeState]
):
    """Morpion move request encoder."""

    game_kind: GameKind = GameKind.MORPION

    def make_move_request(
        self,
        *,
        state: MorpionLikeState,
        seed: Seed,
        scope: Scope,
    ) -> PlayerRequest[MorpionLikeState]:
        """Encode a Morpion move request using the runtime state snapshot."""
        return PlayerRequest(
            schema_version=1,
            scope=scope,
            seed=seed,
            state=TurnStatePlusHistory(
                current_state_tag=state.tag,
                role_to_play=state.turn,
                snapshot=state,
                historical_actions=None,
            ),
        )


def make_player_request_encoder[StateT](
    *,
    game_kind: GameKind,
    state_type: type[StateT],
) -> PlayerRequestEncoder[StateT, Any]:
    """Create a player request encoder for the given game kind."""
    _ = state_type  # witness

    match game_kind:
        case GameKind.CHESS:
            return cast(
                "PlayerRequestEncoder[StateT, Any]", ChessPlayerRequestEncoder()
            )
        case GameKind.CHECKERS:
            return cast(
                "PlayerRequestEncoder[StateT, Any]", CheckersPlayerRequestEncoder()
            )
        case GameKind.INTEGER_REDUCTION:
            return cast(
                "PlayerRequestEncoder[StateT, Any]",
                IntegerReductionPlayerRequestEncoder(),
            )
        case GameKind.MORPION:
            return cast("PlayerRequestEncoder[StateT, Any]", MorpionPlayerRequestEncoder())
        case _:
            raise PlayerRequestEncoderError(game_kind)

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast

from atomheart.board.utils import FenPlusHistory

from chipiron.environments.chess.types import ChessState
from chipiron.environments.types import GameKind
from chipiron.players.communications.player_message import (
    PlayerRequest,
    TurnStatePlusHistory,
)
from chipiron.utils.color import to_valanga_color

if TYPE_CHECKING:
    from valanga import Color
    from valanga.game import Seed

    from chipiron.displays.gui_protocol import Scope

StateT_contra = TypeVar("StateT_contra", contravariant=True)
StateSnapT = TypeVar("StateSnapT", default=Any)


class PlayerRequestEncoder(Protocol[StateT_contra, StateSnapT]):
    game_kind: GameKind

    def make_move_request(
        self,
        *,
        state: StateT_contra,
        seed: Seed,
        scope: Scope,
    ) -> PlayerRequest[StateSnapT]: ...


@dataclass(frozen=True, slots=True)
class ChessPlayerRequestEncoder(PlayerRequestEncoder[ChessState, FenPlusHistory]):
    game_kind: GameKind = GameKind.CHESS

    def make_move_request(
        self, *, state: ChessState, seed: Seed, scope: Scope
    ) -> PlayerRequest[FenPlusHistory]:
        fen_plus_history: FenPlusHistory = state.into_fen_plus_history()

        # Minimal, picklable snapshot: carry only the latest FenPlusHistory.
        # The request stays game-agnostic; players decode/consume the snapshot.
        turn: Color = to_valanga_color(
            getattr(state, "turn", fen_plus_history.current_turn())
        )

        return PlayerRequest(
            schema_version=1,
            scope=scope,
            seed=seed,
            state=TurnStatePlusHistory(
                current_state_tag=getattr(state, "tag", fen_plus_history.current_fen),
                turn=turn,
                snapshot=fen_plus_history,
                historical_actions=None,
            ),
        )


def make_player_request_encoder[StateT](
    *,
    game_kind: GameKind,
    state_type: type[StateT],
) -> PlayerRequestEncoder[StateT, Any]:
    _ = state_type  # witness

    match game_kind:
        case GameKind.CHESS:
            return cast(
                "PlayerRequestEncoder[StateT, Any]", ChessPlayerRequestEncoder()
            )
        case GameKind.CHECKERS:
            raise NotImplementedError(
                "PlayerRequestEncoder for CHECKERS is not implemented yet"
            )
        case _:
            raise ValueError(f"No PlayerRequestEncoder for game_kind={game_kind!r}")


# Backwards-compatible alias (older code imports make_player_encoder)
def make_player_encoder[StateT](
    *,
    game_kind: GameKind,
    state_type: type[StateT],
) -> PlayerRequestEncoder[StateT, Any]:
    return make_player_request_encoder(game_kind=game_kind, state_type=state_type)

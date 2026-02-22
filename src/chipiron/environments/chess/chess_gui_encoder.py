"""Module for chess gui encoder."""

from dataclasses import dataclass

from valanga.game import Seed

from chipiron.displays.gui_protocol import (
    UpdatePayload,
    UpdGameStatus,
    UpdStateGeneric,
)
from chipiron.environments.chess.types import ChessState
from chipiron.environments.types import GameKind
from chipiron.games.game.game_playing_status import PlayingStatus
from chipiron.utils.communication.gui_encoder import GuiEncoder


@dataclass(frozen=True, slots=True)
class ChessGuiEncoder(GuiEncoder[ChessState]):
    """Chessguiencoder implementation."""

    game_kind: GameKind = GameKind.CHESS

    def make_state_payload(
        self,
        *,
        state: ChessState,
        seed: Seed | None,
    ) -> UpdatePayload:
        """Create state payload."""
        fen_plus_history = state.board.into_fen_plus_history()
        historical_moves = getattr(fen_plus_history, "historical_moves", None) or []

        return UpdStateGeneric(
            state_tag=state.tag,
            action_name_history=[str(move) for move in historical_moves],
            adapter_payload=fen_plus_history,
            seed=seed,
        )

    def make_status_payload(
        self,
        *,
        status: PlayingStatus,
    ) -> UpdatePayload:
        """Create status payload."""
        return UpdGameStatus(status=status)

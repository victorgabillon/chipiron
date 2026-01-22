from dataclasses import dataclass

from valanga.game import Seed

from chipiron.displays.gui_protocol import (
    UpdatePayload,
    UpdGameStatus,
    UpdStateChess,
)
from chipiron.environments.chess.types import ChessState
from chipiron.environments.types import GameKind
from chipiron.games.game.game_playing_status import PlayingStatus
from chipiron.utils.communication.gui_encoder import GuiEncoder


@dataclass(frozen=True, slots=True)
class ChessGuiEncoder(GuiEncoder[ChessState]):
    game_kind: GameKind = GameKind.CHESS

    def make_state_payload(
        self,
        *,
        state: ChessState,
        seed: Seed | None,
    ) -> UpdatePayload:
        return UpdStateChess(
            fen_plus_history=state.into_fen_plus_history(),
            seed=seed,
        )

    def make_status_payload(
        self,
        *,
        status: PlayingStatus,
    ) -> UpdatePayload:
        return UpdGameStatus(status=status)

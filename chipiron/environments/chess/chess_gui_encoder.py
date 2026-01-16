
from valanga.game import  Seed

from chipiron.environments.types import GameKind
from chipiron.games.game.game_playing_status import PlayingStatus
from chipiron.utils.communication.gui_messages.gui_messages import (
    UpdGameStatus,
    UpdStateChess,
    UpdatePayload,
)
from atomheart.board.iboard import IBoard

from chipiron.utils.communication.gui_encoder import GuiEncoder

@dataclass(frozen=True, slots=True)
class ChessGuiEncoder(GuiEncoder[IBoard]):

    game_kind: GameKind = GameKind.CHESS
    def make_state_payload(
        self,
        *,
        state: IBoard,
        seed: Seed | None,
    ) -> UpdatePayload:
        return UpdStateChess(
            kind="state_chess",
            fen_plus_history=state.into_fen_plus_history(),
            seed=seed,
        )

    def make_status_payload(
        self,
        *,
        status: PlayingStatus,
    ) -> UpdatePayload:
        return UpdGameStatus(kind="game_status", status=status)
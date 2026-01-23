from typing import cast

from chipiron.environments.chess.chess_gui_encoder import ChessGuiEncoder
from chipiron.environments.types import GameKind
from chipiron.utils.communication.gui_encoder import GuiEncoder


def make_gui_encoder[StateT](
    *,
    game_kind: GameKind,
    state_type: type[StateT],
) -> GuiEncoder[StateT]:
    _ = state_type  # witness / keeps pyright happy
    match game_kind:
        case GameKind.CHESS:
            return cast("GuiEncoder[StateT]", ChessGuiEncoder())
        case _:
            raise ValueError(f"No GuiEncoder for game_kind={game_kind!r}")

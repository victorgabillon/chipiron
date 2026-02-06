"""Factory helpers for GUI encoders."""

from typing import cast

from chipiron.environments.chess.chess_gui_encoder import ChessGuiEncoder
from chipiron.environments.types import GameKind
from chipiron.utils.communication.gui_encoder import GuiEncoder


class GuiEncoderError(ValueError):
    """Raised when a GUI encoder is missing for a game kind."""

    def __init__(self, game_kind: GameKind) -> None:
        """Initialize the error with the missing game kind."""
        super().__init__(f"No GuiEncoder for game_kind={game_kind!r}")


def make_gui_encoder[StateT](
    *,
    game_kind: GameKind,
    state_type: type[StateT],
) -> GuiEncoder[StateT]:
    """Create a GUI encoder appropriate for the given game kind."""
    _ = state_type  # witness / keeps pyright happy
    match game_kind:
        case GameKind.CHESS:
            return cast("GuiEncoder[StateT]", ChessGuiEncoder())
        case _:
            raise GuiEncoderError(game_kind)

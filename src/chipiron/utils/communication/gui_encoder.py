"""Module for gui encoder."""

from typing import TYPE_CHECKING, Protocol

# chipiron/utils/communication/gui_encoder.py

if TYPE_CHECKING:
    from valanga.game import Seed

    from chipiron.displays.gui_protocol import UpdatePayload
    from chipiron.environments.types import GameKind
    from chipiron.games.domain.game.game_playing_status import PlayingStatus


class GuiEncoder[StateT_contra](Protocol):
    """Guiencoder implementation."""

    game_kind: GameKind

    def make_state_payload(
        self,
        *,
        state: StateT_contra,
        seed: Seed | None,
    ) -> UpdatePayload:
        """Create state payload."""
        ...

    def make_status_payload(
        self,
        *,
        status: PlayingStatus,
    ) -> UpdatePayload:
        """Create status payload."""
        ...

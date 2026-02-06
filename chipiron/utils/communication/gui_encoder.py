"""Module for gui encoder."""

from typing import Protocol, TypeVar

from valanga.game import Seed

from chipiron.displays.gui_protocol import UpdatePayload

# chipiron/utils/communication/gui_encoder.py
from chipiron.environments.types import GameKind
from chipiron.games.game.game_playing_status import PlayingStatus

StateT_contra = TypeVar("StateT_contra", contravariant=True)


class GuiEncoder(Protocol[StateT_contra]):
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

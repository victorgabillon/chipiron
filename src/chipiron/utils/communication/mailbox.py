"""Module for mailbox."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from chipiron.displays.gui_protocol import GuiCommand
    from chipiron.players.communications.player_message import PlayerEvent

    type MainMailboxMessage = GuiCommand | PlayerEvent
else:
    # Runtime: keep it loose to avoid import cycles.
    MainMailboxMessage = Any  # type: ignore[misc]

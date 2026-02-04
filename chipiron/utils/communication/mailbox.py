"""Module for mailbox."""
from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    # These imports are ONLY for type checkers (mypy/pyright)
    from chipiron.displays.gui_protocol import GuiCommand  # or GuiCommand type you use
    from chipiron.players.communications.player_message import PlayerEvent

    MainMailboxMessage: TypeAlias = (
        GuiCommand | PlayerEvent
    )  # adjust to your real union
else:
    # Runtime: avoid importing GUI/players to prevent cycles.
    MainMailboxMessage: TypeAlias = Any

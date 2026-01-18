"""Deprecated compatibility shim.

The player runtime handler is now game-agnostic. Any remaining imports of
`handle_player_request_chess` should be migrated to
`chipiron.players.communications.player_runtime.handle_player_request`.
"""

from __future__ import annotations

from chipiron.players.communications.player_runtime import (
    handle_player_request as handle_player_request_chess,
)

__all__ = ["handle_player_request_chess"]

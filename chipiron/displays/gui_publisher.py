"""Module for gui publisher."""

import queue
from dataclasses import dataclass

from chipiron.environments.types import GameKind

from .gui_protocol import GuiUpdate, SchemaVersion, Scope, UpdatePayload


@dataclass(frozen=True, slots=True)
class GuiPublisher:
    """Guipublisher implementation."""

    out: queue.Queue[GuiUpdate]
    schema_version: SchemaVersion
    game_kind: GameKind
    scope: Scope

    def publish(self, payload: UpdatePayload) -> None:
        """Publish."""
        self.out.put(
            GuiUpdate(
                schema_version=self.schema_version,
                game_kind=self.game_kind,
                scope=self.scope,
                payload=payload,
            )
        )

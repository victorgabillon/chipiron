from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .gui_protocol import GuiUpdate, SchemaVersion, Scope, UpdatePayload

if TYPE_CHECKING:
    import queue

    from chipiron.environments.types import GameKind


@dataclass(frozen=True, slots=True)
class GuiPublisher:
    out: queue.Queue[GuiUpdate]
    schema_version: SchemaVersion
    game_kind: GameKind
    scope: Scope

    def publish(self, payload: UpdatePayload) -> None:
        self.out.put(
            GuiUpdate(
                schema_version=self.schema_version,
                game_kind=self.game_kind,
                scope=self.scope,
                payload=payload,
            )
        )

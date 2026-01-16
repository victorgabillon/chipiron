from dataclasses import dataclass
import queue

from chipiron.environments.types import GameKind
from chipiron.utils.communication.gui_messages.gui_messages import GameId, GuiUpdate, UpdatePayload

@dataclass(frozen=True, slots=True)
class GuiPublisher:
    out: queue.Queue[GuiUpdate]
    schema_version: int
    game_kind: GameKind
    game_id: GameId

    def publish(self, payload: UpdatePayload) -> None:
        self.out.put(GuiUpdate(
            schema_version=self.schema_version,
            game_kind=self.game_kind,
            game_id=self.game_id,
            payload=payload,
        ))

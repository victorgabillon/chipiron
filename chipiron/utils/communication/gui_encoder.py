from typing import Protocol, TypeVar

from valanga.game import Seed

# chipiron/utils/communication/gui_encoder.py
from chipiron.environments.types import GameKind
from chipiron.games.game.game_playing_status import PlayingStatus
from chipiron.displays.gui_protocol import UpdatePayload

StateT_contra = TypeVar("StateT_contra", contravariant=True)


class GuiEncoder(Protocol[StateT_contra]):
    game_kind: GameKind

    def make_state_payload(
        self,
        *,
        state: StateT_contra,
        seed: Seed | None,
    ) -> UpdatePayload: ...

    def make_status_payload(
        self,
        *,
        status: PlayingStatus,
    ) -> UpdatePayload: ...

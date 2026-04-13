"""Morpion GUI encoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from chipiron.displays.gui_protocol import UpdatePayload, UpdGameStatus, UpdStateGeneric
from chipiron.environments.morpion.morpion_display import build_morpion_display_payload
from chipiron.environments.morpion.types import (
    MorpionDynamics,
    MorpionState,
)
from chipiron.environments.types import GameKind
from chipiron.utils.communication.gui_encoder import GuiEncoder

if TYPE_CHECKING:
    from valanga.game import Seed

    from chipiron.games.domain.game.game_playing_status import PlayingStatus


@dataclass(frozen=True, slots=True)
class MorpionGuiEncoder(GuiEncoder[MorpionState]):
    """Encode Morpion state updates for the generic GUI."""

    dynamics: MorpionDynamics
    game_kind: GameKind = GameKind.MORPION

    def make_state_payload(
        self,
        *,
        state: MorpionState,
        seed: Seed | None,
    ) -> UpdatePayload:
        """Create state payload."""
        return UpdStateGeneric(
            state_tag=state.tag,
            action_name_history=[],
            adapter_payload=build_morpion_display_payload(
                state=state,
                dynamics=self.dynamics,
            ),
            seed=seed,
        )

    def make_status_payload(
        self,
        *,
        status: PlayingStatus,
    ) -> UpdatePayload:
        """Create status payload."""
        return UpdGameStatus(status=status)

"""Morpion GUI encoder."""

from dataclasses import dataclass

from valanga.game import Seed

from chipiron.displays.gui_protocol import UpdatePayload, UpdGameStatus, UpdStateGeneric
from chipiron.environments.morpion.types import MorpionDynamics, MorpionState
from chipiron.environments.types import GameKind
from chipiron.games.domain.game.game_playing_status import PlayingStatus
from chipiron.utils.communication.gui_encoder import GuiEncoder


@dataclass(frozen=True, slots=True)
class MorpionDisplayPayload:
    """Structured GUI payload for Morpion state rendering."""

    variant: str
    moves: int
    point_count: int
    legal_actions: tuple[str, ...]
    is_terminal: bool


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
        legal_actions = tuple(
            self.dynamics.action_name(state, action)
            for action in self.dynamics.legal_actions(state).get_all()
        )

        return UpdStateGeneric(
            state_tag=state.tag,
            action_name_history=[],
            adapter_payload=MorpionDisplayPayload(
                variant=state.variant.value,
                moves=state.moves,
                point_count=len(state.points),
                legal_actions=legal_actions,
                is_terminal=state.is_game_over(),
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

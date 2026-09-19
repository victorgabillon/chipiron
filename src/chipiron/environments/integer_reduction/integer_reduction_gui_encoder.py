"""Integer reduction GUI encoder."""

from dataclasses import dataclass

from valanga.game import Seed

from chipiron.displays.gui_protocol import UpdatePayload, UpdGameStatus, UpdStateGeneric
from chipiron.environments.integer_reduction.types import (
    IntegerReductionDynamics,
    IntegerReductionState,
)
from chipiron.environments.types import GameKind
from chipiron.games.domain.game.game_playing_status import PlayingStatus
from chipiron.utils.communication.gui_encoder import GuiEncoder


@dataclass(frozen=True, slots=True)
class IntegerReductionDisplayPayload:
    """Structured GUI payload for integer reduction state rendering."""

    value: int
    steps: int
    legal_actions: tuple[str, ...]
    is_terminal: bool


@dataclass(frozen=True, slots=True)
class IntegerReductionGuiEncoder(GuiEncoder[IntegerReductionState]):
    """Encode integer-reduction state updates for the generic GUI."""

    dynamics: IntegerReductionDynamics
    game_kind: GameKind = GameKind.INTEGER_REDUCTION

    def make_state_payload(
        self,
        *,
        state: IntegerReductionState,
        seed: Seed | None,
    ) -> UpdatePayload:
        """Create state payload.

        Chronological action history is injected by ``ObservableGame`` from the
        authoritative game object, so this encoder only projects the current
        integer-reduction state.
        """
        legal_actions = tuple(
            self.dynamics.action_name(state, action)
            for action in self.dynamics.legal_actions(state).get_all()
        )

        return UpdStateGeneric(
            state_tag=state.tag,
            action_name_history=[],
            adapter_payload=IntegerReductionDisplayPayload(
                value=state.value,
                steps=state.steps,
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

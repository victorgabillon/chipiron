"""Checkers GUI encoder."""

from dataclasses import dataclass

from atomheart.games.checkers.generation import generate_legal_moves
from valanga.game import Seed

from chipiron.displays.gui_protocol import UpdatePayload, UpdGameStatus, UpdStateGeneric
from chipiron.environments.checkers.types import (
    CheckersDynamics,
    CheckersRules,
    CheckersState,
)
from chipiron.environments.types import GameKind
from chipiron.games.domain.game.game_playing_status import PlayingStatus
from chipiron.utils.communication.gui_encoder import GuiEncoder


@dataclass(frozen=True, slots=True)
class CheckersGuiEncoder(GuiEncoder[CheckersState]):
    """Encode checkers state updates for the generic GUI."""

    rules: CheckersRules
    game_kind: GameKind = GameKind.CHECKERS

    def make_state_payload(
        self,
        *,
        state: CheckersState,
        seed: Seed | None,
    ) -> UpdatePayload:
        """Create state payload."""
        legal_moves = generate_legal_moves(state, self.rules)
        naming_dynamics = CheckersDynamics(self.rules)

        return UpdStateGeneric(
            state_tag=state.tag,
            action_name_history=[],
            adapter_payload={
                "position_text": state.to_text(),
                "pieces": state.pieces_by_square(),
                "legal_moves": [
                    naming_dynamics.action_name(state, move) for move in legal_moves
                ],
            },
            seed=seed,
        )

    def make_status_payload(
        self,
        *,
        status: PlayingStatus,
    ) -> UpdatePayload:
        """Create status payload."""
        return UpdGameStatus(status=status)

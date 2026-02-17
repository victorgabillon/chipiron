"""Document the module contains the implementation of the CommandLineHumanMoveSelector class, which allows a human player.

to select moves through the command line interface.
"""

from dataclasses import dataclass
from typing import Literal

from valanga import Dynamics, TurnState
from valanga.game import Seed
from valanga.policy import NotifyProgressCallable, Recommendation

from chipiron.utils.logger import chipiron_logger

from .move_selector_types import MoveSelectorTypes


@dataclass
class CommandLineHumanPlayerArgs:
    """Represents the arguments for a human player that selects moves through the command line interface."""

    type: Literal[MoveSelectorTypes.COMMAND_LINE_HUMAN]  # for serialization


@dataclass
class GuiHumanPlayerArgs:
    """Represents the arguments for a human player that selects moves through the GUI."""

    type: Literal[MoveSelectorTypes.GUI_HUMAN]  # for serialization


@dataclass
class CommandLineHumanMoveSelector[StateT: TurnState]:
    """Select moves interactively from command-line input."""

    dynamics: Dynamics[StateT]

    def recommend(
        self,
        state: StateT,
        seed: Seed,
        notify_progress: NotifyProgressCallable | None = None,
    ) -> Recommendation:
        """Recommend one legal move chosen by the user."""
        _ = seed
        _ = notify_progress

        actions = list(self.dynamics.legal_actions(state).get_all())
        names = [self.dynamics.action_name(state, a) for a in actions]
        chipiron_logger.info("Legal moves: %s", names)

        while True:
            name = input("Input your move: ").strip()
            if name in names:
                return Recommendation(recommended_name=name, evaluation=None)
            chipiron_logger.info("Bad move, not legal.")

"""Checkers environment wiring."""

from typing import TYPE_CHECKING

from valanga import StateTag

from chipiron.environments.base import Environment
from chipiron.environments.checkers.checkers_gui_encoder import CheckersGuiEncoder
from chipiron.environments.checkers.tags import CheckersStartTag
from chipiron.environments.checkers.types import CheckersDynamics, CheckersRules, CheckersState
from chipiron.environments.deps import CheckersEnvironmentDeps
from chipiron.environments.types import GameKind
from chipiron.players.communications.player_request_encoder import (
    CheckersPlayerRequestEncoder,
)
from chipiron.players.factory_higher_level import create_player_observer_factory
from chipiron.scripts.chipiron_args import ImplementationArgs

if TYPE_CHECKING:
    from chipiron.players.factory_higher_level import PlayerObserverFactory


class CheckersEnvironmentError(TypeError):
    """Base error for checkers environment configuration issues."""


class CheckersStartTagTypeError(CheckersEnvironmentError):
    """Raised when an invalid start tag is provided."""

    def __init__(self, tag: StateTag) -> None:
        """Initialize the error with the invalid tag."""
        super().__init__(
            f"Checkers environment expects CheckersStartTag, got {type(tag)!r}"
        )


def make_checkers_environment(
    *, deps: CheckersEnvironmentDeps
) -> Environment[CheckersState, str, CheckersStartTag]:
    """Create checkers environment."""

    def build_player_observer_factory(
        *,
        each_player_has_its_own_thread: bool,
        implementation_args: ImplementationArgs,
        universal_behavior: bool,
    ) -> "PlayerObserverFactory":
        return create_player_observer_factory(
            game_kind=GameKind.CHECKERS,
            each_player_has_its_own_thread=each_player_has_its_own_thread,
            implementation_args=implementation_args,
            universal_behavior=universal_behavior,
        )

    def normalize_start_tag(tag: StateTag) -> CheckersStartTag:
        if not isinstance(tag, CheckersStartTag):
            raise CheckersStartTagTypeError(tag)
        return tag

    def make_initial_state(tag: CheckersStartTag) -> CheckersState:
        return CheckersState.from_text(tag.text)

    rules = CheckersRules(forced_capture=deps.forced_capture)

    return Environment(
        game_kind=GameKind.CHECKERS,
        rules=rules,
        dynamics=CheckersDynamics(rules),
        gui_encoder=CheckersGuiEncoder(rules=rules),
        player_encoder=CheckersPlayerRequestEncoder(),
        make_player_observer_factory=build_player_observer_factory,
        normalize_start_tag=normalize_start_tag,
        make_initial_state=make_initial_state,
    )

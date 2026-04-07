"""Morpion environment wiring."""

from typing import TYPE_CHECKING

from atomheart.games.morpion import initial_state as morpion_initial_state
from valanga import SOLO, StateTag

from chipiron.environments.base import Environment
from chipiron.environments.deps import MorpionEnvironmentDeps
from chipiron.environments.morpion.morpion_gui_encoder import MorpionGuiEncoder
from chipiron.environments.morpion.morpion_rules import MorpionRules
from chipiron.environments.morpion.tags import MorpionStartTag
from chipiron.environments.morpion.types import MorpionDynamics, MorpionState
from chipiron.environments.types import GameKind
from chipiron.players.communications.player_request_encoder import (
    MorpionPlayerRequestEncoder,
)
from chipiron.players.factory_higher_level import create_player_observer_factory
from chipiron.scripts.chipiron_args import ImplementationArgs

if TYPE_CHECKING:
    from chipiron.players.factory_higher_level import PlayerObserverFactory


class MorpionEnvironmentError(TypeError):
    """Base error for Morpion environment configuration issues."""


class MorpionStartTagTypeError(MorpionEnvironmentError):
    """Raised when an invalid start tag is provided."""

    def __init__(self, tag: StateTag) -> None:
        """Initialize the error with the invalid tag."""
        super().__init__(
            f"Morpion environment expects MorpionStartTag, got {type(tag)!r}"
        )


def make_morpion_environment(
    *,
    deps: MorpionEnvironmentDeps,
) -> Environment[MorpionState, MorpionState, MorpionStartTag]:
    """Create Morpion environment."""
    del deps

    def build_player_observer_factory(
        *,
        each_player_has_its_own_thread: bool,
        implementation_args: ImplementationArgs,
        universal_behavior: bool,
    ) -> "PlayerObserverFactory":
        return create_player_observer_factory(
            game_kind=GameKind.MORPION,
            each_player_has_its_own_thread=each_player_has_its_own_thread,
            implementation_args=implementation_args,
            universal_behavior=universal_behavior,
        )

    def normalize_start_tag(tag: StateTag) -> MorpionStartTag:
        if not isinstance(tag, MorpionStartTag):
            raise MorpionStartTagTypeError(tag)
        return tag

    def make_initial_state(tag: MorpionStartTag) -> MorpionState:
        return dynamics.wrap_atomheart_state(morpion_initial_state(variant=tag.variant))

    dynamics = MorpionDynamics()

    return Environment(
        game_kind=GameKind.MORPION,
        roles=(SOLO,),
        rules=MorpionRules(),
        dynamics=dynamics,
        gui_encoder=MorpionGuiEncoder(dynamics=dynamics),
        player_encoder=MorpionPlayerRequestEncoder(),
        make_player_observer_factory=build_player_observer_factory,
        normalize_start_tag=normalize_start_tag,
        make_initial_state=make_initial_state,
    )

"""Integer reduction environment wiring."""

from typing import TYPE_CHECKING

from valanga import SOLO, StateTag

from chipiron.environments.base import Environment
from chipiron.environments.deps import IntegerReductionEnvironmentDeps
from chipiron.environments.integer_reduction.integer_reduction_gui_encoder import (
    IntegerReductionGuiEncoder,
)
from chipiron.environments.integer_reduction.integer_reduction_rules import (
    IntegerReductionRules,
)
from chipiron.environments.integer_reduction.tags import IntegerReductionStartTag
from chipiron.environments.integer_reduction.types import (
    IntegerReductionDynamics,
    IntegerReductionState,
)
from chipiron.environments.types import GameKind
from chipiron.players.communications.player_request_encoder import (
    IntegerReductionPlayerRequestEncoder,
)
from chipiron.players.factory_higher_level import create_player_observer_factory
from chipiron.scripts.chipiron_args import ImplementationArgs

if TYPE_CHECKING:
    from chipiron.players.factory_higher_level import PlayerObserverFactory


class IntegerReductionEnvironmentError(TypeError):
    """Base error for integer-reduction environment configuration issues."""


class IntegerReductionStartTagTypeError(IntegerReductionEnvironmentError):
    """Raised when an invalid start tag is provided."""

    def __init__(self, tag: StateTag) -> None:
        """Initialize the error with the invalid tag."""
        super().__init__(
            "Integer reduction environment expects IntegerReductionStartTag, "
            f"got {type(tag)!r}"
        )


def make_integer_reduction_environment(
    *,
    deps: IntegerReductionEnvironmentDeps,
) -> Environment[IntegerReductionState, int, IntegerReductionStartTag]:
    """Create integer-reduction environment."""
    del deps

    def build_player_observer_factory(
        *,
        each_player_has_its_own_thread: bool,
        implementation_args: ImplementationArgs,
        universal_behavior: bool,
    ) -> "PlayerObserverFactory":
        return create_player_observer_factory(
            game_kind=GameKind.INTEGER_REDUCTION,
            each_player_has_its_own_thread=each_player_has_its_own_thread,
            implementation_args=implementation_args,
            universal_behavior=universal_behavior,
        )

    def normalize_start_tag(tag: StateTag) -> IntegerReductionStartTag:
        if not isinstance(tag, IntegerReductionStartTag):
            raise IntegerReductionStartTagTypeError(tag)
        return tag

    def make_initial_state(tag: IntegerReductionStartTag) -> IntegerReductionState:
        return IntegerReductionState(value=tag.value)

    dynamics = IntegerReductionDynamics()

    return Environment(
        game_kind=GameKind.INTEGER_REDUCTION,
        roles=(SOLO,),
        rules=IntegerReductionRules(),
        dynamics=dynamics,
        gui_encoder=IntegerReductionGuiEncoder(dynamics=dynamics),
        player_encoder=IntegerReductionPlayerRequestEncoder(),
        make_player_observer_factory=build_player_observer_factory,
        normalize_start_tag=normalize_start_tag,
        make_initial_state=make_initial_state,
    )

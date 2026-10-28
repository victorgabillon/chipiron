"""Core environment protocols and data structures."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from valanga import Dynamics, State

from chipiron.core.roles import GameRole
from chipiron.environments.types import GameKind
from chipiron.games.domain.game.game_rules import GameRules
from chipiron.players.communications.player_request_encoder import (
    PlayerRequestEncoder,
)
from chipiron.utils.communication.gui_encoder import GuiEncoder

if TYPE_CHECKING:
    from valanga import StateTag

    from chipiron.players.factory_higher_level import PlayerObserverFactory
    from chipiron.scripts.chipiron_args import ImplementationArgs


class TagNormalizer[StartTagOutT_co](Protocol):
    """Protocol for normalizing raw start tags."""

    def __call__(self, tag: StateTag) -> StartTagOutT_co:
        """Normalize a start tag to the environment's expected type."""
        ...


class InitialStateFactory[StateOutT_co, StartTagInT_contra](Protocol):
    """Protocol for creating initial states from start tags."""

    def __call__(self, tag: StartTagInT_contra) -> StateOutT_co:
        """Create the initial state from a normalized start tag."""
        ...


class PlayerObserverFactoryBuilder(Protocol):
    """Protocol for building player observer factories."""

    def __call__(
        self,
        *,
        each_player_has_its_own_thread: bool,
        implementation_args: ImplementationArgs,
        universal_behavior: bool,
    ) -> PlayerObserverFactory:
        """Build a player observer factory with the given configuration."""
        ...


@dataclass(frozen=True)
class Environment[StateT: State, StateSnapT, StartTagT]:
    """Bundle environment wiring and factories for a game kind."""

    game_kind: GameKind
    roles: Sequence[GameRole]
    rules: GameRules[StateT]
    dynamics: Dynamics[StateT]
    gui_encoder: GuiEncoder[StateT]
    player_encoder: PlayerRequestEncoder[StateT, StateSnapT]
    make_player_observer_factory: PlayerObserverFactoryBuilder
    normalize_start_tag: TagNormalizer[StartTagT]
    make_initial_state: InitialStateFactory[StateT, StartTagT]

"""Core environment protocols and data structures."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeVar

from valanga import Dynamics, StateTag

from chipiron.environments.types import GameKind
from chipiron.games.domain.game.game_rules import GameRules
from chipiron.scripts.chipiron_args import ImplementationArgs
from chipiron.utils.communication.gui_encoder import GuiEncoder

if TYPE_CHECKING:
    from chipiron.players.communications.player_request_encoder import (
        PlayerRequestEncoder,
    )
    from chipiron.players.factory_higher_level import PlayerObserverFactory


StateT = TypeVar("StateT")
StateSnapT = TypeVar("StateSnapT")

StartTagT = TypeVar("StartTagT")  # produced by normalize_start_tag
StartTagOutT_co = TypeVar("StartTagOutT_co", covariant=True)
StartTagInT_contra = TypeVar("StartTagInT_contra", contravariant=True)
StateOutT_co = TypeVar("StateOutT_co", covariant=True)


class TagNormalizer(Protocol[StartTagOutT_co]):
    """Protocol for normalizing raw start tags."""

    def __call__(self, tag: StateTag) -> StartTagOutT_co:
        """Normalize a start tag to the environment's expected type."""
        ...


class InitialStateFactory(Protocol[StateOutT_co, StartTagInT_contra]):
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
    ) -> "PlayerObserverFactory":
        """Build a player observer factory with the given configuration."""
        ...


@dataclass(frozen=True)
class Environment[StateT, StateSnapT, StartTagT]:
    """Bundle environment wiring and factories for a game kind."""

    game_kind: GameKind
    rules: GameRules[StateT]
    dynamics: Dynamics[StateT]
    gui_encoder: GuiEncoder[StateT]
    player_encoder: "PlayerRequestEncoder[StateT, StateSnapT]"
    make_player_observer_factory: PlayerObserverFactoryBuilder
    normalize_start_tag: TagNormalizer[StartTagT]
    make_initial_state: InitialStateFactory[StateT, StartTagT]

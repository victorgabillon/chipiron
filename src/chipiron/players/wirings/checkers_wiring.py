"""Module for checkers wiring."""

import random
from dataclasses import dataclass
from typing import Any, NoReturn

from valanga import Color

from chipiron.environments.checkers.types import (
    CheckersDynamics,
    CheckersRules,
    CheckersState,
)
from chipiron.environments.types import GameKind
from chipiron.players.adapters.checkers_adapter import CheckersAdapter
from chipiron.players.factory_pipeline import create_player_with_pipeline
from chipiron.players.game_player import GamePlayer
from chipiron.players.move_selector import factory as move_selector_factory
from chipiron.players.observer_wiring import ObserverWiring
from chipiron.players.player_args import PlayerFactoryArgs


@dataclass(frozen=True)
class BuildCheckersGamePlayerArgs:
    """Buildcheckersgameplayerargs implementation."""

    player_factory_args: PlayerFactoryArgs
    player_color: Color
    implementation_args: object
    universal_behavior: bool


def _no_checkers_master_evaluator(*_args: Any, **_kwargs: Any) -> NoReturn:
    raise NotImplementedError("Tree/value evaluator not implemented for checkers")


def build_checkers_game_player(
    args: BuildCheckersGamePlayerArgs,
) -> GamePlayer[str, CheckersState]:
    """Build checkers game player."""
    random_generator = random.Random(args.player_factory_args.seed)
    _ = args.universal_behavior

    rules = CheckersRules()
    dynamics = CheckersDynamics(rules)

    player_args = args.player_factory_args.player_args

    player = create_player_with_pipeline(
        name=player_args.name,
        main_selector_args=player_args.main_move_selector,
        state_type=CheckersState,
        policy_oracle=None,
        value_oracle=None,
        terminal_oracle=None,
        master_evaluator_from_args=_no_checkers_master_evaluator,
        adapter_builder=lambda selector, _policy_oracle: CheckersAdapter(
            dynamics=dynamics,
            main_move_selector=selector,
        ),
        create_non_tree_selector=lambda selector_args, dyn: (
            move_selector_factory.create_main_move_selector(
                selector_args,
                game_kind=GameKind.CHECKERS,
                dynamics=dyn,
                random_generator=random_generator,
            )
        ),
        random_generator=random_generator,
        runtime_dynamics=dynamics,
        implementation_args=None,
    )

    return GamePlayer(player, args.player_color)


CHECKERS_WIRING: ObserverWiring[str, CheckersState, BuildCheckersGamePlayerArgs] = (
    ObserverWiring(
        build_game_player=build_checkers_game_player,
        build_args_type=BuildCheckersGamePlayerArgs,
    )
)

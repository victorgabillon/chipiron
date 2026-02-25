"""Module for checkers wiring."""

import random
from dataclasses import dataclass

from valanga import Color

from chipiron.environments.checkers.checkers_rules import (
    CheckersRules as CheckersRulesAdapter,
)
from chipiron.environments.checkers.players.adapters.checkers_adapter import (
    CheckersAdapter,
)
from chipiron.environments.checkers.players.evaluators.checkers_piece_count import (
    CheckersMasterEvaluator,
    CheckersOverEventDetector,
    CheckersPieceCountEvaluator,
)
from chipiron.environments.checkers.types import (
    CheckersDynamics,
    CheckersRules,
    CheckersState,
)
from chipiron.environments.types import GameKind
from chipiron.players.boardevaluators.evaluation_scale import get_value_over_enum
from chipiron.players.boardevaluators.master_board_evaluator import (
    MasterBoardEvaluatorArgs,
)
from chipiron.players.factory_pipeline import create_player_with_pipeline
from chipiron.players.game_player import GamePlayer
from chipiron.players.move_selector import factory as move_selector_factory
from chipiron.players.observer_wiring import ObserverWiring
from chipiron.players.oracles import TerminalOracle, ValueOracle
from chipiron.players.player_args import PlayerFactoryArgs


@dataclass(frozen=True)
class BuildCheckersGamePlayerArgs:
    """Buildcheckersgameplayerargs implementation."""

    player_factory_args: PlayerFactoryArgs
    player_color: Color
    implementation_args: object
    universal_behavior: bool


def build_checkers_master_evaluator(
    evaluator_args: MasterBoardEvaluatorArgs,
    value_oracle: ValueOracle[CheckersState] | None,
    terminal_oracle: TerminalOracle[CheckersState] | None,
    rules_adapter: CheckersRulesAdapter,
) -> CheckersMasterEvaluator:
    """Build an anemone-compatible master evaluator for checkers."""
    del value_oracle
    del terminal_oracle
    value_over_enum = get_value_over_enum(evaluator_args.evaluation_scale)

    detector = CheckersOverEventDetector(
        rules=rules_adapter,
        value_over_enum=value_over_enum,
    )

    return CheckersMasterEvaluator(
        evaluator=CheckersPieceCountEvaluator(),
        over=detector,  # Protocol-typed field
        over_detector=detector,  # concrete field for pylint
    )


def build_checkers_game_player(
    args: BuildCheckersGamePlayerArgs,
) -> GamePlayer[str, CheckersState]:
    """Build checkers game player."""
    random_generator = random.Random(args.player_factory_args.seed)
    _ = args.universal_behavior

    atom_rules = CheckersRules()
    rules_adapter = CheckersRulesAdapter(inner=atom_rules)
    dynamics = CheckersDynamics(atom_rules)

    player_args = args.player_factory_args.player_args

    def master_evaluator_from_args(
        evaluator_args: MasterBoardEvaluatorArgs,
        value_oracle: ValueOracle[CheckersState] | None,
        terminal_oracle: TerminalOracle[CheckersState] | None,
    ) -> CheckersMasterEvaluator:
        return build_checkers_master_evaluator(
            evaluator_args=evaluator_args,
            value_oracle=value_oracle,
            terminal_oracle=terminal_oracle,
            rules_adapter=rules_adapter,
        )

    player = create_player_with_pipeline(
        name=player_args.name,
        main_selector_args=player_args.main_move_selector,
        state_type=CheckersState,
        policy_oracle=None,
        value_oracle=None,
        terminal_oracle=None,
        master_evaluator_from_args=master_evaluator_from_args,
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

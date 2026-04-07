"""Morpion wiring."""

import random
from dataclasses import dataclass

from anemone.node_evaluation.tree.single_agent.factory import (
    NodeMaxEvaluationFactory,
)
from valanga import SoloRole

from chipiron.core.oracles import TerminalOracle, ValueOracle
from chipiron.environments.morpion.players.adapters.morpion_adapter import (
    MorpionAdapter,
)
from chipiron.environments.morpion.players.evaluators.morpion_state_evaluator import (
    MorpionMasterEvaluator,
    build_morpion_master_evaluator,
)
from chipiron.environments.morpion.types import MorpionDynamics, MorpionState
from chipiron.environments.types import GameKind
from chipiron.players import Player
from chipiron.players.boardevaluators.master_board_evaluator_args import (
    MasterBoardEvaluatorArgs,
)
from chipiron.players.factory_pipeline import (
    create_player_with_standard_adapter_pipeline,
)
from chipiron.players.game_player import GamePlayer
from chipiron.players.move_selector import factory as move_selector_factory
from chipiron.players.move_selector.tree_and_value_args import TreeAndValueAppArgs
from chipiron.players.observer_wiring import ObserverWiring
from chipiron.players.player_args import PlayerFactoryArgs


@dataclass(frozen=True)
class BuildMorpionGamePlayerArgs:
    """Morpion game-player build arguments."""

    player_factory_args: PlayerFactoryArgs
    player_role: SoloRole
    implementation_args: object
    universal_behavior: bool


def build_morpion_game_player(
    args: BuildMorpionGamePlayerArgs,
) -> GamePlayer[MorpionState, MorpionState]:
    """Build Morpion game player."""
    random_generator = random.Random(args.player_factory_args.seed)
    _ = args.implementation_args
    _ = args.universal_behavior

    dynamics = MorpionDynamics()
    player_args = args.player_factory_args.player_args

    def master_evaluator_from_args(
        evaluator_args: MasterBoardEvaluatorArgs,
        value_oracle: ValueOracle[MorpionState] | None,
        terminal_oracle: TerminalOracle[MorpionState, SoloRole] | None,
    ) -> MorpionMasterEvaluator:
        del value_oracle
        del terminal_oracle
        return build_morpion_master_evaluator(
            evaluation_scale=evaluator_args.evaluation_scale,
        )

    main_selector_args = player_args.main_move_selector
    if not isinstance(main_selector_args, TreeAndValueAppArgs):
        player = create_player_with_standard_adapter_pipeline(
            name=player_args.name,
            main_selector_args=main_selector_args,
            state_type=MorpionState,
            policy_oracle=None,
            value_oracle=None,
            terminal_oracle=None,
            master_evaluator_from_args=master_evaluator_from_args,
            game_kind=GameKind.MORPION,
            random_generator=random_generator,
            runtime_dynamics=dynamics,
            adapter_factory=lambda selector: MorpionAdapter(
                dynamics=dynamics,
                main_move_selector=selector,
            ),
            implementation_args=None,
        )
        return GamePlayer(player, args.player_role)

    master_state_evaluator = master_evaluator_from_args(
        main_selector_args.evaluator_args.master_board_evaluator,
        None,
        None,
    )
    main_move_selector = move_selector_factory.create_tree_and_value_move_selector(
        args=main_selector_args.anemone_args,
        state_type=MorpionState,
        accelerate_when_winning=main_selector_args.accelerate_when_winning,
        master_state_value_evaluator=master_state_evaluator,
        node_tree_evaluation_factory=NodeMaxEvaluationFactory(),
        state_representation_factory=None,
        random_generator=random_generator,
        dynamics=dynamics,
        implementation_args=None,
    )

    player = Player(
        name=player_args.name,
        adapter=MorpionAdapter(
            dynamics=dynamics,
            main_move_selector=main_move_selector,
        ),
    )

    return GamePlayer(player, args.player_role)


MORPION_WIRING: ObserverWiring[
    MorpionState,
    MorpionState,
    BuildMorpionGamePlayerArgs,
] = ObserverWiring(
    build_game_player=build_morpion_game_player,
    build_args_type=BuildMorpionGamePlayerArgs,
)

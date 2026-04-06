"""Integer reduction wiring."""

import random
from dataclasses import dataclass

from valanga import SoloRole

from chipiron.core.oracles import TerminalOracle, ValueOracle
from chipiron.environments.integer_reduction.players.adapters.integer_reduction_adapter import (
    IntegerReductionAdapter,
)
from chipiron.environments.integer_reduction.players.evaluators.integer_reduction_state_evaluator import (
    IntegerReductionMasterEvaluator,
    build_integer_reduction_master_evaluator,
)
from chipiron.environments.integer_reduction.types import (
    IntegerReductionDynamics,
    IntegerReductionState,
)
from chipiron.environments.types import GameKind
from chipiron.players.boardevaluators.master_board_evaluator_args import (
    MasterBoardEvaluatorArgs,
)
from chipiron.players.factory_pipeline import (
    create_player_with_standard_adapter_pipeline,
)
from chipiron.players.game_player import GamePlayer
from chipiron.players.observer_wiring import ObserverWiring
from chipiron.players.player_args import PlayerFactoryArgs


@dataclass(frozen=True)
class BuildIntegerReductionGamePlayerArgs:
    """Integer reduction game-player build arguments."""

    player_factory_args: PlayerFactoryArgs
    player_role: SoloRole
    implementation_args: object
    universal_behavior: bool


def build_integer_reduction_game_player(
    args: BuildIntegerReductionGamePlayerArgs,
) -> GamePlayer[int, IntegerReductionState]:
    """Build integer-reduction game player."""
    random_generator = random.Random(args.player_factory_args.seed)
    _ = args.implementation_args
    _ = args.universal_behavior

    dynamics = IntegerReductionDynamics()
    player_args = args.player_factory_args.player_args

    def master_evaluator_from_args(
        evaluator_args: MasterBoardEvaluatorArgs,
        value_oracle: ValueOracle[IntegerReductionState] | None,
        terminal_oracle: TerminalOracle[IntegerReductionState, SoloRole] | None,
    ) -> IntegerReductionMasterEvaluator:
        del value_oracle
        del terminal_oracle
        return build_integer_reduction_master_evaluator(
            evaluation_scale=evaluator_args.evaluation_scale,
        )

    player = create_player_with_standard_adapter_pipeline(
        name=player_args.name,
        main_selector_args=player_args.main_move_selector,
        state_type=IntegerReductionState,
        policy_oracle=None,
        value_oracle=None,
        terminal_oracle=None,
        master_evaluator_from_args=master_evaluator_from_args,
        game_kind=GameKind.INTEGER_REDUCTION,
        random_generator=random_generator,
        runtime_dynamics=dynamics,
        adapter_factory=lambda selector: IntegerReductionAdapter(
            dynamics=dynamics,
            main_move_selector=selector,
        ),
        implementation_args=None,
    )

    return GamePlayer(player, args.player_role)


INTEGER_REDUCTION_WIRING: ObserverWiring[
    int,
    IntegerReductionState,
    BuildIntegerReductionGamePlayerArgs,
] = ObserverWiring(
    build_game_player=build_integer_reduction_game_player,
    build_args_type=BuildIntegerReductionGamePlayerArgs,
)

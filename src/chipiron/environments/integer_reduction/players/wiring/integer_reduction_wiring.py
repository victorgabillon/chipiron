"""Integer reduction wiring."""

import random
from dataclasses import dataclass
from datetime import datetime

from anemone.tree_and_value_branch_selector import TreeAndValueBranchSelector
from valanga import SoloRole

from chipiron.core.oracles import TerminalOracle, ValueOracle
from chipiron.debug.tree_search_debug_selector import DebugTreeSearchSelector
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
from chipiron.players import Player
from chipiron.players.boardevaluators.master_board_evaluator_args import (
    MasterBoardEvaluatorArgs,
)
from chipiron.players.move_selector import factory as move_selector_factory
from chipiron.players.move_selector.tree_and_value_args import TreeAndValueAppArgs
from chipiron.players.factory_pipeline import (
    create_player_with_standard_adapter_pipeline,
)
from chipiron.players.game_player import GamePlayer
from chipiron.players.observer_wiring import ObserverWiring
from chipiron.players.player_args import PlayerFactoryArgs
from chipiron.players.player_ids import PlayerConfigTag
from chipiron.utils.path_runtime import get_output_root


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

    main_selector_args = player_args.main_move_selector
    if not isinstance(main_selector_args, TreeAndValueAppArgs):
        player = create_player_with_standard_adapter_pipeline(
            name=player_args.name,
            main_selector_args=main_selector_args,
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

    master_state_evaluator = master_evaluator_from_args(
        main_selector_args.evaluator_args.master_board_evaluator,
        None,
        None,
    )
    main_move_selector = move_selector_factory.create_tree_and_value_move_selector(
        args=main_selector_args.anemone_args,
        state_type=IntegerReductionState,
        accelerate_when_winning=main_selector_args.accelerate_when_winning,
        master_state_value_evaluator=master_state_evaluator,
        state_representation_factory=None,
        random_generator=random_generator,
        dynamics=dynamics,
        implementation_args=None,
    )
    if _is_debug_tree_search_player(player_args.name):
        if not isinstance(main_move_selector, TreeAndValueBranchSelector):
            raise TypeError(
                "Integer-reduction debug wrapping requires a "
                "TreeAndValueBranchSelector."
            )
        main_move_selector = DebugTreeSearchSelector(
            base=main_move_selector,
            session_root=_make_debug_session_root(
                seed=args.player_factory_args.seed,
            ),
            state_to_debug_string=lambda state: str(state.value),
        )

    player = Player(
        name=player_args.name,
        adapter=IntegerReductionAdapter(
            dynamics=dynamics,
            main_move_selector=main_move_selector,
        ),
    )

    return GamePlayer(player, args.player_role)


def _is_debug_tree_search_player(player_name: str) -> bool:
    """Return whether the parsed player args correspond to the debug tree config."""
    return player_name == PlayerConfigTag.INTEGER_REDUCTION_TREE_BASIC_DEBUG.value


def _make_debug_session_root(*, seed: int) -> str:
    """Return the match-level root used for per-move integer-reduction sessions."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return str(
        get_output_root()
        / "runs"
        / "debug"
        / "integer_reduction"
        / f"{timestamp}_seed_{seed}"
    )


INTEGER_REDUCTION_WIRING: ObserverWiring[
    int,
    IntegerReductionState,
    BuildIntegerReductionGamePlayerArgs,
] = ObserverWiring(
    build_game_player=build_integer_reduction_game_player,
    build_args_type=BuildIntegerReductionGamePlayerArgs,
)

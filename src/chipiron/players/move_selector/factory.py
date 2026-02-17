"""Document the module provides a factory function for creating the main move selector based on the given arguments."""

import random
from collections.abc import Mapping
from typing import Any

from anemone import TreeAndValuePlayerArgs, create_tree_and_value_branch_selector
from anemone.dynamics import SearchDynamics, normalize_search_dynamics
from anemone.hooks.search_hooks import PriorityCheckFactory, SearchHooks
from anemone.node_evaluation.node_direct_evaluation.node_direct_evaluator import (
    MasterStateEvaluator,
)
from anemone.node_selector.opening_instructions import OpeningInstructor
from valanga import (
    Dynamics,
    RepresentationFactory,
    StateModifications,
    TurnState,
)
from valanga.evaluator_types import EvaluatorInput
from valanga.policy import BranchSelector

from chipiron.environments.types import GameKind
from chipiron.players.move_selector.move_selector_args import NonTreeMoveSelectorArgs
from chipiron.utils.logger import chipiron_logger

from .anemone_hooks import ChessFeatureExtractor
from .modifiers import (
    AccelerateWhenWinning,
    ComposedBranchSelector,
    chess_progress_gain_zeroing,
)
from .priority_checks.pv_attacked_open_all import PvAttackedOpenAllPriorityCheck
from .random_args import RandomSelectorArgs
from .random_selector import create_random_selector
from .registry import get_game_specific_selector_factory


class MissingTreeSearchDynamicsError(ValueError):
    """Raised when tree search dynamics cannot be built from provided inputs."""

    DEFAULT_MESSAGE = "Tree search requires `dynamics` or `search_dynamics_override`."


class MissingGameSpecificSelectorFactoryError(ValueError):
    """Raised when no game-specific selector factory is registered for a game kind."""

    def __init__(self, game_kind: GameKind) -> None:
        """Initialize the error with the missing game kind."""
        super().__init__(
            f"No game-specific selector factory registered for {game_kind}"
        )


def create_main_move_selector[TurnStateT: TurnState](
    move_selector_args: NonTreeMoveSelectorArgs,
    *,
    game_kind: GameKind,
    dynamics: Dynamics[TurnStateT],
    random_generator: random.Random,
) -> BranchSelector[TurnStateT]:
    """Create the main move selector based on the given arguments.

    Args:
        move_selector_args (NonTreeMoveSelectorArgs): Move selector args parsed from config.
        game_kind (GameKind): The kind of game (CHESS, etc.) to determine game-specific handling.
        dynamics (Dynamics[TurnStateT]): The game dynamics.
        random_generator (random.Random): The random number generator.

    Returns:
        BranchSelector: The main move selector.

    Raises:
        ValueError: If the given move selector instance or arguments are invalid.

    """
    main_move_selector: BranchSelector[TurnStateT]
    chipiron_logger.debug("Create main move selector")

    # Handle generic selectors
    match move_selector_args:
        case RandomSelectorArgs():
            main_move_selector = create_random_selector(
                dynamics=dynamics,
                rng=random_generator,
            )
        case _:
            # Delegate to game-specific factory
            factory = get_game_specific_selector_factory(game_kind)
            if factory is None:
                raise MissingGameSpecificSelectorFactoryError(game_kind)
            main_move_selector = factory(move_selector_args, dynamics, random_generator)

    return main_move_selector


def create_tree_and_value_move_selector[TurnStateT: TurnState](
    args: TreeAndValuePlayerArgs,
    *,
    state_type: type[TurnStateT],
    accelerate_when_winning: bool = False,
    master_state_evaluator: MasterStateEvaluator,
    state_representation_factory: (
        RepresentationFactory[TurnStateT, StateModifications, EvaluatorInput] | None
    ),
    random_generator: random.Random,
    dynamics: Dynamics[TurnStateT] | None = None,
    search_dynamics_override: SearchDynamics[TurnStateT, Any] | None = None,
) -> BranchSelector[TurnStateT]:
    """Create a tree-and-value move selector with a prebuilt evaluator."""

    def pv_attacked_open_all_factory(
        params: Mapping[str, Any],
        random_generator: random.Random,
        hooks: SearchHooks | None,
        opening_instructor: OpeningInstructor,
    ) -> PvAttackedOpenAllPriorityCheck:
        feature_extractor = hooks.feature_extractor if hooks is not None else None
        return PvAttackedOpenAllPriorityCheck(
            opening_instructor=opening_instructor,
            feature_extractor=feature_extractor,
            random_generator=random_generator,
            probability=float(params.get("probability", 0.5)),
            feature_key=str(params.get("feature_key", "tactical_threat")),
        )

    priority_check_registry: dict[str, PriorityCheckFactory] = {
        "pv_attacked_open_all": pv_attacked_open_all_factory,
    }

    hooks = SearchHooks(
        feature_extractor=ChessFeatureExtractor(),
        priority_check_registry=priority_check_registry,
    )

    search_dynamics: SearchDynamics[TurnStateT, Any]
    if search_dynamics_override is not None:
        search_dynamics = search_dynamics_override
    else:
        if dynamics is None:
            raise MissingTreeSearchDynamicsError(
                MissingTreeSearchDynamicsError.DEFAULT_MESSAGE
            )
        search_dynamics = normalize_search_dynamics(dynamics)

    base_selector = create_tree_and_value_branch_selector(
        state_type=state_type,
        master_state_evaluator=master_state_evaluator,
        state_representation_factory=state_representation_factory,
        args=args,
        random_generator=random_generator,
        hooks=hooks,
        dynamics=search_dynamics,
    )

    if accelerate_when_winning:
        return ComposedBranchSelector(
            base=base_selector,
            dynamics=search_dynamics,
            modifiers=(
                AccelerateWhenWinning(
                    progress_gain_fn=chess_progress_gain_zeroing,
                ),
            ),
        )
    return base_selector

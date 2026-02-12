"""Document the module provides a factory function for creating the main move selector based on the given arguments."""

import random
from collections.abc import Mapping
from typing import Any, TypeVar

from anemone import TreeAndValuePlayerArgs, create_tree_and_value_branch_selector
from anemone.hooks.search_hooks import PriorityCheckFactory, SearchHooks
from anemone.node_evaluation.node_direct_evaluation.node_direct_evaluator import (
    MasterStateEvaluator,
)
from anemone.node_selector.opening_instructions import OpeningInstructor
from valanga import RepresentationFactory, StateModifications, TurnState
from valanga.evaluator_types import EvaluatorInput
from valanga.policy import BranchSelector

from chipiron.environments.chess.types import ChessState
from chipiron.players.move_selector.move_selector_args import NonTreeMoveSelectorArgs
from chipiron.utils.logger import chipiron_logger

from . import human, stockfish
from .anemone_hooks import ChessFeatureExtractor
from .modifiers import (
    AccelerateWhenWinning,
    ComposedBranchSelector,
    chess_progress_gain_zeroing,
)
from .priority_checks.pv_attacked_open_all import PvAttackedOpenAllPriorityCheck
from .random import Random, create_random

TurnStateT = TypeVar("TurnStateT", bound=TurnState)


def create_main_move_selector(
    move_selector_instance_or_args: NonTreeMoveSelectorArgs,
    *,
    random_generator: random.Random,
) -> BranchSelector[ChessState]:
    """Create the main move selector based on the given arguments.

    Args:
        move_selector_instance_or_args (NonTreeMoveSelectorArgs): The arguments or instance of the move selector.
        random_generator (random.Random): The random number generator.

    Returns:
        BranchSelector: The main move selector.

    Raises:
        ValueErr    or: If the given move selector instance or arguments are invalid.

    """
    main_move_selector: BranchSelector[ChessState]
    chipiron_logger.debug("Create main move selector")

    match move_selector_instance_or_args:
        case Random():
            main_move_selector = create_random(random_generator=random_generator)
        case stockfish.StockfishPlayer():
            main_move_selector = move_selector_instance_or_args
        case human.CommandLineHumanPlayerArgs():
            main_move_selector = human.CommandLineHumanMoveSelector()

    return main_move_selector


def create_tree_and_value_move_selector(
    args: TreeAndValuePlayerArgs,
    *,
    state_type: type[TurnStateT],
    accelerate_when_winning: bool = False,
    master_state_evaluator: MasterStateEvaluator,
    state_representation_factory: (
        RepresentationFactory[TurnStateT, StateModifications, EvaluatorInput] | None
    ),
    random_generator: random.Random,
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

    base_selector = create_tree_and_value_branch_selector(
        state_type=state_type,
        master_state_evaluator=master_state_evaluator,
        state_representation_factory=state_representation_factory,
        args=args,
        random_generator=random_generator,
        hooks=hooks,
    )

    if accelerate_when_winning:
        return ComposedBranchSelector(
            base=base_selector,
            modifiers=(
                AccelerateWhenWinning(
                    progress_gain_fn=chess_progress_gain_zeroing,
                ),
            ),
        )
    return base_selector

"""Document the module provides a factory function for creating the main move selector based on the given arguments."""

import random
from typing import TypeVar

from anemone import TreeAndValuePlayerArgs, create_tree_and_value_branch_selector
from anemone.node_evaluation.node_direct_evaluation.node_direct_evaluator import (
    MasterStateEvaluator,
)
from valanga import RepresentationFactory, StateModifications, TurnState
from valanga.evaluator_types import EvaluatorInput
from valanga.policy import BranchSelector

from chipiron.players.move_selector.move_selector_args import NonTreeMoveSelectorArgs
from chipiron.utils.logger import chipiron_logger

from . import human, stockfish
from .modifiers import (
    AccelerateWhenWinning,
    ComposedBranchSelector,
    chess_progress_gain_zeroing,
)
from .random import Random, create_random

TurnStateT = TypeVar("TurnStateT", bound=TurnState)


def create_main_move_selector(
    move_selector_instance_or_args: NonTreeMoveSelectorArgs,
    *,
    random_generator: random.Random,
) -> BranchSelector[TurnState]:
    """Create the main move selector based on the given arguments.

    Args:
        move_selector_instance_or_args (NonTreeMoveSelectorArgs): The arguments or instance of the move selector.
        random_generator (random.Random): The random number generator.

    Returns:
        BranchSelector: The main move selector.

    Raises:
        ValueErr    or: If the given move selector instance or arguments are invalid.

    """
    main_move_selector: BranchSelector[TurnState]
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
    base_selector = create_tree_and_value_branch_selector(
        state_type=state_type,
        master_state_evaluator=master_state_evaluator,
        state_representation_factory=state_representation_factory,
        args=args,
        random_generator=random_generator,
    )

    if accelerate_when_winning and hasattr(state_type, "is_zeroing"):
        return ComposedBranchSelector(
            base=base_selector,
            modifiers=(
                AccelerateWhenWinning(
                    progress_gain_fn=chess_progress_gain_zeroing,
                ),
            ),
        )
    return base_selector

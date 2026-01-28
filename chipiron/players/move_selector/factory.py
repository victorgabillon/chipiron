"""
This module provides a factory function for creating the main move selector based on the given arguments.
"""

from __future__ import annotations

import random
from queue import Queue
from typing import TYPE_CHECKING, TypeAlias, TypeVar, cast

from anemone import TreeAndValuePlayerArgs, create_tree_and_value_branch_selector
from anemone.utils.dataclass import IsDataclass as AnemoneIsDataclass
from valanga import TurnState
from valanga.policy import BranchSelector

from chipiron.environments.chess.types import ChessState
from chipiron.games.game.game_manager import MainMailboxMessage
from chipiron.utils.logger import chipiron_logger

from ...utils.queue_protocols import PutQueue
from . import human, stockfish
from .random import Random, create_random

NonTreeMoveSelectorArgs: TypeAlias = (
    human.CommandLineHumanPlayerArgs | Random | stockfish.StockfishPlayer
)

TurnStateT = TypeVar("TurnStateT", bound=TurnState)

if TYPE_CHECKING:
    from anemone.node_evaluation.node_direct_evaluation.node_direct_evaluator import (
        MasterStateEvaluator,
    )
    from valanga import RepresentationFactory, StateModifications
    from valanga.evaluator_types import EvaluatorInput


def create_main_move_selector(
    move_selector_instance_or_args: NonTreeMoveSelectorArgs,
    *,
    random_generator: random.Random,
) -> BranchSelector[ChessState]:
    """
    Create the main move selector based on the given arguments.

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
    master_state_evaluator: MasterStateEvaluator,
    state_representation_factory: (
        RepresentationFactory[TurnStateT, StateModifications, EvaluatorInput] | None
    ),
    random_generator: random.Random,
    queue_progress_player: PutQueue[MainMailboxMessage] | None,
) -> BranchSelector[TurnStateT]:
    """
    Create a tree-and-value move selector with a prebuilt evaluator.
    """
    return create_tree_and_value_branch_selector(
        state_type=state_type,
        master_state_evaluator=master_state_evaluator,
        state_representation_factory=state_representation_factory,
        args=args,
        random_generator=random_generator,
        queue_progress_player=cast(
            "Queue[AnemoneIsDataclass] | None", queue_progress_player
        ),
    )

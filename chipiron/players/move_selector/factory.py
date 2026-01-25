"""
This module provides a factory function for creating the main move selector based on the given arguments.
"""

import random
from typing import Type, TypeAlias

from anemone import TreeAndValuePlayerArgs, create_tree_and_value_branch_selector
from valanga import State
from valanga.policy import BranchSelector

from chipiron.players.boardevaluators.master_board_evaluator import (
    create_master_board_evaluator,
)
from chipiron.players.boardevaluators.table_base.factory import AnySyzygyTable
from chipiron.utils.logger import chipiron_logger

from ...utils.dataclass import IsDataclass
from ...utils.queue_protocols import PutQueue
from . import human, stockfish
from .random import Random, create_random

AllMoveSelectorArgs: TypeAlias = (
    TreeAndValuePlayerArgs
    | human.CommandLineHumanPlayerArgs
    | human.GuiHumanPlayerArgs
    | Random
    | stockfish.StockfishPlayer
)


def create_main_move_selector[StateT: State](
    move_selector_instance_or_args: AllMoveSelectorArgs,
    syzygy: AnySyzygyTable | None,
    random_generator: random.Random,
    queue_progress_player: PutQueue[IsDataclass] | None,
) -> BranchSelector:
    """
    Create the main move selector based on the given arguments.

    Args:
        move_selector_instance_or_args (AllMoveSelectorArgs): The arguments or instance of the move selector.
        syzygy (AnySyzygyTable | None): The syzygy table.
        random_generator (random.Random): The random number generator.

    Returns:
        BranchSelector: The main move selector.

    Raises:
        ValueErr    or: If the given move selector instance or arguments are invalid.

    """
    main_move_selector: BranchSelector
    chipiron_logger.debug("Create main move selector")

    match move_selector_instance_or_args:
        case Random():
            main_move_selector = create_random(random_generator=random_generator)
        case TreeAndValuePlayerArgs():
            tree_args: TreeAndValuePlayerArgs = move_selector_instance_or_args
            main_state_evaluator = create_master_board_evaluator(
                board_evaluator=tree_args.board_evaluator,
                syzygy=syzygy,
                evaluation_scale=tree_args.evaluation_scale,
            )
            main_move_selector = create_tree_and_value_branch_selector(
                state_type=Type(StateT),
                master_state_evaluator=main_state_evaluator,
                args=tree_args,
                syzygy=syzygy,
                random_generator=random_generator,
                queue_progress_player=queue_progress_player,
            )
        case stockfish.StockfishPlayer():
            main_move_selector = move_selector_instance_or_args
        case human.CommandLineHumanPlayerArgs():
            main_move_selector = human.CommandLineHumanMoveSelector()
        case other:
            raise ValueError(
                f"player creator: can not find {other} of type {type(other)}"
            )
    return main_move_selector

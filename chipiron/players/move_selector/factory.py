"""
This module provides a factory function for creating the main move selector based on the given arguments.
"""

import queue
import random
from typing import Any, TypeAlias

from chipiron.players.boardevaluators.table_base.syzygy_table import SyzygyTable
from chipiron.utils.logger import chipiron_logger

from ...utils.dataclass import IsDataclass
from . import human, move_selector, stockfish, treevalue
from .random import Random, create_random

AllMoveSelectorArgs: TypeAlias = (
    treevalue.TreeAndValuePlayerArgs
    | human.CommandLineHumanPlayerArgs
    | human.GuiHumanPlayerArgs
    | Random
    | stockfish.StockfishPlayer
)


def create_main_move_selector(
    move_selector_instance_or_args: AllMoveSelectorArgs,
    syzygy: SyzygyTable[Any] | None,
    random_generator: random.Random,
    queue_progress_player: queue.Queue[IsDataclass] | None,
) -> move_selector.MoveSelector:
    """
    Create the main move selector based on the given arguments.

    Args:
        move_selector_instance_or_args (AllMoveSelectorArgs): The arguments or instance of the move selector.
        syzygy (SyzygyTable | None): The syzygy table.
        random_generator (random.Random): The random number generator.

    Returns:
        move_selector.MoveSelector: The main move selector.

    Raises:
        ValueError: If the given move selector instance or arguments are invalid.

    """
    main_move_selector: move_selector.MoveSelector
    chipiron_logger.info("Create main move selector")

    match move_selector_instance_or_args:
        case Random():
            main_move_selector = create_random(random_generator=random_generator)
        case treevalue.TreeAndValuePlayerArgs():
            tree_args: treevalue.TreeAndValuePlayerArgs = move_selector_instance_or_args
            main_move_selector = treevalue.create_tree_and_value_builders(
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

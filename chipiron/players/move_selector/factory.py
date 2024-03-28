import random
from typing import TypeAlias

from chipiron.players.boardevaluators.table_base.syzygy import SyzygyTable
from . import human
from . import move_selector
from . import stockfish
from . import treevalue
from .random import Random, create_random

AllMoveSelectorArgs: TypeAlias = (
        treevalue.TreeAndValuePlayerArgs |
        human.CommandLineHumanPlayerArgs |
        human.GuiHumanPlayerArgs |
        Random |
        stockfish.StockfishPlayer
)


def create_main_move_selector(
        move_selector_instance_or_args: AllMoveSelectorArgs,
        syzygy: SyzygyTable | None,
        random_generator: random.Random
) -> move_selector.MoveSelector:
    main_move_selector: move_selector.MoveSelector
    print('create main move selector')

    match move_selector_instance_or_args:
        case Random():
            main_move_selector = create_random(
                random_generator=random_generator
            )
        case treevalue.TreeAndValuePlayerArgs():
            tree_args: treevalue.TreeAndValuePlayerArgs = move_selector_instance_or_args
            main_move_selector = treevalue.create_tree_and_value_builders(
                args=tree_args,
                syzygy=syzygy,
                random_generator=random_generator
            )
        case stockfish.StockfishPlayer():
            main_move_selector = move_selector_instance_or_args
        case human.CommandLineHumanPlayerArgs():
            main_move_selector = human.CommandLineHumanMoveSelector()
        case other:
            raise ValueError(f'player creator: can not find {other} of type {type(other)}')
    return main_move_selector

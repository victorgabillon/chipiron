from .move_selector_types import MoveSelectorTypes
from chipiron.utils.null_object import NullObject

import random

from . import move_selector
from . import treevalue
from . import human
from . import stockfish
from .random import Random, create_random
from typing import TypeAlias

AllMoveSelectorArgs: TypeAlias = (treevalue.TreeAndValuePlayerArgs
                                  | human.HumanPlayerArgs | Random |
                                  stockfish.StockfishPlayer)


def create_main_move_selector(
        move_selector_instance_or_args: AllMoveSelectorArgs,
        syzygy,
        random_generator: random.Random
) -> move_selector.MoveSelector:
    main_move_selector: move_selector.MoveSelector
    print('create main move')

    match move_selector_instance_or_args:
        case Random():
            main_move_selector = create_random(
                random_generator=random_generator
            )
        case treevalue.TreeAndValuePlayerArgs():
            tree_args : treevalue.TreeAndValuePlayerArgs = move_selector_instance_or_args
            main_move_selector = treevalue.create_tree_and_value_builders(
                args=tree_args,
                syzygy=syzygy,
                random_generator=random_generator
            )
        case stockfish.StockfishPlayer():
            main_move_selector = move_selector_instance_or_args
        case human.HumanPlayerArgs():
            main_move_selector = NullObject()  # TODO is it necessary?
        case other:
            raise ValueError(f'player creator: can not find {other} of type {type(other)}')
    return main_move_selector

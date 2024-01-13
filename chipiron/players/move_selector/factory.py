from .move_selector_types import MoveSelectorTypes
from chipiron.utils.null_object import NullObject

import random

from . import move_selector
from . import treevalue
from . import human
from . import stockfish
AllMoveSelectorArgs = treevalue.TreeAndValuePlayerArgs | human.HumanPlayerArgs| stockfish.StockfishPlayer


def create_main_move_selector(
        move_selector_instance_or_args: AllMoveSelectorArgs,
        syzygy,
        random_generator: random.Random
) -> move_selector.MoveSelector:
    main_move_selector: move_selector.MoveSelector
    print('create main move')

    match move_selector_instance_or_args.type:
        case 'RandomPlayer':
            main_move_selector = move_selector.Random()
        case MoveSelectorTypes.TreeAndValue:
            main_move_selector = treevalue.create_tree_and_value_builders(args=move_selector_instance_or_args,
                                                                          syzygy=syzygy,
                                                                          random_generator=random_generator)
        case 'Stockfish':
            main_move_selector = move_selector_instance_or_args
        case 'Human':
            main_move_selector = NullObject()  # TODO is it necessary?
        case other:
            raise ValueError(f'player creator: can not find {other} of type {type(other)}')
    return main_move_selector

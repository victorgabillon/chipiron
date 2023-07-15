"""
Factory to build node selectors
"""
from . import NodeSelector
from .uniform import Uniform
from .recur_zipf_base import RecurZipfBase
from .sequool2 import Sequool
import chipiron.players.treevalue.trees as trees


def create(
        arg: dict,
        opening_instructor,
        random_generator,
) -> NodeSelector:
    """
    Creation of a node selector
    """

    tree_builder_type: str = arg['type']
    node_move_opening_selector: NodeSelector

    match tree_builder_type:
        case 'Uniform':
            node_move_opening_selector = Uniform(opening_instructor=opening_instructor)
        case 'RecurZipfBase':
            node_move_opening_selector = RecurZipfBase(
                arg=arg,
                random_generator=random_generator,
                opening_instructor=opening_instructor
            )

        case 'Sequool':
            node_move_opening_selector = Sequool(
                opening_instructor=opening_instructor,
                all_nodes_not_opened=trees.RangedDescendants()
            )
        case other:
            raise Exception('tree builder: can not find ' + other)

    return node_move_opening_selector

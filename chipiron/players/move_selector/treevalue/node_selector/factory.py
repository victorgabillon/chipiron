"""
Factory to build node selectors
"""
from .node_selector import NodeSelector
from .uniform import Uniform
from .recur_zipf_base import RecurZipfBase
from .sequool import create_sequool
from .opening_instructions import OpeningInstructor
from dataclasses import dataclass
from enum import Enum


class NodeSelectorType(Enum):
    RecurZipfBase: str = 'RecurZipfBase'
    Sequool: str = 'Sequool'
    Uniform: str = 'Uniform'


@dataclass
class NodeSelectorArgs:
    type: NodeSelectorType


def create(
        arg: NodeSelectorArgs,
        opening_instructor: OpeningInstructor,
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
            node_move_opening_selector = create_sequool(opening_instructor=opening_instructor,
                                                        args=arg['sequool'])

        case other:
            raise Exception('tree builder: can not find ' + other)

    return node_move_opening_selector

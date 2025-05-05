"""
Factory to build node selectors
"""

import random
from dataclasses import dataclass
from typing import Literal, TypeAlias

from .node_selector import NodeSelector
from .node_selector_types import NodeSelectorType
from .opening_instructions import OpeningInstructor
from .recurzipf.recur_zipf_base import RecurZipfBase, RecurZipfBaseArgs
from .sequool import SequoolArgs, create_sequool
from .uniform import Uniform


@dataclass
class UniformArgs:
    """
    Arguments for the Uniform node selector.

    """

    type: Literal[NodeSelectorType.Uniform]


AllNodeSelectorArgs: TypeAlias = RecurZipfBaseArgs | SequoolArgs | UniformArgs


def create(
    args: AllNodeSelectorArgs,
    opening_instructor: OpeningInstructor,
    random_generator: random.Random,
) -> NodeSelector:
    """
    Creation of a node selector
    """

    node_move_opening_selector: NodeSelector

    match args.type:
        case NodeSelectorType.Uniform:
            node_move_opening_selector = Uniform(opening_instructor=opening_instructor)
        case NodeSelectorType.RecurZipfBase:
            assert isinstance(args, RecurZipfBaseArgs)
            node_move_opening_selector = RecurZipfBase(
                args=args,
                random_generator=random_generator,
                opening_instructor=opening_instructor,
            )

        case NodeSelectorType.Sequool:
            assert isinstance(args, SequoolArgs)
            node_move_opening_selector = create_sequool(
                opening_instructor=opening_instructor,
                random_generator=random_generator,
                args=args,
            )

        case other:
            raise ValueError(
                f"node selector construction: can not find {other}  {args} in file {__name__}"
            )

    return node_move_opening_selector

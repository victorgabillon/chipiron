"""
Factory to build node selectors
"""
from .node_selector import NodeSelector
from .uniform import Uniform
from .recurzipf.recur_zipf_base import RecurZipfBase, RecurZipfBaseArgs
from .sequool import create_sequool, SequoolArgs
from .opening_instructions import OpeningInstructor
from .node_selector_types import NodeSelectorType
from .node_selector_args import NodeSelectorArgs

AllNodeSelectorArgs = RecurZipfBaseArgs | SequoolArgs | NodeSelectorArgs


def create(
        args: AllNodeSelectorArgs,
        opening_instructor: OpeningInstructor,
        random_generator,
) -> NodeSelector:
    """
    Creation of a node selector
    """

    node_move_opening_selector: NodeSelector

    match args.type:
        case NodeSelectorType.Uniform:
            node_move_opening_selector = Uniform(opening_instructor=opening_instructor)
        case NodeSelectorType.RecurZipfBase:
            node_move_opening_selector = RecurZipfBase(
                args=args,
                random_generator=random_generator,
                opening_instructor=opening_instructor
            )

        case NodeSelectorType.Sequool:
            node_move_opening_selector = create_sequool(opening_instructor=opening_instructor,
                                                        args=args)

        case other:
            raise ValueError(f'node selector construction: can not find {other}  {args} in file {__name__}')

    return node_move_opening_selector
"""
factory for sequool node selector
"""
import chipiron.players.move_selector.treevalue.trees as trees
from .sequool2 import Sequool
from chipiron.players.move_selector.treevalue.node_selector.opening_instructions import OpeningInstructor
from chipiron.players.move_selector.treevalue.node_indices.types import IndexComputationType
from dataclasses import dataclass
from ..node_selector_args import NodeSelectorArgs


@dataclass
class SequoolArgs(NodeSelectorArgs):
    index_computation: IndexComputationType


def create_sequool(
        opening_instructor: OpeningInstructor,
        args: SequoolArgs
) -> Sequool:
    """

    Args:
        opening_instructor: an opening instructor object
        args: dictionary of arguments

    Returns: a sequool node selector object

    """

    sequool: Sequool = Sequool(
        opening_instructor=opening_instructor,
        all_nodes_not_opened=trees.RangedDescendants(),
    )

    return sequool

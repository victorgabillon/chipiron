"""
factory for sequool node selector
"""
import chipiron.players.move_selector.treevalue.trees as trees
from .sequool2 import Sequool
from chipiron.players.move_selector.treevalue.node_selector.opening_instructions import OpeningInstructor
from .index_computation import UpdateAllIndices, update_all_indices_base, IndexComputationType, update_all_indices_recurzipfsequool
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
    update_all_indices: UpdateAllIndices
    match args.index_computation:
        case IndexComputationType.MinGlobalChange:
            update_all_indices = update_all_indices_base
        case IndexComputationType.RecurZipf:
            update_all_indices = update_all_indices_recurzipfsequool
        case IndexComputationType.MinLocalChange:
            update_all_indices = update_all_indices_recurzipfsequool
        case other:
            raise ValueError(f'player creator: can not find {other} in {__name__}')

    sequool: Sequool = Sequool(
        opening_instructor=opening_instructor,
        all_nodes_not_opened=trees.RangedDescendants(),
        update_all_indices=update_all_indices
    )

    return sequool

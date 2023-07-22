import chipiron.players.treevalue.trees as trees
from .sequool2 import Sequool
from chipiron.players.treevalue.node_selector.opening_instructions import OpeningInstructor
from .index_computation import UpdateAllIndices, update_all_indices_base


def create_sequool(
        opening_instructor: OpeningInstructor,
        args: dict
) -> Sequool:
    update_all_indices: UpdateAllIndices
    match args['index_computation']:
        case 'base':
            update_all_indices = update_all_indices_base
        case other:
            raise Exception(f'player creator: can not find {other}')

    sequool: Sequool = Sequool(
        opening_instructor=opening_instructor,
        all_nodes_not_opened=trees.RangedDescendants(),
        update_all_indices=update_all_indices
    )

    return sequool

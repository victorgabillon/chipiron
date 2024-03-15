from dataclasses import make_dataclass
from typing import Callable, Type, Any

import chipiron.players.move_selector.treevalue.nodes as nodes
from chipiron.players.move_selector.treevalue.indices.node_indices.index_data import NodeExplorationData, \
    RecurZipfQuoolExplorationData, MinMaxPathValue, IntervalExplo, \
    MaxDepthDescendants
from chipiron.players.move_selector.treevalue.indices.node_indices.index_types import IndexComputationType

ExplorationIndexDataFactory = Callable[[nodes.TreeNode], NodeExplorationData | None]


def create_exploration_index_data(
        tree_node: nodes.TreeNode,
        index_computation: IndexComputationType | None = None,
        depth_index: bool = False
) -> NodeExplorationData | None:
    exploration_index_data: NodeExplorationData | None
    base_index_dataclass_name: Type[NodeExplorationData] | None
    match index_computation:
        case None:
            base_index_dataclass_name = None
        case IndexComputationType.MinLocalChange:
            base_index_dataclass_name = IntervalExplo
            # exploration_index_data: IntervalExplo = IntervalExplo(tree_node=tree_node)
        case IndexComputationType.MinGlobalChange:
            base_index_dataclass_name = MinMaxPathValue
            # exploration_index_data: MinMaxPathValue = MinMaxPathValue(tree_node=tree_node)
        case IndexComputationType.RecurZipf:
            base_index_dataclass_name = RecurZipfQuoolExplorationData
            # exploration_index_data = RecurZipfQuoolExplorationData(tree_node=tree_node)
        case other:
            raise ValueError(f'not finding good case for {other} in file {__name__}')

    index_dataclass_name: Any
    if depth_index:
        assert base_index_dataclass_name is not None
        # adding a field to the dataclass for keeping track of the depth
        index_dataclass_name = make_dataclass('DepthExtendedDataclass',
                                              fields=[],
                                              bases=(base_index_dataclass_name, MaxDepthDescendants))
    else:
        index_dataclass_name = base_index_dataclass_name

    if index_dataclass_name is not None:
        exploration_index_data = index_dataclass_name(tree_node=tree_node)
    else:
        exploration_index_data = None

    return exploration_index_data

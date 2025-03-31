"""
This module provides functions for creating exploration index data for tree nodes.

The main function in this module is `create_exploration_index_data`, which takes a tree node and optional parameters
to create the exploration index data for that node.

The module also defines the `ExplorationIndexDataFactory` type, which is a callable type for creating exploration index data.

Functions:
- create_exploration_index_data: Creates exploration index data for a given tree node.

Types:
- ExplorationIndexDataFactory: A callable type for creating exploration index data.
"""

from dataclasses import make_dataclass
from typing import Any, Callable, Type

from chipiron.players.move_selector.treevalue.indices.node_indices.index_data import (
    IntervalExplo,
    MaxDepthDescendants,
    MinMaxPathValue,
    NodeExplorationData,
    RecurZipfQuoolExplorationData,
)
from chipiron.players.move_selector.treevalue.indices.node_indices.index_types import (
    IndexComputationType,
)
from chipiron.players.move_selector.treevalue.nodes.tree_node import TreeNode

ExplorationIndexDataFactory = Callable[[TreeNode[Any]], NodeExplorationData | None]


def create_exploration_index_data(
    tree_node: TreeNode[Any],
    index_computation: IndexComputationType | None = None,
    depth_index: bool = False,
) -> NodeExplorationData | None:
    """
    Creates exploration index data for a given tree node.

    Args:
        tree_node (TreeNode): The tree node for which to create the exploration index data.
        index_computation (IndexComputationType | None, optional): The type of index computation to use. Defaults to None.
        depth_index (bool, optional): Whether to include depth information in the index data. Defaults to False.

    Returns:
        NodeExplorationData | None: The created exploration index data.

    Raises:
        ValueError: If the index_computation value is not recognized.
    """
    exploration_index_data: NodeExplorationData | None
    base_index_dataclass_name: Type[NodeExplorationData] | None
    match index_computation:
        case None:
            base_index_dataclass_name = None
        case IndexComputationType.MinLocalChange:
            base_index_dataclass_name = IntervalExplo
        case IndexComputationType.MinGlobalChange:
            base_index_dataclass_name = MinMaxPathValue
        case IndexComputationType.RecurZipf:
            base_index_dataclass_name = RecurZipfQuoolExplorationData
        case other:
            raise ValueError(f"not finding good case for {other} in file {__name__}")

    index_dataclass_name: Any
    if depth_index:
        assert base_index_dataclass_name is not None
        # adding a field to the dataclass for keeping track of the depth
        index_dataclass_name = make_dataclass(
            "DepthExtendedDataclass",
            fields=[],
            bases=(base_index_dataclass_name, MaxDepthDescendants),
        )
    else:
        index_dataclass_name = base_index_dataclass_name

    if index_dataclass_name is not None:
        exploration_index_data = index_dataclass_name(tree_node=tree_node)
    else:
        exploration_index_data = None

    return exploration_index_data

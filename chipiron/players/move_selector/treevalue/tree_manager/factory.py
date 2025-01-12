"""
This module provides a factory function for creating an AlgorithmNodeTreeManager object.

The AlgorithmNodeTreeManager is responsible for managing the tree structure of algorithm nodes,
performing updates on the nodes, and handling evaluation queries.

"""

import chipiron.players.move_selector.treevalue.updates as upda
from chipiron.players.move_selector.treevalue import node_factory
from chipiron.players.move_selector.treevalue.indices.index_manager import (
    NodeExplorationIndexManager,
    create_exploration_index_manager,
)
from chipiron.players.move_selector.treevalue.indices.node_indices.index_types import (
    IndexComputationType,
)
from chipiron.players.move_selector.treevalue.node_evaluator import (
    EvaluationQueries,
    NodeEvaluator,
)
from chipiron.players.move_selector.treevalue.updates.index_updater import IndexUpdater

from .algorithm_node_tree_manager import AlgorithmNodeTreeManager
from .tree_manager import TreeManager


def create_algorithm_node_tree_manager(
    node_evaluator: NodeEvaluator | None,
    algorithm_node_factory: node_factory.AlgorithmNodeFactory,
    index_computation: IndexComputationType | None,
    index_updater: IndexUpdater | None,
) -> AlgorithmNodeTreeManager:
    """
    Create an AlgorithmNodeTreeManager object.

    Args:
        node_evaluator: The NodeEvaluator object used for evaluating nodes in the tree.
        algorithm_node_factory: The AlgorithmNodeFactory object used for creating algorithm nodes.
        index_computation: The type of index computation to be used.
        index_updater: The IndexUpdater object used for updating the indices.

    Returns:
        An AlgorithmNodeTreeManager object.

    """
    tree_manager: TreeManager = TreeManager(node_factory=algorithm_node_factory)

    algorithm_node_updater: upda.AlgorithmNodeUpdater = (
        upda.create_algorithm_node_updater(index_updater=index_updater)
    )

    evaluation_queries: EvaluationQueries = EvaluationQueries()

    exploration_index_manager: NodeExplorationIndexManager = (
        create_exploration_index_manager(index_computation=index_computation)
    )

    algorithm_node_tree_manager: AlgorithmNodeTreeManager = AlgorithmNodeTreeManager(
        node_evaluator=node_evaluator,
        tree_manager=tree_manager,
        algorithm_node_updater=algorithm_node_updater,
        evaluation_queries=evaluation_queries,
        index_manager=exploration_index_manager,
    )

    return algorithm_node_tree_manager

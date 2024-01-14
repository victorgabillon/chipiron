import chipiron.players.move_selector.treevalue.nodes as nodes

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class NodeExplorationManager:
    tree_node: nodes.TreeNode
    index: float | None = None


@dataclass(slots=True)
class RecurZipfQuoolExplorationManager(NodeExplorationManager):
    # the 'proba' associated by recursively multiplying 1/rank of the node with the max zipf_factor of the parents
    zipf_factored_proba: float | None = None


def create_node_exploration_manager(
        tree_node: nodes.TreeNode,
        index_computation: Any = None
) -> NodeExplorationManager:
    #print('index_computation',index_computation)

    if index_computation is None:
        node_exploration_manager: NodeExplorationManager = NodeExplorationManager(tree_node=tree_node)
    else:
        node_exploration_manager: NodeExplorationManager = RecurZipfQuoolExplorationManager(tree_node=tree_node)

    return node_exploration_manager

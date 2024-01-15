import chipiron.players.move_selector.treevalue.nodes as nodes

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class NodeExplorationManager:
    tree_node: nodes.TreeNode
    index: float | None = None

    def dot_description(self):
        return f'index:{self.index}'


@dataclass(slots=True)
class RecurZipfQuoolExplorationManager(NodeExplorationManager):
    # the 'proba' associated by recursively multiplying 1/rank of the node with the max zipf_factor of the parents
    zipf_factored_proba: float | None = None

    def dot_description(self):
        return f'index:{self.index} zipf_factored_proba:{self.zipf_factored_proba}'


def create_node_exploration_manager(
        tree_node: nodes.TreeNode,
        index_computation: Any = None
) -> NodeExplorationManager:
    if index_computation is None:
        node_exploration_manager: NodeExplorationManager = NodeExplorationManager(tree_node=tree_node)
    else:
        node_exploration_manager: NodeExplorationManager = RecurZipfQuoolExplorationManager(tree_node=tree_node)

    return node_exploration_manager

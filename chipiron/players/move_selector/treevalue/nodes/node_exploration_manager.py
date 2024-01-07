import chipiron.players.move_selector.treevalue.nodes as nodes

from dataclasses import dataclass, field


@dataclass(slots=True)
class NodeExplorationManager:
    tree_node: nodes.TreeNode
    index: float | None = None


def create_node_exploration_manager(
        tree_node: nodes.TreeNode
) -> NodeExplorationManager:
    node_exploration_manager: NodeExplorationManager = NodeExplorationManager(tree_node=tree_node)

    return node_exploration_manager

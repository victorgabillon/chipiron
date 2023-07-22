import chipiron.players.treevalue.nodes as nodes


class NodeExplorationManager:
    tree_node: nodes.TreeNode
    index: float

    def __init__(
            self,
            tree_node: nodes.TreeNode
    ) -> None:
        self.tree_node = tree_node
        self.index = None

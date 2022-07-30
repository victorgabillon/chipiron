import chipiron as ch
import chipiron.players.treevalue.nodes as node
from chipiron.players.treevalue import node_factory as node_fact
from chipiron.players.treevalue.trees.move_and_value_tree import MoveAndValueTree


class TreeExpander:
    """
    class in charge of expanding a tree both in terms of adding moves or nodes
    """
    tree: MoveAndValueTree
    node_factory: node_fact.TreeNodeFactory

    def __init__(self, tree: MoveAndValueTree, node_factory: node_fact.TreeNodeFactory) -> None:
        self.tree = tree
        self.node_factory = node_factory

    def create_tree_node(self,
                         board: ch.chess.IBoard,
                         board_modifications,
                         half_move: int,
                         parent_node) -> node.TreeNode:
        board_depth: int = half_move - self.tree.tree_root_half_move
        new_node: node.TreeNode = self.node_factory.create(board=board,
                                                           half_move=half_move,
                                                           count=self.tree.nodes_count,
                                                           parent_node=parent_node,
                                                           board_depth=board_depth)
        self.tree.nodes_count += 1
        self.tree.board_evaluator.compute_representation(new_node, parent_node, board_modifications)
        self.tree.board_evaluator.add_evaluation_query(new_node)
        return new_node

    def create_tree_move(self,
                         board: ch.chess.IBoard,
                         half_move: int,
                         fast_rep: str,
                         parent_node: node.TreeNode) -> node.TreeNode:
        tree_node = self.tree.descendants[half_move][fast_rep]  # add it to the list of descendants
        tree_node.add_parent(parent_node)
        return tree_node

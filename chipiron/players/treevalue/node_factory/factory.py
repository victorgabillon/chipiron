from typing import Protocol
from chipiron.players.treevalue.nodes.tree_node import TreeNode
from ..nodes.tree_node_with_descendants import NodeWithDescendants


def create_root_node(board, half_move, board_evaluators_wrapper):
    root_node = NodeWithDescendants(board=board, half_move=half_move, id_number=0, parent_node=None)
    # TODO move it from here maybe maybe this should be in the node factory
    board_evaluators_wrapper.compute_representation(root_node, None, None)
    board_evaluators_wrapper.add_evaluation_query(root_node)
    board_evaluators_wrapper.evaluate_all_queried_nodes()
    return root_node


def create_root_node(board,
                     half_move: int,
                     board_evaluators_wrapper) -> TreeNode:
    root_node: TreeNode = NodeWithDescendants(board=board, half_move=half_move, id_number=0, parent_node=None)
    board_evaluators_wrapper.compute_representation(root_node, None, None)
    board_evaluators_wrapper.add_evaluation_query(root_node)
    return root_node


class TreeNodeFactory(Protocol):

    def create(self,
               board,
               half_move: int,
               count: int,
               parent_node,
               board_depth: int) -> TreeNode:
        ...

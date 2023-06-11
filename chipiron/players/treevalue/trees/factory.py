import chipiron as ch
from chipiron.players.treevalue.trees.move_and_value_tree import MoveAndValueTree
from .descendants import RangedDescendants
import chipiron.players.treevalue.node_factory as nod_fac


def create_tree(
        board: ch.chess.BoardChi,
        node_factory: nod_fac.AlgorithmNodeFactory,
        board_evaluator) -> MoveAndValueTree:

    root_node = node_factory.create(board=board, half_move=board.ply(), count=0, parent_node=None, board_depth=0)
    board_evaluator.compute_representation(root_node.tree_node,
                                           None,
                                           None)
    board_evaluator.add_evaluation_query(root_node)

    board_evaluator.evaluate_all_queried_nodes()

    descendants: RangedDescendants = RangedDescendants()

    move_and_value_tree: MoveAndValueTree = MoveAndValueTree(root_node=root_node,
                                                             descendants=descendants)

    return move_and_value_tree

import chipiron as ch
from chipiron.players.treevalue.trees.move_and_value_tree import MoveAndValueTree
from chipiron.players.treevalue import node_factory


def create_tree(
        board_evaluators_wrapper,
        board: ch.chess.BoardChi) -> MoveAndValueTree:

    root_node = node_factory.create_root_node(board, board.ply(), board_evaluators_wrapper)

    move_and_value_tree: MoveAndValueTree = MoveAndValueTree(root_node=root_node)

    return move_and_value_tree

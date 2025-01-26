from typing import Any
from unittest.mock import MagicMock

import chess

from chipiron.environments.chess.board import IBoard
from chipiron.players.move_selector.treevalue.nodes.tree_node import TreeNode


class TestTreeNode:
    def setup_method(self) -> None:
        print("Setting up the test for the TreeNode class.")
        self.mock_board = MagicMock(spec=IBoard)
        self.mock_board.legal_moves = MagicMock()
        self.mock_board.legal_moves.get_all = MagicMock(return_value=["e2e4", "d2d4"])
        self.mock_board.turn = chess.WHITE
        self.tree_node: TreeNode[Any] = TreeNode(
            id_=1,
            half_move_=0,
            board_=self.mock_board,
            parent_nodes_={},
            all_legal_moves_generated=False,
            non_opened_legal_moves=set(),
            moves_children_={},
        )

    def test_all_legal_moves_generated_true(self) -> None:
        self.tree_node.all_legal_moves_generated = True
        self.tree_node.moves_children_ = {0: None}
        self.tree_node.non_opened_legal_moves = {0}
        self.tree_node.test_all_legal_moves_generated()

    def test_all_legal_moves_generated_false(self) -> None:
        self.tree_node.all_legal_moves_generated = False
        self.tree_node.moves_children_ = {0: None}
        self.tree_node.test_all_legal_moves_generated()

    def test_all_legal_moves_generated_no_moves(self) -> None:
        self.tree_node.all_legal_moves_generated = False
        self.mock_board.legal_moves.get_all = MagicMock(return_value=[])
        self.tree_node.test_all_legal_moves_generated()


if __name__ == "__main__":
    test = TestTreeNode()
    test.setup_method()
    test.test_all_legal_moves_generated_true()
    test.test_all_legal_moves_generated_false()
    test.test_all_legal_moves_generated_no_moves()
    print("All tests passed!")

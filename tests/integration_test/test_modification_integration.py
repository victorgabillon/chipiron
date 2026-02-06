"""Module for test modification integration."""

import random
from typing import TYPE_CHECKING

import chess
import pytest
from anemone.nodes.algorithm_node import AlgorithmNode
from anemone.tree_and_value_branch_selector import (
    TreeAndValueBranchSelector,
)
from atomheart.board import IBoard, create_board
from atomheart.board.utils import FenPlusHistory

from chipiron.environments.chess.types import ChessState
from chipiron.players.factory import create_chipiron_player
from chipiron.scripts.chipiron_args import ImplementationArgs

if TYPE_CHECKING:
    import anemone.trees as trees
    import atomheart.board as boards
    from anemone.basics import TreeDepth
    from anemone.nodes import ITreeNode
    from anemone.tree_exploration import (
        TreeExploration,
    )

    from chipiron.players import Player


@pytest.mark.parametrize(("use_rust_boards"), (True, False))
def test_modifications(
    use_rust_boards: bool,
) -> None:
    """Test modifications."""
    board_one: IBoard = create_board(
        use_rust_boards=use_rust_boards,
        use_board_modification=False,
        sort_legal_moves=True,
        fen_with_history=FenPlusHistory(current_fen=chess.STARTING_FEN),
    )

    random_generator_one: random.Random = random.Random(0)
    player_one: Player[FenPlusHistory, ChessState] = create_chipiron_player(
        implementation_args=ImplementationArgs(
            use_rust_boards=use_rust_boards, use_board_modification=False
        ),
        random_generator=random_generator_one,
        universal_behavior=True,
        tree_branch_limit=1000,
    )

    main_move_selector_one = player_one.adapter.main_move_selector
    assert isinstance(main_move_selector_one, TreeAndValueBranchSelector)

    tree_exploration_one: TreeExploration = (
        main_move_selector_one.create_tree_exploration(
            state=ChessState(board=board_one)
        )
    )
    tree_one: trees.Tree = tree_exploration_one.explore(
        random_generator=random_generator_one
    ).tree

    board_two: IBoard = create_board(
        use_rust_boards=use_rust_boards,
        use_board_modification=True,
        sort_legal_moves=True,
        fen_with_history=FenPlusHistory(current_fen=chess.STARTING_FEN),
    )

    random_generator_two: random.Random = random.Random(0)

    player_two: Player = create_chipiron_player(
        implementation_args=ImplementationArgs(
            use_rust_boards=use_rust_boards, use_board_modification=True
        ),
        universal_behavior=True,
        random_generator=random_generator_two,
        tree_branch_limit=1000,
    )

    main_move_selector_two = player_two.adapter.main_move_selector
    assert isinstance(main_move_selector_two, TreeAndValueBranchSelector)

    tree_exploration_two: TreeExploration = (
        main_move_selector_two.create_tree_exploration(
            state=ChessState(board=board_two)
        )
    )
    tree_two: trees.MoveAndValueTree = tree_exploration_two.explore(
        random_generator=random_generator_two
    ).tree

    node_one: AlgorithmNode
    tree_depth: TreeDepth
    board_key: boards.boardKey
    for tree_depth, board_key, node_one in tree_one.descendants.iter_on_all_nodes():
        node_two: ITreeNode = tree_two.descendants.descendants_at_tree_depth[
            tree_depth
        ][board_key]
        assert isinstance(node_one, AlgorithmNode)
        assert isinstance(node_two, AlgorithmNode)

        assert node_one.state_representation == node_two.state_representation


if __name__ == "__main__":
    use_rusty_board: bool
    for use_rusty_board in [True, False]:
        test_modifications(
            use_rust_boards=use_rusty_board,
        )
    print("all tests passed")

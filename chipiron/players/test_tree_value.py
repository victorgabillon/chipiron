import random
from typing import Any

import chess
import pytest

import chipiron.environments.chess.board as boards
import chipiron.players.move_selector.treevalue.trees as trees
from chipiron.environments import HalfMove
from chipiron.environments.chess.board import IBoard, create_board
from chipiron.environments.chess.board.utils import FenPlusHistory
from chipiron.players import Player
from chipiron.players.factory import create_chipiron_player
from chipiron.players.move_selector.treevalue.nodes import ITreeNode
from chipiron.players.move_selector.treevalue.nodes.algorithm_node import AlgorithmNode
from chipiron.players.move_selector.treevalue.tree_and_value_move_selector import (
    TreeAndValueMoveSelector,
)
from chipiron.players.move_selector.treevalue.tree_exploration import TreeExploration


@pytest.mark.parametrize(
    ('use_rust_boards'),
    (True, False)
)
def test_random(
        use_rust_boards: bool
) -> None:
    board_one: IBoard = create_board(
        use_rust_boards=use_rust_boards,
        use_board_modification=False,
        sort_legal_moves=True,
        fen_with_history=FenPlusHistory(
            current_fen=chess.STARTING_FEN)
    )

    random_generator_one: random.Random = random.Random(0)

    player_one: Player = create_chipiron_player(
        depth=1,
        use_rusty_board=use_rust_boards,
        random_generator=random_generator_one

    )

    main_move_selector_one = player_one.main_move_selector
    assert isinstance(main_move_selector_one, TreeAndValueMoveSelector)

    tree_exploration_one: TreeExploration = main_move_selector_one.create_tree_exploration(board=board_one)
    tree_one: trees.MoveAndValueTree = tree_exploration_one.explore(
        random_generator=random_generator_one
    ).tree

    board_two: IBoard = create_board(
        use_rust_boards=use_rust_boards,
        use_board_modification=True,
        sort_legal_moves=True,
        fen_with_history=FenPlusHistory(
            current_fen=chess.STARTING_FEN)
    )

    random_generator_two: random.Random = random.Random(0)
    player_two: Player = create_chipiron_player(
        depth=1,
        use_rusty_board=use_rust_boards,
        random_generator=random_generator_two
    )

    main_move_selector_two = player_two.main_move_selector
    assert isinstance(main_move_selector_two, TreeAndValueMoveSelector)

    tree_exploration_two: TreeExploration = main_move_selector_two.create_tree_exploration(board=board_two)
    tree_two: trees.MoveAndValueTree = tree_exploration_two.explore(
        random_generator=random_generator_two
    ).tree

    node_one: ITreeNode[Any]
    half_move: HalfMove
    board_key: boards.boardKey
    for half_move, board_key, node_one in tree_one.descendants.iter_on_all_nodes():
        node_two: ITreeNode[Any] = tree_two.descendants.descendants_at_half_move[half_move][board_key]
        assert (isinstance(node_one, AlgorithmNode))
        assert (isinstance(node_two, AlgorithmNode))

        assert (node_one.id == node_two.id)

    node_two_: ITreeNode[Any]
    half_move_: HalfMove
    board_key_: boards.boardKey
    for half_move_, board_key_, node_two_ in tree_two.descendants.iter_on_all_nodes():
        node_one_: ITreeNode[Any] = tree_one.descendants.descendants_at_half_move[half_move_][board_key_]
        assert (isinstance(node_one_, AlgorithmNode))
        assert (isinstance(node_two_, AlgorithmNode))

        assert (node_one_.id == node_two_.id)  # a test for universal behavior that is probably not in the right place!


if __name__ == '__main__':
    use_rusty_board: bool
    for use_rusty_board in [True, False]:
        test_random(
            use_rust_boards=use_rusty_board,
        )
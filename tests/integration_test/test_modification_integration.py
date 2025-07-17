import random
from typing import Any

import chess
import pytest

import chipiron.environments.chess_env.board as boards
import chipiron.players.move_selector.treevalue.trees as trees
from chipiron.environments import HalfMove
from chipiron.environments.chess_env.board import IBoard, create_board
from chipiron.environments.chess_env.board.utils import FenPlusHistory
from chipiron.players import Player
from chipiron.players.factory import create_chipiron_player
from chipiron.players.move_selector.treevalue.nodes import ITreeNode
from chipiron.players.move_selector.treevalue.nodes.algorithm_node import AlgorithmNode
from chipiron.players.move_selector.treevalue.tree_and_value_move_selector import (
    TreeAndValueMoveSelector,
)
from chipiron.players.move_selector.treevalue.tree_exploration import TreeExploration
from chipiron.scripts.chipiron_args import ImplementationArgs


@pytest.mark.parametrize(("use_rust_boards"), (True, False))
def test_modifications(
    use_rust_boards: bool,
) -> None:
    board_one: IBoard = create_board(
        use_rust_boards=use_rust_boards,
        use_board_modification=False,
        sort_legal_moves=True,
        fen_with_history=FenPlusHistory(current_fen=chess.STARTING_FEN),
    )

    random_generator_one: random.Random = random.Random(0)
    player_one: Player = create_chipiron_player(
        implementation_args=ImplementationArgs(
            use_rust_boards=use_rust_boards, use_board_modification=False
        ),
        random_generator=random_generator_one,
        universal_behavior=True,
    )

    main_move_selector_one = player_one.main_move_selector
    assert isinstance(main_move_selector_one, TreeAndValueMoveSelector)

    tree_exploration_one: TreeExploration = (
        main_move_selector_one.create_tree_exploration(board=board_one)
    )
    tree_one: trees.MoveAndValueTree = tree_exploration_one.explore(
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
    )

    main_move_selector_two = player_two.main_move_selector
    assert isinstance(main_move_selector_two, TreeAndValueMoveSelector)

    tree_exploration_two: TreeExploration = (
        main_move_selector_two.create_tree_exploration(board=board_two)
    )
    tree_two: trees.MoveAndValueTree = tree_exploration_two.explore(
        random_generator=random_generator_two
    ).tree

    node_one: AlgorithmNode
    half_move: HalfMove
    board_key: boards.boardKey
    for half_move, board_key, node_one in tree_one.descendants.iter_on_all_nodes():
        node_two: ITreeNode[Any] = tree_two.descendants.descendants_at_half_move[
            half_move
        ][board_key]
        assert isinstance(node_one, AlgorithmNode)
        assert isinstance(node_two, AlgorithmNode)

        assert node_one.board_representation == node_two.board_representation


if __name__ == "__main__":

    use_rusty_board: bool
    for use_rusty_board in [True, False]:
        test_modifications(
            use_rust_boards=use_rusty_board,
        )

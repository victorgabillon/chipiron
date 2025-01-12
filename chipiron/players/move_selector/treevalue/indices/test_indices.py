"""
This module contains functions for testing the indices used in the move selector tree.

The main functions in this module are:
- `make_tree_from_file`: Creates a move and value tree from a YAML file.
- `check_from_file`: Compares the indices computed from the tree with the expected indices from a YAML file.
- `check_index`: Tests the indices for a specific index computation type and tree file.
- `test_indices`: Runs the index tests for multiple index computation types and tree files.
"""

from enum import Enum
from math import isclose
from typing import Any

import chess
import yaml

import chipiron.players.move_selector.treevalue.node_factory as node_factory
import chipiron.players.move_selector.treevalue.search_factory as search_factories
import chipiron.players.move_selector.treevalue.tree_manager as tree_manager
import chipiron.players.move_selector.treevalue.trees as trees
from chipiron.environments.chess.board.factory import (
    create_board_chi_from_pychess_board,
)
from chipiron.players.move_selector.treevalue.indices.index_manager.factory import (
    create_exploration_index_manager,
)
from chipiron.players.move_selector.treevalue.indices.index_manager.node_exploration_manager import (
    NodeExplorationIndexManager,
    update_all_indices,
)
from chipiron.players.move_selector.treevalue.indices.node_indices.index_types import (
    IndexComputationType,
)
from chipiron.players.move_selector.treevalue.nodes.algorithm_node.algorithm_node import (
    AlgorithmNode,
)
from chipiron.players.move_selector.treevalue.nodes.itree_node import ITreeNode
from chipiron.players.move_selector.treevalue.tree_manager.tree_expander import (
    TreeExpansion,
    TreeExpansions,
)
from chipiron.players.move_selector.treevalue.trees.descendants import RangedDescendants
from chipiron.players.move_selector.treevalue.trees.move_and_value_tree import (
    MoveAndValueTree,
)
from chipiron.utils.small_tools import path


class TestResult(Enum):
    """
    Enumeration for the test results.
    """

    __test__ = False
    PASSED = 0
    FAILED = 1
    WARNING = 2


def make_tree_from_file(
    file_path: path, index_computation: IndexComputationType
) -> MoveAndValueTree:
    """
    Creates a move and value tree from a file.

    Args:
        file_path (path): The path to the file containing the tree data.
        index_computation (IndexComputationType): The type of index computation to use.

    Returns:
        MoveAndValueTree: The created move and value tree.
    """

    # atm it is very ad hoc to test index so takes a lots of shortcut, will be made more general when needed
    with open(file_path, "r") as file:
        tree_yaml = yaml.safe_load(file)
    print("tree", tree_yaml)
    yaml_nodes = tree_yaml["nodes"]

    node_factory_name: str = "Base_with_algorithm_tree_node"
    tree_node_factory: node_factory.Base[Any] = node_factory.create_node_factory(
        node_factory_name=node_factory_name
    )

    search_factory: search_factories.SearchFactoryP = search_factories.SearchFactory(
        node_selector_args=None,
        opening_type=None,
        random_generator=None,
        index_computation=index_computation,
    )

    algorithm_node_factory: node_factory.AlgorithmNodeFactory
    algorithm_node_factory = node_factory.AlgorithmNodeFactory(
        tree_node_factory=tree_node_factory,
        board_representation_factory=None,
        exploration_index_data_create=search_factory.node_index_create,
    )
    descendants: RangedDescendants = RangedDescendants()

    algo_tree_manager: tree_manager.AlgorithmNodeTreeManager = (
        tree_manager.create_algorithm_node_tree_manager(
            node_evaluator=None,
            algorithm_node_factory=algorithm_node_factory,
            index_computation=index_computation,
            index_updater=None,
        )
    )

    half_moves = {}
    id_nodes = {}
    for yaml_node in yaml_nodes:
        # print('yaml_node[id] ',yaml_node['id'] )
        if yaml_node["id"] == 0:
            tree_expansions = TreeExpansions()

            board = chess.Board.from_chess960_pos(yaml_node["id"])
            board.turn = chess.WHITE
            board_chi = create_board_chi_from_pychess_board(chess_board=board)

            root_node: ITreeNode[Any] = algorithm_node_factory.create(
                board=board_chi,
                half_move=0,
                count=yaml_node["id"],
                parent_node=None,
                move_from_parent=None,
                modifications=None,
            )
            assert isinstance(root_node, AlgorithmNode)
            root_node.minmax_evaluation.value_white_minmax = yaml_node["value"]
            half_moves[yaml_node["id"]] = 0
            id_nodes[yaml_node["id"]] = root_node

            descendants.add_descendant(root_node)
            move_and_value_tree: MoveAndValueTree = MoveAndValueTree(
                root_node=root_node, descendants=descendants
            )
            tree_expansions.add(
                TreeExpansion(
                    child_node=root_node,
                    parent_node=None,
                    board_modifications=None,
                    creation_child_node=True,
                    move=None,
                )
            )
            root_node.tree_node.all_legal_moves_generated = True
            # algo_tree_manager.update_backward(tree_expansions=tree_expansions)

        else:
            tree_expansions = TreeExpansions()
            first_parent = yaml_node["parents"]
            half_move = half_moves[first_parent] + 1
            half_moves[yaml_node["id"]] = half_move
            parent_node = id_nodes[first_parent]
            board = chess.Board.from_chess960_pos(yaml_node["id"])
            board.turn = not parent_node.tree_node.board_.turn
            board_chi = create_board_chi_from_pychess_board(chess_board=board)

            tree_expansion: TreeExpansion = algo_tree_manager.tree_manager.open_node(
                tree=move_and_value_tree,
                parent_node=parent_node,
                board=board_chi,
                modifications=None,
                move=yaml_node["id"],
            )
            tree_expansions.add(
                TreeExpansion(
                    child_node=tree_expansion.child_node,
                    parent_node=tree_expansion.parent_node,
                    board_modifications=tree_expansion.board_modifications,
                    creation_child_node=tree_expansion.creation_child_node,
                    move=yaml_node["id"],
                )
            )
            assert isinstance(tree_expansion.child_node, AlgorithmNode)
            tree_expansion.child_node.tree_node.all_legal_moves_generated = True
            id_nodes[yaml_node["id"]] = tree_expansion.child_node
            tree_expansion.child_node.minmax_evaluation.value_white_minmax = yaml_node[
                "value"
            ]
            tree_expansion.child_node.minmax_evaluation.value_white_evaluator = (
                yaml_node["value"]
            )
            assert tree_expansion.move is not None
            parent_node.minmax_evaluation.moves_not_over.append(tree_expansion.move)
            algo_tree_manager.update_backward(tree_expansions=tree_expansions)

    # print('move_and_value_tree', move_and_value_tree.descendants)
    return move_and_value_tree


def check_from_file(file_path: path, tree: MoveAndValueTree) -> None:
    """
    Check the values in the given file against the values in the tree.

    Args:
        file_path (str): The path to the file containing the values to check.
        tree (MoveAndValueTree): The tree containing the values to compare against.

    Returns:
        None
    """
    with open(file_path, "r") as file:
        tree_yaml = yaml.safe_load(file)
    print("tree", tree_yaml)
    yaml_nodes = tree_yaml["nodes"]

    tree_nodes: trees.RangedDescendants = tree.descendants

    half_move: int
    for half_move in tree_nodes:
        # todo how are we sure that the hm comes in order?
        # print('hmv', half_move)
        parent_node: ITreeNode[Any]
        for parent_node in tree_nodes[half_move].values():
            assert isinstance(parent_node, AlgorithmNode)
            yaml_index = eval(str(yaml_nodes[parent_node.id]["index"]))
            assert parent_node.exploration_index_data is not None
            print(
                f"id {parent_node.id} expected value {yaml_index} "
                f"|| computed value {parent_node.exploration_index_data.index}"
                f' {type(yaml_nodes[parent_node.id]["index"])}'
                f" {type(parent_node.exploration_index_data.index)}"
                f"{(yaml_index == parent_node.exploration_index_data.index)}"
            )
            if yaml_index is None:
                assert parent_node.exploration_index_data.index is None
            else:
                assert parent_node.exploration_index_data.index is not None
                assert isclose(
                    yaml_index, parent_node.exploration_index_data.index, abs_tol=1e-8
                )


def check_index(index_computation: IndexComputationType, tree_file: path) -> TestResult:
    """
    Checks the index for a given tree file and index computation type.

    Args:
        index_computation (IndexComputationType): The type of index computation.
        tree_file (path): The path to the tree file.

    Returns:
        TestResult: The result of the index check.

    Raises:
        None

    """
    try:
        tree_path = f"data/trees/{tree_file}/{tree_file}_{index_computation.value}.yaml"
    except Exception:
        print(f"!!!!!!Warning!!!!! : no testing file {tree_path}")
        return TestResult.WARNING

    tree_path = f"data/trees/{tree_file}/{tree_file}.yaml"
    tree: MoveAndValueTree = make_tree_from_file(
        index_computation=index_computation, file_path=tree_path
    )

    index_manager: NodeExplorationIndexManager = create_exploration_index_manager(
        index_computation=index_computation
    )

    print("index_manager", index_manager)
    update_all_indices(tree, index_manager)
    # print_all_indices(
    #    tree
    # )
    file_index = f"data/trees/{tree_file}/{tree_file}_{index_computation.value}.yaml"
    check_from_file(file_path=file_index, tree=tree)

    return TestResult.PASSED


def test_indices() -> None:
    """
    Test the index computations on multiple tree files.

    This function iterates over a list of index computations and tree files,
    and performs a test for each combination. The results of the tests are
    stored in a dictionary.

    Returns:
        None
    """
    index_computations: list[IndexComputationType] = [
        IndexComputationType.MinGlobalChange,
        IndexComputationType.RecurZipf,
        IndexComputationType.MinLocalChange,
    ]

    tree_files = ["tree_1", "tree_2"]

    results: dict[TestResult, int] = {}
    for tree_file in tree_files:

        if tree_file == "tree_2":
            index_computations_ = [
                IndexComputationType.MinGlobalChange,
                IndexComputationType.MinLocalChange,
            ]
        else:
            index_computations_ = index_computations

        for index_computation in index_computations_:
            print(f"---testing {index_computation} on {tree_file}")
            res: TestResult = check_index(
                index_computation=index_computation, tree_file=tree_file
            )
            if res in results:
                results[res] += 1
            else:
                results[res] = 1
    print(f"finished Test: {results}")
    assert results[TestResult.PASSED] == 5


if __name__ == "__main__":
    test_indices()

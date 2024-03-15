from enum import Enum
from math import isclose

import chess
import yaml
from chipiron.players.move_selector.treevalue.node_indices.factory import create_exploration_index_manager
from chipiron.players.move_selector.treevalue.node_indices.index_types import IndexComputationType
from chipiron.players.move_selector.treevalue.node_indices.node_exploration_manager import update_all_indices, \
    NodeExplorationIndexManager

import chipiron.environments.chess.board as boards
import chipiron.players.move_selector.treevalue.node_factory as node_factory
import chipiron.players.move_selector.treevalue.nodes as nodes
import chipiron.players.move_selector.treevalue.search_factory as search_factories
import chipiron.players.move_selector.treevalue.tree_manager as tree_manager
import chipiron.players.move_selector.treevalue.trees as trees
from chipiron.environments.chess.board.board import BoardChi
from chipiron.players.move_selector.treevalue.tree_manager.tree_expander import TreeExpansions, TreeExpansion
from chipiron.players.move_selector.treevalue.trees.descendants import RangedDescendants
from chipiron.players.move_selector.treevalue.trees.move_and_value_tree import MoveAndValueTree
from chipiron.utils.small_tools import path


class TestResult(Enum):
    PASSED = 0
    FAILED = 1
    WARNING = 2


def make_tree_from_file(
        file_path: path,
        index_computation
) -> MoveAndValueTree:
    # atm it is very ad hoc to test index so takes a lots of shortcut, will be made more general when needed
    with open(file_path, 'r') as file:
        tree_yaml = yaml.safe_load(file)
    print('tree', tree_yaml)
    yaml_nodes = tree_yaml['nodes']

    node_factory_name: str = 'Base'
    tree_node_factory: node_factory.Base = node_factory.create_node_factory(
        node_factory_name=node_factory_name
    )

    search_factory: search_factories.SearchFactoryP = search_factories.SearchFactory(
        node_selector_args=None,
        opening_type=None,
        random_generator=None,
        index_computation=index_computation
    )

    algorithm_node_factory: node_factory.AlgorithmNodeFactory
    algorithm_node_factory = node_factory.AlgorithmNodeFactory(
        tree_node_factory=tree_node_factory,
        board_representation_factory=None,
        exploration_index_data_create=search_factory.node_index_create
    )
    descendants: RangedDescendants = RangedDescendants()

    algo_tree_manager: tree_manager.AlgorithmNodeTreeManager = tree_manager.create_algorithm_node_tree_manager(
        node_evaluator=None,
        algorithm_node_factory=algorithm_node_factory,
        index_computation=index_computation,
        index_updater=None
    )

    half_moves = {}
    id_nodes = {}
    for yaml_node in yaml_nodes:
        # print('yaml_node[id] ',yaml_node['id'] )
        if yaml_node['id'] == 0:
            tree_expansions = TreeExpansions()

            board = chess.Board.from_chess960_pos(yaml_node['id'])
            board.turn = chess.WHITE
            board_chi = BoardChi(board=board)
            root_node: nodes.AlgorithmNode = algorithm_node_factory.create(
                board=board_chi,
                half_move=0,
                count=yaml_node['id'],
                parent_node=None,
                modifications=None
            )
            root_node.minmax_evaluation.value_white_minmax = yaml_node['value']
            half_moves[yaml_node['id']] = 0
            id_nodes[yaml_node['id']] = root_node

            descendants.add_descendant(root_node)
            move_and_value_tree: MoveAndValueTree = MoveAndValueTree(
                root_node=root_node,
                descendants=descendants
            )
            tree_expansions.add(
                TreeExpansion(
                    child_node=root_node,
                    parent_node=None,
                    board_modifications=None,
                    creation_child_node=True
                )
            )
            root_node.tree_node.all_legal_moves_generated = True
            # algo_tree_manager.update_backward(tree_expansions=tree_expansions)

        else:
            tree_expansions = TreeExpansions()
            first_parent = yaml_node['parents']
            half_move = half_moves[first_parent] + 1
            half_moves[yaml_node['id']] = half_move
            parent_node = id_nodes[first_parent]
            board = chess.Board.from_chess960_pos(yaml_node['id'])
            board.turn = not parent_node.tree_node.board_.turn
            board_chi = boards.BoardChi(board=board)

            tree_expansion: TreeExpansion = algo_tree_manager.tree_manager.open_node(
                tree=move_and_value_tree,
                parent_node=parent_node,
                board=board_chi,
                modifications=None,
                move=yaml_node['id']
            )
            tree_expansions.add(
                TreeExpansion(
                    child_node=tree_expansion.child_node,
                    parent_node=tree_expansion.parent_node,
                    board_modifications=tree_expansion.board_modifications,
                    creation_child_node=tree_expansion.creation_child_node
                )
            )
            assert isinstance(tree_expansion.child_node, nodes.AlgorithmNode)
            tree_expansion.child_node.tree_node.all_legal_moves_generated = True
            id_nodes[yaml_node['id']] = tree_expansion.child_node
            tree_expansion.child_node.minmax_evaluation.value_white_minmax = yaml_node['value']
            tree_expansion.child_node.minmax_evaluation.value_white_evaluator = yaml_node['value']
            parent_node.minmax_evaluation.children_not_over.append(tree_expansion.child_node)
            algo_tree_manager.update_backward(tree_expansions=tree_expansions)

    # print('move_and_value_tree', move_and_value_tree.descendants)
    return move_and_value_tree


def check_from_file(file, tree):
    with open(file, 'r') as file:
        tree_yaml = yaml.safe_load(file)
    print('tree', tree_yaml)
    yaml_nodes = tree_yaml['nodes']

    tree_nodes: trees.RangedDescendants = tree.descendants

    half_move: int
    for half_move in tree_nodes:
        # todo how are we sure that the hm comes in order?
        # print('hmv', half_move)
        parent_node: nodes.AlgorithmNode
        for parent_node in tree_nodes[half_move].values():
            yaml_index = eval(str(yaml_nodes[parent_node.id]['index']))
            assert parent_node.exploration_index_data is not None
            print(
                f'id {parent_node.id} expected value {yaml_index} '
                f'|| computed value {parent_node.exploration_index_data.index}'
                f' {type(yaml_nodes[parent_node.id]["index"])}'
                f' {type(parent_node.exploration_index_data.index)}'
                f'{(yaml_index == parent_node.exploration_index_data.index)}')
            if yaml_index is None:
                assert (parent_node.exploration_index_data.index is None)
            else:
                assert parent_node.exploration_index_data.index is not None
                assert isclose(yaml_index, parent_node.exploration_index_data.index,
                               abs_tol=1e-8)


def check_index(
        index_computation,
        tree_file
) -> TestResult:
    try:
        tree_path = f'data/trees/{tree_file}/{tree_file}_{index_computation.value}.yaml'
    except Exception:
        print(f'!!!!!!Warning!!!!! : no testing file {tree_path}')
        return TestResult.WARNING

    tree_path = f'data/trees/{tree_file}/{tree_file}.yaml'
    tree = make_tree_from_file(
        index_computation=index_computation,
        file_path=tree_path
    )

    index_manager: NodeExplorationIndexManager = create_exploration_index_manager(index_computation=index_computation)

    print('index_manager', index_manager)
    update_all_indices(
        tree,
        index_manager
    )
    # print_all_indices(
    #    tree
    # )
    file_index = f'data/trees/{tree_file}/{tree_file}_{index_computation.value}.yaml'
    check_from_file(file=file_index, tree=tree)

    return TestResult.PASSED


def test_indices():
    index_computations = [IndexComputationType.MinGlobalChange,
                          IndexComputationType.RecurZipf,
                          IndexComputationType.MinLocalChange]

    tree_files = ['tree_1', 'tree_2']

    results: dict[TestResult, int] = {}
    for index_computation in index_computations:
        for tree_file in tree_files:
            print(f'---testing {index_computation} on {tree_file}')
            res: TestResult = check_index(index_computation, tree_file)
            if res in results:
                results[res] += 1
            else:
                results[res] = 1

    print(f'finished TEst: {results}')


if __name__ == '__main__':
    test_indices()

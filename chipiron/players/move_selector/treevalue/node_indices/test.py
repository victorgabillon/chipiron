from chipiron.players.move_selector.treevalue.node_indices.node_exploration_manager import update_all_indices, \
    NodeExplorationIndexManager, print_all_indices
from chipiron.players.move_selector.treevalue.node_indices.factory import create_exploration_index_manager
from chipiron.players.move_selector.treevalue.node_indices.index_types import IndexComputationType
import chipiron.players.move_selector.treevalue.trees as trees
import chipiron.players.move_selector.treevalue.nodes as nodes
from math import isclose
from chipiron.utils.small_tools import path
import yaml
import chipiron.players.move_selector.treevalue.node_factory as node_factory
from chipiron.players.move_selector.treevalue.trees.move_and_value_tree import MoveAndValueTree
from chipiron.utils.my_value_sorted_dict import sort_dic
from chipiron.environments.chess.board import BoardChi
from chipiron.players.move_selector.treevalue.trees.descendants import RangedDescendants
import chess


def make_tree_from_file(
        file: path,
        index_computation
) -> MoveAndValueTree:
    # atm it is very ad hoc to test index so takes a lots of short cut, will be made more general when needed
    with open(file, 'r') as file:
        tree_yaml = yaml.safe_load(file)
    print('tree', tree_yaml)
    yaml_nodes = tree_yaml['nodes']

    node_factory_name: str = 'Base'
    tree_node_factory: node_factory.Base = node_factory.create_node_factory(
        node_factory_name=node_factory_name
    )

    algorithm_node_factory: node_factory.AlgorithmNodeFactory
    algorithm_node_factory = node_factory.AlgorithmNodeFactory(
        tree_node_factory=tree_node_factory,
        board_representation_factory=None,
        index_computation=index_computation
    )
    descendants: RangedDescendants = RangedDescendants()
    half_moves = {}
    id_nodes = {}
    for yaml_node in yaml_nodes:
        if yaml_node['id'] == 0:
            board = BoardChi.from_chess960_pos(yaml_node['id'])
            board.turn = chess.WHITE
            root_node: nodes.AlgorithmNode = algorithm_node_factory.create(
                board=board,
                half_move=0,
                count=yaml_node['id'],
                parent_node=None,
                board_depth=0,
                modifications=None
            )
            root_node.minmax_evaluation.value_white_minmax = yaml_node['value']
            half_moves[yaml_node['id']] = 0
            id_nodes[yaml_node['id']] = root_node

            descendants.add_descendant(root_node)
        else:
            first_parent = yaml_node['parents']
            half_move = half_moves[first_parent] + 1
            half_moves[yaml_node['id']] = half_move
            parent_node = id_nodes[first_parent]
            board = BoardChi.from_chess960_pos(yaml_node['id'])
            board.turn = not parent_node.tree_node.board_.turn
            node: nodes.AlgorithmNode = algorithm_node_factory.create(
                board=board,
                half_move=half_move,
                count=yaml_node['id'],
                parent_node=id_nodes[first_parent],
                board_depth=0,
                modifications=None
            )
            id_nodes[yaml_node['id']] = node
            node.minmax_evaluation.value_white_minmax = yaml_node['value']
            parent_node.minmax_evaluation.children_sorted_by_value_[node] = ((-1) ** half_move) * yaml_node['value']
            parent_node.minmax_evaluation.children_sorted_by_value_ = sort_dic(
                parent_node.minmax_evaluation.children_sorted_by_value_)
            descendants.add_descendant(node)

    move_and_value_tree: MoveAndValueTree = MoveAndValueTree(root_node=root_node,
                                                             descendants=descendants)

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
            print(
                f'id {parent_node.id} expected value {yaml_index} || computed value {parent_node.exploration_index_data.index} {type(yaml_nodes[parent_node.id]["index"])}'
                f' {type(parent_node.exploration_index_data.index)}'
                f'{(yaml_index == parent_node.exploration_index_data.index)}')
            if yaml_index is None:
                assert (parent_node.exploration_index_data.index is None)
            else:
                assert isclose(yaml_index, parent_node.exploration_index_data.index,
                               abs_tol=1e-8)


def check_index(index_computation):
    tree = make_tree_from_file(
        index_computation=index_computation,
        file='data/trees/tree_1.yaml'
    )

    index_manager: NodeExplorationIndexManager = create_exploration_index_manager(index_computation=index_computation)

    update_all_indices(
        tree,
        index_manager
    )
    print_all_indices(
        tree
    )
    file = f'data/trees/tree_1_{index_computation.value}.yaml'
    check_from_file(file=file, tree=tree)


index_computations = [IndexComputationType.MinGlobalChange,
                      IndexComputationType.RecurZipf,
                      IndexComputationType.MinLocalChange]

for index_computation in index_computations:
    print(f'---testing {index_computation}')
    check_index(index_computation)

    print('end ALL OK')

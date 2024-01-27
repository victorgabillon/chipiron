from chipiron.players.move_selector.treevalue.node_indices.node_exploration_manager import update_all_indices, \
    NodeExplorationIndexManager, print_all_indices
from chipiron.players.move_selector.treevalue.node_indices.factory import create_exploration_index_manager
from chipiron.players.move_selector.treevalue.trees.factory import make_tree_from_file
from chipiron.players.move_selector.treevalue.node_indices.index_types import IndexComputationType
import yaml
import chipiron.players.move_selector.treevalue.trees as trees
import chipiron.players.move_selector.treevalue.nodes as nodes
from math import isclose


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
            print('tr', parent_node.id, float(yaml_nodes[parent_node.id]['index']),
                  parent_node.exploration_index_data.index,
                  type(yaml_nodes[parent_node.id]['index']), type(parent_node.exploration_index_data.index),
                  (yaml_nodes[parent_node.id]['index'] == parent_node.exploration_index_data.index))
            assert isclose(float(yaml_nodes[parent_node.id]['index']), parent_node.exploration_index_data.index,
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
                      IndexComputationType.RecurZipf]

for index_computation in index_computations:
    check_index(index_computation)

    print('end ALL OK')

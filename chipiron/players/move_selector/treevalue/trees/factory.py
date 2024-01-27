"""
MoveAndValueTreeFactory
"""
import chess

import chipiron.environments.chess.board as boards
from chipiron.players.move_selector.treevalue.trees.move_and_value_tree import MoveAndValueTree
import chipiron.players.move_selector.treevalue.node_factory as nod_fac
from chipiron.players.move_selector.treevalue.node_evaluator import NodeEvaluator, EvaluationQueries
import chipiron.players.move_selector.treevalue.nodes as nodes
from .descendants import RangedDescendants
from chipiron.utils.small_tools import path
import yaml
import chipiron.players.move_selector.treevalue.node_factory as node_factory
from .move_and_value_tree import MoveAndValueTree
from chipiron.utils.my_value_sorted_dict import sort_dic
from chipiron.environments.chess.board import BoardChi

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
            root_node: nodes.AlgorithmNode = algorithm_node_factory.create(
                board=BoardChi.from_chess960_pos(yaml_node['id']),
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
            node: nodes.AlgorithmNode = algorithm_node_factory.create(
                board=BoardChi.from_chess960_pos(yaml_node['id']),
                half_move=half_move,
                count=yaml_node['id'],
                parent_node=id_nodes[first_parent],
                board_depth=0,
                modifications=None
            )
            id_nodes[yaml_node['id']] = node
            node.minmax_evaluation.value_white_minmax = yaml_node['value']
            parent_node.minmax_evaluation.children_sorted_by_value_[node] = ((-1)**half_move)*yaml_node['value']
            parent_node.minmax_evaluation.children_sorted_by_value_ = sort_dic(parent_node.minmax_evaluation.children_sorted_by_value_)
            descendants.add_descendant(node)

    move_and_value_tree: MoveAndValueTree = MoveAndValueTree(root_node=root_node,
                                                             descendants=descendants)

    return move_and_value_tree


class MoveAndValueTreeFactory:
    """
    MoveAndValueTreeFactory
    """
    node_factory: nod_fac.AlgorithmNodeFactory
    node_evaluator: NodeEvaluator

    def __init__(
            self,
            node_factory: nod_fac.AlgorithmNodeFactory,
            node_evaluator: NodeEvaluator
    ) -> None:
        """
        creates the tree factory
        Args:
            node_factory:
            node_evaluator:
        """
        self.node_factory = node_factory
        self.node_evaluator = node_evaluator

    def create(
            self,
            starting_board: boards.BoardChi
    ) -> MoveAndValueTree:
        """
        creates the tree
        Args:
            starting_board: the starting position

        Returns:

        """
        print()
        root_node: nodes.AlgorithmNode = self.node_factory.create(
            board=starting_board,
            half_move=starting_board.ply(),
            count=0,
            parent_node=None,
            board_depth=0,
            modifications=None
        )

        evaluation_queries: EvaluationQueries = EvaluationQueries()

        self.node_evaluator.add_evaluation_query(
            node=root_node,
            evaluation_queries=evaluation_queries
        )

        self.node_evaluator.evaluate_all_queried_nodes(evaluation_queries=evaluation_queries)
        # todo this use of the node evaluator looks weird no?

        descendants: RangedDescendants = RangedDescendants()
        descendants.add_descendant(root_node)

        move_and_value_tree: MoveAndValueTree = MoveAndValueTree(root_node=root_node,
                                                                 descendants=descendants)

        return move_and_value_tree

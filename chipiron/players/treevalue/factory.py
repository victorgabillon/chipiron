import random

from .tree_and_value_player import TreeAndValuePlayer
from . import tree_manager as tree_man
from chipiron.players.treevalue import node_factory

from chipiron.players import MoveSelector
from .trees.factory import MoveAndValueTreeFactory

from players.boardevaluators.neural_networks.input_converters.factory import create_board_representation


def create_tree_and_value_builders(args: dict,
                                   syzygy,
                                   random_generator: random.Random) -> MoveSelector:
    node_evaluator: tree_man.NodeEvaluator = tree_man.create_node_evaluator(
        arg_board_evaluator=args['board_evaluator'],
        syzygy=syzygy
    )

    node_factory_name: str = args['node_factory_name'] if 'node_factory_name' in args else 'Base'

    tree_node_factory: node_factory.TreeNodeFactory = node_factory.create_node_factory(
        node_factory_name=node_factory_name
    )

    board_representation_factory: object = create_board_representation(args=args['board_evaluator'])

    algorithm_node_factory: node_factory.AlgorithmNodeFactory
    algorithm_node_factory = node_factory.AlgorithmNodeFactory(
        tree_node_factory=tree_node_factory,
        board_representation_factory=board_representation_factory)

    tree_factory = MoveAndValueTreeFactory(
        node_factory=algorithm_node_factory,
        node_evaluator=node_evaluator
    )

    tree_manager: tree_man.AlgorithmNodeTreeManager
    tree_manager = tree_man.create_algorithm_node_tree_manager(
        algorithm_node_factory=algorithm_node_factory,
        node_evaluator=node_evaluator,
    )

    tree_move_selector: TreeAndValuePlayer = TreeAndValuePlayer(args=args['tree_builder'],
                                                                tree_manager=tree_manager,
                                                                random_generator=random_generator,
                                                                tree_factory=tree_factory)
    return tree_move_selector

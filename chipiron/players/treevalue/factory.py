import random
from chipiron.players.boardevaluators.factory import create_node_evaluator
from .tree_and_value_player import TreeAndValuePlayer
from . import tree_manager as tree_man
from chipiron.players.treevalue import node_factory

from chipiron.players import MoveSelector


def create_tree_and_value_builders(args: dict,
                                   syzygy,
                                   random_generator: random.Random) -> MoveSelector:
    board_evaluators_wrapper = create_node_evaluator(args['board_evaluator'], syzygy)



    node_factory_name: str = args['node_factory_name'] if 'node_factory_name' in args else 'Base'

    tree_node_factory: node_factory.TreeNodeFactory = node_factory.create_node_factory(
        node_factory_name=node_factory_name)

    algorithm_node_factory: node_factory.AlgorithmNodeFactory
    algorithm_node_factory = node_factory.AlgorithmNodeFactory(
        tree_node_factory=tree_node_factory)

    tree_manager: tree_man.AlgorithmNodeTreeManager
    tree_manager = tree_man.create_algorithm_node_tree_manager(
        algorithm_node_factory=algorithm_node_factory,
        board_evaluators_wrapper=board_evaluators_wrapper)

    tree_move_selector: TreeAndValuePlayer = TreeAndValuePlayer(args=args['tree_builder'],
                                                                tree_manager=tree_manager,
                                                                random_generator=random_generator,
                                                                board_evaluators_wrapper=board_evaluators_wrapper,
                                                                node_factory=algorithm_node_factory)
    return tree_move_selector

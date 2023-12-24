import random

from .tree_and_value_player import TreeAndValuePlayer
from . import tree_manager as tree_man
from players.move_selector.treevalue import node_factory

from .trees.factory import MoveAndValueTreeFactory

from chipiron.players.boardevaluators.neural_networks.input_converters.factory import create_board_representation

from dataclasses import dataclass

TreeAndValueType = 'TreeAndValue'


@dataclass
class TreeAndValuePlayerArgs:
    type: str
    move_explorer_priority: str
    opening_type: str
    board_evaluator: tree_man.NodeEvaluatorsArgs

    def __post_init__(self):
        if self.type != TreeAndValueType:
            raise ValueError('Expecting TreeAndValue as name')


def create_tree_and_value_builders(args: TreeAndValuePlayerArgs,
                                   syzygy,
                                   random_generator: random.Random) -> TreeAndValuePlayer:
    node_evaluator: tree_man.NodeEvaluator = tree_man.create_node_evaluator(
        arg_board_evaluator=args.board_evaluator,
        syzygy=syzygy
    )

    # node_factory_name: str = args['node_factory_name'] if 'node_factory_name' in args else 'Base'
    node_factory_name: str = 'Base'

    tree_node_factory: node_factory.TreeNodeFactory = node_factory.create_node_factory(
        node_factory_name=node_factory_name
    )

    board_representation_factory: object = create_board_representation(
        board_representation=args.board_evaluator.representation
    )

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

    tree_move_selector: TreeAndValuePlayer = TreeAndValuePlayer(opening_type=args.opening_type,
                                                                tree_manager=tree_manager,
                                                                random_generator=random_generator,
                                                                tree_factory=tree_factory)
    return tree_move_selector

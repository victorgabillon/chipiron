import random
from chipiron.players.boardevaluators.factory import create_node_evaluator
from .tree_and_value_player import TreeAndValuePlayer
from chipiron.players.treevalue.node_selector.opening_instructions import OpeningInstructor
from chipiron.players.treevalue.stopping_criterion import StoppingCriterion, create_stopping_criterion
from . import node_selector as node_sel
from . import tree_manager as tree_man
from chipiron.players.treevalue import node_factory

from chipiron.players import MoveSelector


def create_tree_and_value_builders(args: dict,
                                   syzygy,
                                   random_generator: random.Random) -> MoveSelector:
    board_evaluators_wrapper = create_node_evaluator(args['board_evaluator'], syzygy)

    opening_instructor: OpeningInstructor \
        = OpeningInstructor(args['tree_builder']['opening_type'], random_generator) if 'opening_type' in args[
        'tree_builder'] else None

    node_selector: node_sel.NodeSelector = node_sel.create(
        arg=args['tree_builder'],
        opening_instructor=opening_instructor,
        random_generator=random_generator)

    stopping_criterion: StoppingCriterion = create_stopping_criterion(arg=args['tree_builder']['stopping_criterion'],
                                                                      node_selector=node_selector)

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

    tree_move_selector: TreeAndValuePlayer = TreeAndValuePlayer(arg=args['tree_builder'],
                                                                tree_manager=tree_manager,
                                                                stopping_criterion=stopping_criterion,
                                                                random_generator=random_generator,
                                                                board_evaluators_wrapper=board_evaluators_wrapper,
                                                                node_selector=node_selector,
                                                                node_factory=algorithm_node_factory)
    return tree_move_selector

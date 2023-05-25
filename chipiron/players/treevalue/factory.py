import random
from chipiron.players.boardevaluators.factory import create_node_evaluator
from .tree_and_value_player import TreeAndValuePlayer
from chipiron.players.treevalue.node_selector.opening_instructions import OpeningInstructor
from chipiron.players.treevalue.stopping_criterion import StoppingCriterion, create_stopping_criterion
from . import node_selector as node_sel
from . import tree_manager as tree_man

from chipiron.players import MoveSelector


def create_tree_and_value_builders(arg: dict,
                                   syzygy,
                                   random_generator: random.Random) -> MoveSelector:
    board_evaluators_wrapper = create_node_evaluator(arg['board_evaluator'], syzygy)

    opening_instructor: OpeningInstructor \
        = OpeningInstructor(arg['tree_builder']['opening_type'], random_generator) if 'opening_type' in arg[
        'tree_builder'] else None

    node_selector: node_sel.NodeSelector = node_sel.create(
        arg=arg['tree_builder'],
        opening_instructor=opening_instructor,
        random_generator=random_generator)

    stopping_criterion: StoppingCriterion = create_stopping_criterion(arg=arg['tree_builder']['stopping_criterion'],
                                                                      node_selector=node_selector)

    tree_manager: tree_man.TreeManager
    tree_manager = tree_man.create_tree_manager(
        args=arg,
        board_evaluators_wrapper=board_evaluators_wrapper)

    tree_move_selector: TreeAndValuePlayer = TreeAndValuePlayer(arg=arg['tree_builder'],
                                                                tree_manager=tree_manager,
                                                                stopping_criterion=stopping_criterion,
                                                                random_generator=random_generator,
                                                                board_evaluators_wrapper=board_evaluators_wrapper,
                                                                node_selector = node_selector)
    return tree_move_selector

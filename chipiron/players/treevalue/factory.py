import random
from . import node_selector
from chipiron.players.boardevaluators.factory import create_node_evaluator
from .tree_and_value_player import TreeAndValuePlayer
from chipiron.players.treevalue.node_selector.opening_instructions import OpeningInstructor
from .node_selector.node_selector import NodeSelector


def create_tree_and_value_builders(arg: dict,
                                   syzygy,
                                   random_generator: random.Random) -> TreeAndValuePlayer:
    board_evaluators_wrapper = create_node_evaluator(arg['board_evaluator'], syzygy)
    tree_builder_type: str = arg['tree_builder']['type']

    opening_instructor: OpeningInstructor = OpeningInstructor(arg['tree_builder']['opening_type'],
                                                              random_generator) if 'opening_type' in arg[
        'tree_builder'] else None


    node_move_opening_selector: NodeSelector
    match tree_builder_type:
        case 'Uniform':
            node_move_opening_selector = node_selector.Uniform(arg['tree_builder'], board_evaluators_wrapper)
        case 'RecurZipf':
            node_move_opening_selector = node_selector.RecurZipf(arg['tree_builder'], board_evaluators_wrapper)
        case 'RecurZipfBase':
            node_move_opening_selector = node_selector.RecurZipfBase(arg=arg['tree_builder'],
                                                                     random_generator=random_generator,
                                                                     opening_instructor=opening_instructor)
        case 'RecurZipfBase2':
            node_move_opening_selector = node_selector.RecurZipfBase2(arg['tree_builder'])
        case 'VertiZipf':
            node_move_opening_selector = node_selector.VertiZipf(arg['tree_builder'])
        case 'RecurVertiZipf':
            node_move_opening_selector = node_selector.RecurVertiZipf(arg['tree_builder'])
        case 'Sequool':
            node_move_opening_selector = node_selector.Sequool(arg['tree_builder'])
        case 'Sequool2':
            node_move_opening_selector = node_selector.Sequool2(arg['tree_builder'])
        case 'ZipfSequool':
            node_move_opening_selector = node_selector.ZipfSequool(arg['tree_builder'])
        case 'ZipfSequool2':
            node_move_opening_selector = node_selector.ZipfSequool2(arg['tree_builder'])
        case other:
            raise ('tree builder: can not find ' + other)

    tree_builder: TreeAndValuePlayer = TreeAndValuePlayer(arg=arg['tree_builder'],
                                                          random_generator=random_generator,
                                                          node_move_opening_selector=node_move_opening_selector,
                                                          board_evaluators_wrapper=board_evaluators_wrapper)
    return tree_builder

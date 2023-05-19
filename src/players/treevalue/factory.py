import random
from src.players.treevalue.node_selector.uniform import Uniform
from src.players.boardevaluators.factory import create_node_evaluator
from src.players.treevalue.node_selector.recur_zipf import RecurZipf
from src.players.treevalue.node_selector.recur_zipf_base import RecurZipfBase
from src.players.treevalue.node_selector.recur_zipf_base2 import RecurZipfBase2
from src.players.treevalue.node_selector.verti_zipf import VertiZipf
from src.players.treevalue.node_selector.recur_verti_zipf import RecurVertiZipf
from src.players.treevalue.node_selector.sequool import Sequool
from src.players.treevalue.node_selector.sequool2 import Sequool2
from src.players.treevalue.node_selector.zipf_sequool2 import ZipfSequool2
from src.players.treevalue.node_selector.zipf_sequool import ZipfSequool
from src.players.treevalue.tree_and_value_player import TreeAndValuePlayer
from src.players.treevalue.trees.opening_instructions import OpeningInstructor
from src.players.treevalue.trees.move_and_value_tree import MoveAndValueTreeBuilder
from src.players.treevalue.trees.factory import create_move_and_value_tree_builder
from src.players.treevalue.node_selector.node_selector import NodeSelector
from src.players.treevalue.stopping_criterion import create_stopping_criterion, StoppingCriterion


def create_tree_move_selector(arg: dict,
                              syzygy,
                              random_generator: random.Random) -> TreeAndValuePlayer:
    board_evaluators_wrapper = create_node_evaluator(arg['board_evaluator'], syzygy)
    tree_builder_type: str = arg['tree_builder']['type']

    opening_instructor: OpeningInstructor \
        = OpeningInstructor(arg['tree_builder']['opening_type'], random_generator) if 'opening_type' in arg['tree_builder'] else None

    print('pppppppppppp', opening_instructor, arg)

    node_move_opening_selector: NodeSelector
    match tree_builder_type:
        case 'Uniform':
            node_move_opening_selector = Uniform(arg['tree_builder'], board_evaluators_wrapper)
        case 'RecurZipf':
            node_move_opening_selector = RecurZipf(arg['tree_builder'], board_evaluators_wrapper)
        case 'RecurZipfBase':
            node_move_opening_selector = RecurZipfBase(arg=arg['tree_builder'],
                                                       random_generator=random_generator,
                                                       opening_instructor=opening_instructor)
        case 'RecurZipfBase2':
            node_move_opening_selector = RecurZipfBase2(arg['tree_builder'])
        case 'VertiZipf':
            node_move_opening_selector = VertiZipf(arg['tree_builder'])
        case 'RecurVertiZipf':
            node_move_opening_selector = RecurVertiZipf(arg['tree_builder'])
        case 'Sequool':
            node_move_opening_selector = Sequool(arg['tree_builder'])
        case 'Sequool2':
            node_move_opening_selector = Sequool2(arg['tree_builder'])
        case 'ZipfSequool':
            node_move_opening_selector = ZipfSequool(arg['tree_builder'])
        case 'ZipfSequool2':
            node_move_opening_selector = ZipfSequool2(arg['tree_builder'])
        case other:
            raise ('tree builder: can not find ' + other)

    stopping_criterion: StoppingCriterion = create_stopping_criterion(arg=arg['tree_builder']['stopping_criterion'],
                                                                      node_selector=node_move_opening_selector)

    tree_builder: MoveAndValueTreeBuilder = create_move_and_value_tree_builder(args=arg['tree_builder'])

    tree_move_selector: TreeAndValuePlayer = TreeAndValuePlayer(arg=arg['tree_builder'],
                                                                random_generator=random_generator,
                                                                node_move_opening_selector=node_move_opening_selector,
                                                                stopping_criterion=stopping_criterion,
                                                                board_evaluators_wrapper=board_evaluators_wrapper,
                                                                tree_builder=tree_builder)

    return tree_move_selector

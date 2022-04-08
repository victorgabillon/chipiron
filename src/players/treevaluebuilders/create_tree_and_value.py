import sys
from src.players.treevaluebuilders.uniform import Uniform
from src.players.boardevaluators.factory import create_node_evaluator
from src.players.treevaluebuilders.recur_zipf import RecurZipf
from src.players.treevaluebuilders.recur_zipf_base import RecurZipfBase
from src.players.treevaluebuilders.node_factory. import RecurZipfBase
from src.players.treevaluebuilders.recur_zipf_base2 import RecurZipfBase2
from src.players.treevaluebuilders.verti_zipf import VertiZipf
from src.players.treevaluebuilders.recur_verti_zipf import RecurVertiZipf
from src.players.treevaluebuilders.sequool import Sequool
from src.players.treevaluebuilders.sequool2 import Sequool2
from src.players.treevaluebuilders.zipf_sequool2 import ZipfSequool2
from src.players.treevaluebuilders.zipf_sequool import ZipfSequool
from src.players.treevaluebuilders.tree_and_value_player import TreeAndValuePlayer


def create_tree_and_value_builders(arg, syzygy, random_generator):
    board_evaluators_wrapper = create_node_evaluator(arg['board_evaluator'], syzygy)
    tree_builder_type = arg['tree_builder']['type']

    if tree_builder_type == 'Uniform':
        node_move_opening_selectoboard_evaluators_wrapperr = Uniform(arg['tree_builder'], board_evaluators_wrapper)
    elif tree_builder_type == 'RecurZipf':
        node_move_opening_selector = RecurZipf(arg['tree_builder'], board_evaluators_wrapper)
    elif tree_builder_type == 'RecurZipfBase':
        node_move_opening_selector = RecurZipfBase(arg['tree_builder'], random_generator)
        node_factory = RecurZipfBase()
    elif tree_builder_type == 'RecurZipfBase2':
        node_move_opening_selector = RecurZipfBase2(arg['tree_builder'])
    elif tree_builder_type == 'VertiZipf':
        node_move_opening_selector = VertiZipf(arg['tree_builder'])
    elif tree_builder_type == 'RecurVertiZipf':
        node_move_opening_selector = RecurVertiZipf(arg['tree_builder'])
    elif tree_builder_type == 'Sequool':
        node_move_opening_selector = Sequool(arg['tree_builder'])
    elif tree_builder_type == 'Sequool2':
        node_move_opening_selector = Sequool2(arg['tree_builder'])
    elif tree_builder_type == 'ZipfSequool':
        node_move_opening_selector = ZipfSequool(arg['tree_builder'])
    elif tree_builder_type == 'ZipfSequool2':
        node_move_opening_selector = ZipfSequool2(arg['tree_builder'])
    else:
        sys.exit('tree builder: can not find ' + arg['tree_builder']['type'])

    tree_builder = TreeAndValuePlayer(arg=arg,
                                      random_generator=random_generator,
                                      node_move_opening_selector=node_move_opening_selector,
                                      node_factory=node_factory,
                                      board_evaluators_wrapper=board_evaluators_wrapper)
    return tree_builder

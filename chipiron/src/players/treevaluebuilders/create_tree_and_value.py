import sys
from src.players.treevaluebuilders.uniform import Uniform
from src.players.boardevaluators.create_board_evaluator import create_board_evaluator
from src.players.treevaluebuilders.recur_zipf import RecurZipf
from src.players.treevaluebuilders.recur_zipf_base import RecurZipfBase
from src.players.treevaluebuilders.verti_zipf import VertiZipf
from src.players.treevaluebuilders.recur_verti_zipf import RecurVertiZipf
from src.players.treevaluebuilders.sequool import Sequool
from src.players.treevaluebuilders.sequool2 import Sequool2
from src.players.treevaluebuilders.zipf_sequool2 import ZipfSequool2

from src.players.treevaluebuilders.zipf_sequool import ZipfSequool


def create_tree_and_value_builders(arg, chess_simulator,syzygy):
    tree_builder_type = arg['tree_builder']['type']

    if tree_builder_type == 'Uniform':
        tree_builder = Uniform(arg['tree_builder'])
    elif tree_builder_type == 'RecurZipf':
        tree_builder = RecurZipf(arg['tree_builder'])
    elif tree_builder_type == 'RecurZipfBase':
        tree_builder = RecurZipfBase(arg['tree_builder'])
    elif tree_builder_type == 'VertiZipf':
        tree_builder = VertiZipf(arg['tree_builder'])
    elif tree_builder_type == 'RecurVertiZipf':
        tree_builder = RecurVertiZipf(arg['tree_builder'])
    elif tree_builder_type == 'Sequool':
        tree_builder = Sequool(arg['tree_builder'])
    elif tree_builder_type == 'Sequool2':
        tree_builder = Sequool2(arg['tree_builder'])
    elif tree_builder_type == 'ZipfSequool':
        tree_builder = ZipfSequool(arg['tree_builder'])
    elif tree_builder_type == 'ZipfSequool2':
        tree_builder = ZipfSequool2(arg['tree_builder'])
    else:
        sys.exit('tree builder: can not find ' + arg['tree_builder']['type'])

    tree_builder.environment = chess_simulator
    tree_builder.board_evaluators_wrapper = create_board_evaluator(arg['board_evaluator'], syzygy)

    return tree_builder

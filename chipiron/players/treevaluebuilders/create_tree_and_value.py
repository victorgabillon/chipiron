import sys
from players.treevaluebuilders.Uniform import Uniform
from players.boardevaluators.create_board_evaluator import create_board_evaluator
from players.treevaluebuilders.recur_zipf import RecurZipf
from players.treevaluebuilders.VertiZipf import VertiZipf
from players.treevaluebuilders.recur_verti_zipf import RecurVertiZipf
from players.treevaluebuilders.sequool import Sequool
from players.treevaluebuilders.sequool2 import Sequool2
from players.treevaluebuilders.ZipfSequool2 import ZipfSequool2

from players.treevaluebuilders.ZipfSequool import ZipfSequool


def create_tree_and_value_builders(arg, chess_simulator,syzygy):
    tree_builder_type = arg['tree_builder']['type']

    if tree_builder_type == 'Uniform':
        tree_builder = Uniform(arg['tree_builder'])
    elif tree_builder_type == 'RecurZipf':
        tree_builder = RecurZipf(arg['tree_builder'])
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
        sys.exit('cant find ' + arg['tree_builder']['type'])

    tree_builder.environment = chess_simulator
    tree_builder.board_evaluators_wrapper = create_board_evaluator(arg['board_evaluator'], syzygy)

    return tree_builder

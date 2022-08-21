import random
from chipiron.players.boardevaluators.factory import create_node_evaluator
from .tree_and_value_player import TreeAndValuePlayer


def create_tree_and_value_builders(arg: dict,
                                   syzygy,
                                   random_generator: random.Random) -> TreeAndValuePlayer:

    board_evaluators_wrapper = create_node_evaluator(arg['board_evaluator'], syzygy)

    tree_builder: TreeAndValuePlayer = TreeAndValuePlayer(arg=arg['tree_builder'],
                                                          random_generator=random_generator,
                                                          board_evaluators_wrapper=board_evaluators_wrapper)
    return tree_builder

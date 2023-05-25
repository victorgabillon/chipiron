import chipiron as ch
from .tree_exploration import create_tree_exploration
from chipiron.players.treevalue.stopping_criterion import StoppingCriterion
from . import tree_manager as tree_man

class TreeAndValuePlayer:
    # pretty empty class but might be useful when dealing with multi round and time , no?

    def __init__(self,
                 arg: dict,
                 node_selector,
                 random_generator,
                 board_evaluators_wrapper,
                 stopping_criterion: StoppingCriterion,
                 tree_manager: tree_man.TreeManager):
        self.board_evaluators_wrapper = board_evaluators_wrapper
        self.arg = arg
        self.random_generator = random_generator
        self.stopping_criterion = stopping_criterion
        self.tree_manager = tree_manager
        self.node_selector = node_selector

    def select_move(self, board: ch.chess.BoardChi):
        tree_exploration = create_tree_exploration(
            tree_manager= self.tree_manager,
            stopping_criterion=self.stopping_criterion,
            node_selector=self.node_selector,
            arg=self.arg,
            random_generator=self.random_generator,
            board=board,
            board_evaluators_wrapper=self.board_evaluators_wrapper)
        best_move = tree_exploration.explore()

        return best_move

    def print_info(self):
        super().print_info()
        print('type: Tree and Value')

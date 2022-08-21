import chipiron as ch
from .tree_exploration import create_tree_exploration


class TreeAndValuePlayer:
    # pretty empty class but might be useful when dealing with multi round and time , no?

    def __init__(self,
                 arg: dict,
                 random_generator,
                 board_evaluators_wrapper):
        self.board_evaluators_wrapper = board_evaluators_wrapper
        self.arg = arg
        self.random_generator = random_generator

    def select_move(self, board: ch.chess.BoardChi):
        tree_exploration = create_tree_exploration(
            arg=self.arg,
            random_generator=self.random_generator,
            board=board,
            board_evaluators_wrapper=self.board_evaluators_wrapper)
        best_move = tree_exploration.explore()

        return best_move

    def print_info(self):
        super().print_info()
        print('type: Tree and Value')

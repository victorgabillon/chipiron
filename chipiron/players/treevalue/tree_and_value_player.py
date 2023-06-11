import chipiron as ch
from .tree_exploration import create_tree_exploration
from . import tree_manager as tree_man

class TreeAndValuePlayer:
    # pretty empty class but might be useful when dealing with multi round and time , no?

    tree_manager : tree_man.AlgorithmNodeTreeManager

    def __init__(self,
                 args: dict,
                 random_generator,
                 board_evaluators_wrapper,
                 tree_manager: tree_man.AlgorithmNodeTreeManager,
                 node_factory):
        self.board_evaluators_wrapper = board_evaluators_wrapper
        self.args = args
        self.random_generator = random_generator
        self.tree_manager = tree_manager
        self.node_factory = node_factory

    def select_move(self, board: ch.chess.BoardChi):
        tree_exploration = create_tree_exploration(
            tree_manager= self.tree_manager,
            args=self.args,
            random_generator=self.random_generator,
            board=board,
            board_evaluators_wrapper=self.board_evaluators_wrapper,
        node_factory=self.node_factory)
        best_move = tree_exploration.explore()

        return best_move

    def print_info(self):
        super().print_info()
        print('type: Tree and Value')

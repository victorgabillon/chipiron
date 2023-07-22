import chipiron as ch
from .tree_exploration import create_tree_exploration
from . import tree_manager as tree_man
from .trees.factory import MoveAndValueTreeFactory
import chipiron.environments.chess.board as boards

class TreeAndValuePlayer:
    # pretty empty class but might be useful when dealing with multi round and time , no?

    tree_manager: tree_man.AlgorithmNodeTreeManager

    def __init__(self,
                 args: dict,
                 random_generator,
                 tree_manager: tree_man.AlgorithmNodeTreeManager,
                 tree_factory: MoveAndValueTreeFactory):
        self.args = args
        self.random_generator = random_generator
        self.tree_manager = tree_manager
        self.tree_factory = tree_factory

    def select_move(self, board: boards.BoardChi):
        tree_exploration = create_tree_exploration(
            tree_manager=self.tree_manager,
            args=self.args,
            random_generator=self.random_generator,
            starting_board=board,
            tree_factory=self.tree_factory)
        best_move = tree_exploration.explore()

        return best_move

    def print_info(self):
        super().print_info()
        print('type: Tree and Value')

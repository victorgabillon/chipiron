from .tree_exploration import create_tree_exploration
from . import tree_manager as tree_man
from .trees.factory import MoveAndValueTreeFactory
import chipiron.environments.chess.board as boards


class TreeAndValuePlayer:
    # pretty empty class but might be useful when dealing with multi round and time , no?

    tree_manager: tree_man.AlgorithmNodeTreeManager

    def __init__(self,
                 random_generator,
                 opening_type,
                 tree_manager: tree_man.AlgorithmNodeTreeManager,
                 tree_factory: MoveAndValueTreeFactory):
        self.random_generator = random_generator
        self.tree_manager = tree_manager
        self.tree_factory = tree_factory
        self.opening_type = opening_type

    def select_move(self, board: boards.BoardChi):
        tree_exploration = create_tree_exploration(
            tree_manager=self.tree_manager,
            random_generator=self.random_generator,
            starting_board=board,
            tree_factory=self.tree_factory,
            opening_type=self.opening_type)
        best_move = tree_exploration.explore()

        return best_move

    def print_info(self):
        super().print_info()
        print('type: Tree and Value')

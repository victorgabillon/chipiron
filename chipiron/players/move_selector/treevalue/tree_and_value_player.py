import random

from .tree_exploration import create_tree_exploration, TreeExploration
from .trees.factory import MoveAndValueTreeFactory
import chipiron.environments.chess.board as boards
from chipiron.players.move_selector.move_selector import MoveRecommendation
from .stopping_criterion import AllStoppingCriterionArgs
from . import recommender_rule
from . import tree_manager as tree_man
from dataclasses import dataclass
from chipiron.players.move_selector.treevalue.search_factory import NodeSelectorFactory
from chipiron.utils import seed


@dataclass
class TreeAndValueMoveSelector:
    # pretty empty class but might be useful when dealing with multi round and time , no?

    tree_manager: tree_man.AlgorithmNodeTreeManager
    tree_factory: MoveAndValueTreeFactory
    stopping_criterion_args: AllStoppingCriterionArgs
    node_selector_create: NodeSelectorFactory
    random_generator: random.Random
    recommend_move_after_exploration: recommender_rule.AllRecommendFunctionsArgs

    def select_move(
            self,
            board: boards.BoardChi,
            move_seed: seed
    ) -> MoveRecommendation:
        tree_exploration: TreeExploration = create_tree_exploration(
            tree_manager=self.tree_manager,
            node_selector_create=self.node_selector_create,
            starting_board=board,
            tree_factory=self.tree_factory,
            stopping_criterion_args=self.stopping_criterion_args,
            recommend_move_after_exploration=self.recommend_move_after_exploration
        )
        self.random_generator.seed(move_seed)
        move_recommendation: MoveRecommendation = tree_exploration.explore(
            random_generator=self.random_generator
        )

        return move_recommendation

    def print_info(self):
        super().print_info()
        print('type: Tree and Value')

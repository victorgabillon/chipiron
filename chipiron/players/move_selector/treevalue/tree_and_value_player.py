from .tree_exploration import create_tree_exploration, TreeExploration
from .trees.factory import MoveAndValueTreeFactory
import chipiron.environments.chess.board as boards
from chipiron.players.move_selector.move_selector import MoveRecommendation
from .stopping_criterion import AllStoppingCriterionArgs
from . import recommender_rule
from . import tree_manager as tree_man
from . import node_selector


class TreeAndValueMoveSelector:
    # pretty empty class but might be useful when dealing with multi round and time , no?

    tree_manager: tree_man.AlgorithmNodeTreeManager
    opening_type: node_selector.OpeningType
    tree_factory: MoveAndValueTreeFactory
    stopping_criterion_args: AllStoppingCriterionArgs

    def __init__(self,
                 random_generator,
                 opening_type: node_selector.OpeningType,
                 node_selector_args: node_selector.AllNodeSelectorArgs,
                 tree_manager: tree_man.AlgorithmNodeTreeManager,
                 tree_factory: MoveAndValueTreeFactory,
                 stopping_criterion_args: AllStoppingCriterionArgs,
                 recommend_move_after_exploration: recommender_rule.RecommenderRule
                 ) -> None:
        self.random_generator = random_generator
        self.tree_manager = tree_manager
        self.tree_factory = tree_factory
        self.opening_type = opening_type
        self.node_selector_args = node_selector_args
        self.stopping_criterion_args = stopping_criterion_args
        self.recommend_move_after_exploration = recommend_move_after_exploration

    def select_move(
            self,
            board: boards.BoardChi
    ) -> MoveRecommendation:
        tree_exploration: TreeExploration = create_tree_exploration(
            tree_manager=self.tree_manager,
            node_selector_args=self.node_selector_args,
            random_generator=self.random_generator,
            starting_board=board,
            tree_factory=self.tree_factory,
            opening_type=self.opening_type,
            stopping_criterion_args=self.stopping_criterion_args,
            recommend_move_after_exploration=self.recommend_move_after_exploration
        )
        move_recommendation: MoveRecommendation = tree_exploration.explore(random_generator=self.random_generator)

        return move_recommendation

    def print_info(self):
        super().print_info()
        print('type: Tree and Value')

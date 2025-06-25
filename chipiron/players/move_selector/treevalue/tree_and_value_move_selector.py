"""
This module contains the implementation of the TreeAndValueMoveSelector class, which is responsible for selecting moves
based on a tree and value strategy.

The TreeAndValueMoveSelector class uses a tree-based approach to explore possible moves and select the best move based on
a value function. It utilizes a tree manager, a tree factory, stopping criterion arguments, a node selector factory, a
random generator, and recommendation functions to guide the move selection process.

The TreeAndValueMoveSelector class provides the following methods:
- select_move: Selects the best move based on the tree and value strategy.
- print_info: Prints information about the move selector type.
"""

import queue
import random
from dataclasses import dataclass

import chipiron.environments.chess.board as boards
from chipiron.players.move_selector.move_selector import MoveRecommendation
from chipiron.players.move_selector.treevalue.progress_monitor.progress_monitor import (
    AllStoppingCriterionArgs,
)
from chipiron.players.move_selector.treevalue.search_factory import NodeSelectorFactory
from chipiron.utils import seed
from chipiron.utils.dataclass import IsDataclass

from . import recommender_rule
from . import tree_manager as tree_man
from .tree_exploration import TreeExploration, create_tree_exploration
from .trees.factory import MoveAndValueTreeFactory


@dataclass
class TreeAndValueMoveSelector:
    """
    The TreeAndValueMoveSelector class is responsible for selecting moves based on a tree and value strategy.

    Attributes:
    - tree_manager: The tree manager responsible for managing the algorithm nodes.
    - tree_factory: The tree factory responsible for creating move and value trees.
    - stopping_criterion_args: The stopping criterion arguments used to determine when to stop the tree exploration.
    - node_selector_create: The node selector factory used to create node selectors for tree exploration.
    - random_generator: The random generator used for randomization during tree exploration.
    - recommend_move_after_exploration: The recommendation functions used to recommend a move after tree exploration.
    """

    # pretty empty class but might be useful when dealing with multi round and time , no?

    tree_manager: tree_man.AlgorithmNodeTreeManager
    tree_factory: MoveAndValueTreeFactory
    stopping_criterion_args: AllStoppingCriterionArgs
    node_selector_create: NodeSelectorFactory
    random_generator: random.Random
    recommend_move_after_exploration: recommender_rule.AllRecommendFunctionsArgs
    queue_progress_player: queue.Queue[IsDataclass] | None

    def select_move(self, board: boards.IBoard, move_seed: seed) -> MoveRecommendation:
        """
        Selects the best move based on the tree and value strategy.

        Args:
        - board: The current board state.
        - move_seed: The seed used for randomization during move selection.

        Returns:
        - The recommended move based on the tree and value strategy.
        """
        tree_exploration: TreeExploration = self.create_tree_exploration(board=board)
        self.random_generator.seed(move_seed)

        move_recommendation: MoveRecommendation = tree_exploration.explore(
            random_generator=self.random_generator
        ).move_recommendation

        return move_recommendation

    def create_tree_exploration(
        self,
        board: boards.IBoard,
    ) -> TreeExploration:
        tree_exploration: TreeExploration = create_tree_exploration(
            tree_manager=self.tree_manager,
            node_selector_create=self.node_selector_create,
            starting_board=board,
            tree_factory=self.tree_factory,
            stopping_criterion_args=self.stopping_criterion_args,
            recommend_move_after_exploration=self.recommend_move_after_exploration,
            queue_progress_player=self.queue_progress_player,
        )
        return tree_exploration

    def print_info(self) -> None:
        """
        Prints information about the move selector type.
        """
        print("type: Tree and Value")

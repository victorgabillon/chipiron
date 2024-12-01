"""
This module contains the implementation of the TreeExploration class, which is responsible for managing a search
 for the best move in a given chess position using a tree-based approach.

The TreeExploration class is used to create and manage a tree structure that represents the possible moves and
 their evaluations in a chess position. It provides methods for exploring the tree, selecting the best move,
  and printing information during the move computation.

The module also includes helper functions for creating a TreeExploration object and its dependencies.

Classes:
- TreeExploration: Manages the search for the best move using a tree-based approach.

Functions:
- create_tree_exploration: Creates a TreeExploration object with the specified dependencies.
"""

import random
from dataclasses import dataclass
from typing import Any

import chipiron.environments.chess.board as boards
from chipiron.environments.chess.move import IMove
from chipiron.players.move_selector.move_selector import MoveRecommendation
from chipiron.players.move_selector.treevalue.recommender_rule.recommender_rule import \
    recommend_move_after_exploration_generic
from chipiron.players.move_selector.treevalue.search_factory import NodeSelectorFactory
from . import node_selector as node_sel
from . import recommender_rule
from . import tree_manager as tree_man
from . import trees
from .stopping_criterion import StoppingCriterion, create_stopping_criterion, AllStoppingCriterionArgs
from .trees.factory import MoveAndValueTreeFactory


@dataclass
class TreeExploration:
    """
    Tree Exploration is an object to manage one best move search.

    Attributes:
    - tree: The tree structure representing the possible moves and their evaluations.
    - tree_manager: The manager for the tree structure.
    - node_selector: The selector for choosing nodes and moves to open in the tree.
    - recommend_move_after_exploration: The recommender rule for selecting the best move after the tree exploration.
    - stopping_criterion: The stopping criterion for determining when to stop the tree exploration.

    Methods:
    - print_info_during_move_computation: Prints information during the move computation.
    - explore: Explores the tree to find the best move.
    """
    # TODO Not sure why this class is not simply the TreeAndValuePlayer Class
    #  but might be useful when dealing with multi round and time , no?

    tree: trees.MoveAndValueTree
    tree_manager: tree_man.AlgorithmNodeTreeManager
    node_selector: node_sel.NodeSelector
    recommend_move_after_exploration: recommender_rule.AllRecommendFunctionsArgs
    stopping_criterion: StoppingCriterion

    def print_info_during_move_computation(self, random_generator: random.Random) -> None:
        """
        Prints information during the move computation.

        Args:
        - random_generator: The random number generator.

        Returns:
        - None
        """
        current_best_move: str
        if self.tree.root_node.minmax_evaluation.best_move_sequence:
            current_best_move = str(self.tree.root_node.minmax_evaluation.best_move_sequence[0])
        else:
            current_best_move = '?'
        if random_generator.random() < 1:
            str_progress = self.stopping_criterion.get_string_of_progress(self.tree)
            print(
                f'{str_progress} | current best move:  {current_best_move} | current white value: {self.tree.root_node.minmax_evaluation.value_white_minmax})')
            # ,end='\r')
            self.tree.root_node.minmax_evaluation.print_moves_sorted_by_value_and_exploration()
            self.tree_manager.print_best_line(tree=self.tree)

    def explore(
            self,
            random_generator: random.Random
    ) -> MoveRecommendation:
        """
        Explores the tree to find the best move.

        Args:
        - random_generator: The random number generator.

        Returns:
        - MoveRecommendation: The recommended move and its evaluation.
        """

        # by default the first tree expansion is the creation of the tree node
        tree_expansions: tree_man.TreeExpansions = tree_man.TreeExpansions()

        tree_expansion: tree_man.TreeExpansion = tree_man.TreeExpansion(
            child_node=self.tree.root_node,
            parent_node=None,
            board_modifications=None,
            creation_child_node=True,
            move=None
        )
        tree_expansions.add_creation(tree_expansion=tree_expansion)

        while self.stopping_criterion.should_we_continue(tree=self.tree):
            assert (not self.tree.root_node.is_over())
            # print info
            self.print_info_during_move_computation(random_generator=random_generator)

            # choose the moves and nodes to open
            opening_instructions: node_sel.OpeningInstructions
            opening_instructions = self.node_selector.choose_node_and_move_to_open(
                tree=self.tree,
                latest_tree_expansions=tree_expansions
            )

            # make sure we do not break the stopping criterion
            opening_instructions_subset: node_sel.OpeningInstructions
            opening_instructions_subset = self.stopping_criterion.respectful_opening_instructions(
                opening_instructions=opening_instructions,
                tree=self.tree)

            # open the nodes
            tree_expansions = self.tree_manager.open_instructions(
                tree=self.tree,
                opening_instructions=opening_instructions_subset
            )

            # self.node_selector.communicate_expansions()
            self.tree_manager.update_backward(tree_expansions=tree_expansions)
            self.tree_manager.update_indices(tree=self.tree)

        # trees.save_raw_data_to_file(tree=self.tree)
        # self.tree_manager.print_some_stats(tree=self.tree)
        # for move, child in self.tree.root_node.moves_children.items():
        #    print(f'{move} {self.tree.root_node.moves_children[move].minmax_evaluation.get_value_white()}'
        #          f' {child.minmax_evaluation.over_event.get_over_tag()}')
        # print(f'evaluation for white: {self.tree.root_node.minmax_evaluation.get_value_white()}')

        best_move: IMove = recommend_move_after_exploration_generic(
            self.recommend_move_after_exploration,
            tree=self.tree,
            random_generator=random_generator)

        self.tree_manager.print_best_line(tree=self.tree)  # todo maybe almost best chosen line no?

        move_recommendation: MoveRecommendation = MoveRecommendation(
            move=best_move,
            evaluation=self.tree.evaluate()
        )

        return move_recommendation


def create_tree_exploration(
        node_selector_create: NodeSelectorFactory,
        starting_board: boards.IBoard[Any],
        tree_manager: tree_man.AlgorithmNodeTreeManager,
        tree_factory: MoveAndValueTreeFactory,
        stopping_criterion_args: AllStoppingCriterionArgs,
        recommend_move_after_exploration: recommender_rule.AllRecommendFunctionsArgs
) -> TreeExploration:
    """
    Creates a TreeExploration object with the specified dependencies.

    Args:
    - node_selector_create: The factory function for creating the node selector.
    - starting_board: The starting chess board position.
    - tree_manager: The manager for the tree structure.
    - tree_factory: The factory for creating the tree structure.
    - stopping_criterion_args: The arguments for creating the stopping criterion.
    - recommend_move_after_exploration: The recommender rule for selecting the best move after the tree exploration.

    Returns:
    - TreeExploration: The created TreeExploration object.
    """

    # creates the tree
    move_and_value_tree: trees.MoveAndValueTree = tree_factory.create(starting_board=starting_board)

    # creates the node selector
    node_selector: node_sel.NodeSelector = node_selector_create()

    stopping_criterion: StoppingCriterion = create_stopping_criterion(
        args=stopping_criterion_args,
        node_selector=node_selector
    )

    tree_exploration: TreeExploration = TreeExploration(
        tree=move_and_value_tree,
        tree_manager=tree_manager,
        stopping_criterion=stopping_criterion,
        node_selector=node_selector,
        recommend_move_after_exploration=recommend_move_after_exploration
    )
    return tree_exploration

"""
Tree Exploration
"""
import random
from chipiron.players.move_selector.treevalue.search_factory import NodeSelectorFactory
import chess

from .trees.factory import MoveAndValueTreeFactory
from .stopping_criterion import StoppingCriterion, create_stopping_criterion, StoppingCriterionArgs
import chipiron.environments.chess.board as boards

from . import trees
from . import tree_manager as tree_man
from . import node_selector as node_sel
from . import recommender_rule

from dataclasses import dataclass
from chipiron.players.move_selector.move_selector import MoveRecommendation
from chipiron.players.move_selector.treevalue.recommender_rule.recommender_rule import recommend_move_after_exploration_generic

@dataclass
class TreeExploration:
    """
    Tree Exploration is an object to manage one best move search
    """
    # TODO Not sure why this class is not simply the TreeAndValuePlayer Class
    #  but might be useful when dealing with multi round and time , no?

    tree: trees.MoveAndValueTree
    tree_manager: tree_man.AlgorithmNodeTreeManager
    node_selector: node_sel.NodeSelector
    recommend_move_after_exploration: recommender_rule.RecommenderRule
    stopping_criterion: StoppingCriterion

    def print_info_during_move_computation(self,
                                           random_generator
                                           ):
        if self.tree.root_node.minmax_evaluation.best_node_sequence:
            current_best_child = self.tree.root_node.minmax_evaluation.best_node_sequence[0]
            current_best_move = self.tree.root_node.moves_children.inverse[current_best_child]
            assert (
                    self.tree.root_node.minmax_evaluation.get_value_white() == current_best_child.minmax_evaluation.get_value_white())

        else:
            current_best_move = '?'
        if random_generator.random() < 5:
            str_progress = self.stopping_criterion.get_string_of_progress(self.tree)
            print(
                f'{str_progress} | current best move:  {current_best_move} | current white value: {self.tree.root_node.minmax_evaluation.value_white_minmax})')
            # ,end='\r')
            self.tree.root_node.minmax_evaluation.print_children_sorted_by_value_and_exploration()
            self.tree_manager.print_best_line(tree=self.tree)

    def explore(
            self,
            random_generator: random.Random
    ) -> MoveRecommendation:

        # by default the first tree expansion is the creation of the tree node
        tree_expansions: tree_man.TreeExpansions = tree_man.TreeExpansions()
        tree_expansion: tree_man.TreeExpansion = tree_man.TreeExpansion(
            child_node=self.tree.root_node,
            parent_node=None,
            board_modifications=None,
            creation_child_node=True
        )
        tree_expansions.add_creation(tree_expansion)

        while self.stopping_criterion.should_we_continue(tree=self.tree):
            assert (not self.tree.root_node.is_over())
            # print info
            # self.print_info_during_move_computation(random_generator=random_generator)

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

          #  opening_instructions_subset.print_info()
            # open the nodes
            tree_expansions: tree_man.TreeExpansions = self.tree_manager.open_instructions(
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

        best_move: chess.Move = recommend_move_after_exploration_generic(
            self.recommend_move_after_exploration,
            tree=self.tree,
            random_generator=random_generator)

        self.tree_manager.print_best_line(tree=self.tree)  # todo maybe almost best chosen line no?

        move_recommendation: MoveRecommendation = MoveRecommendation(
            move=best_move,
            evaluation=self.tree.root_node.minmax_evaluation.get_value_white())
        return move_recommendation


def create_tree_exploration(
        node_selector_create: NodeSelectorFactory,
        starting_board: boards.BoardChi,
        tree_manager: tree_man.AlgorithmNodeTreeManager,
        tree_factory: MoveAndValueTreeFactory,
        stopping_criterion_args: StoppingCriterionArgs,
        recommend_move_after_exploration: recommender_rule.RecommenderRule
) -> TreeExploration:
    """
    Creation of the tree exploration to init all object necessary
     for a tree search to find one move in a given stating board
    Args:
        node_selector_create:
        stopping_criterion_args:
        recommend_move_after_exploration:
        starting_board:
        tree_manager:
        tree_factory:

    Returns:

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

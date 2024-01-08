from dataclasses import dataclass
from typing import Protocol, Literal
from enum import Enum

import chess

from chipiron.utils.small_tools import softmax
from chipiron.players.boardevaluators.over_event import HowOver

import chipiron.players.move_selector.treevalue.trees as trees

AlmostEqualLogistic_Literal = 'almost_equal_logistic'


# theses are functions but i still use dataclasses instead
# of partial to be able to easily construct from yaml files using dacite

@dataclass
class AlmostEqualLogistic:
    type: Literal[AlmostEqualLogistic_Literal]
    temperature: int

    def __call__(
            self,
            tree: trees.MoveAndValueTree,
            random_generator
    ) -> chess.Move:
        # TODO this should be given at construction but postponed for now because of dataclasses
        from random import Random
        # find the best first move allowing for random choice for almost equally valued moves.
        best_root_children = tree.root_node.minmax_evaluation.get_all_of_the_best_moves(
            how_equal='almost_equal_logistic')
        print('We have as bests: ',
              [tree.root_node.moves_children.inverse[best] for best in best_root_children])
        best_child = random_generator.choice(best_root_children)
        if tree.root_node.minmax_evaluation.over_event.how_over == HowOver.WIN:
            assert (best_child.minmax_evaluation.over_event.how_over == HowOver.WIN)
        best_move = tree.root_node.moves_children.inverse[best_child]

        return best_move


AllRecommendFunctionsArgs = AlmostEqualLogistic


class RecommenderRuleTypes(str, Enum):
    AlmostEqualLogistic: str = AlmostEqualLogistic_Literal


@dataclass
class RecommenderRule(Protocol):
    type: RecommenderRuleTypes

    def __call__(
            self,
            tree: trees.MoveAndValueTree,
            random_generator
    ) -> chess.Move:
        ...


def recommend_move_after_exploration(
        self,
        tree: trees.MoveAndValueTree
):
    # todo the preference for action that have been explored more is not super clear,
    #  is it weel implemented, ven for debug?

    # for debug we fix the choice in the next lines
    # if global_variables.deterministic_behavior:
    #     print(' FIXED CHOICE FOR DEBUG')
    #     best_child = self.tree.root_node.get_all_of_the_best_moves(how_equal='considered_equal')[-1]
    #     print('We have as best: ', self.tree.root_node.moves_children.inverse[best_child])
    #     best_move = self.tree.root_node.moves_children.inverse[best_child]

    selection_rule = self.move_selection_rule.type
    if selection_rule == 'softmax':
        temperature = self.move_selection_rule.temperature
        values = [tree.root_node.subjective_value_of(node) for node in
                  tree.root_node.moves_children.values()]

        softmax_ = softmax(values, temperature)
        print(values)
        print('SOFTMAX', temperature, [i / sum(softmax_) for i in softmax_],
              sum([i / sum(softmax_) for i in softmax_]))

        move_as_list = self.random_generator.choices(
            list(tree.root_node.moves_children.keys()),
            weights=softmax_, k=1)
        best_move = move_as_list[0]
    elif selection_rule == 'almost_equal' or selection_rule == 'almost_equal_logistic':
        # find the best first move allowing for random choice for almost equally valued moves.
        best_root_children = tree.root_node.minmax_evaluation.get_all_of_the_best_moves(
            how_equal=selection_rule)
        print('We have as bests: ',
              [tree.root_node.moves_children.inverse[best] for best in best_root_children])
        best_child = self.random_generator.choice(best_root_children)
        if tree.root_node.minmax_evaluation.over_event.how_over == HowOver.WIN:
            assert (best_child.minmax_evaluation.over_event.how_over == HowOver.WIN)
        best_move = tree.root_node.moves_children.inverse[best_child]
    else:
        raise (ValueError('move_selection_rule is not valid it seems'))
    return best_move

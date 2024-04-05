import random
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Literal

import chess

import chipiron.players.move_selector.treevalue.trees as trees
from chipiron.players.boardevaluators.basic_evaluation import value_base
from chipiron.players.boardevaluators.over_event import HowOver
from chipiron.players.move_selector.treevalue.nodes.algorithm_node.algorithm_node import AlgorithmNode
from chipiron.players.move_selector.treevalue.nodes.utils import is_winning
from chipiron.utils.small_tools import softmax


class RecommenderRuleTypes(str, Enum):
    AlmostEqualLogistic: str = 'almost_equal_logistic'
    Softmax: str = 'softmax'


# theses are functions but i still use dataclasses instead
# of partial to be able to easily construct from yaml files using dacite

@dataclass
class AlmostEqualLogistic:
    type: Literal[RecommenderRuleTypes.AlmostEqualLogistic]
    temperature: float

    def __call__(
            self,
            tree: trees.MoveAndValueTree,
            random_generator: random.Random
    ) -> chess.Move:
        # TODO this should be given at construction but postponed for now because of dataclasses
        # find the best first move allowing for random choice for almost equally valued moves.
        best_root_children = tree.root_node.minmax_evaluation.get_all_of_the_best_moves(
            how_equal='almost_equal_logistic')
        print('We have as bests: ',
              [tree.root_node.moves_children.inverse[best] for best in best_root_children])
        best_child = random_generator.choice(best_root_children)
        if tree.root_node.minmax_evaluation.over_event.how_over == HowOver.WIN:
            assert (best_child.minmax_evaluation.over_event.how_over == HowOver.WIN)
        best_move = tree.root_node.moves_children.inverse[best_child]

        assert isinstance(best_move, chess.Move)
        return best_move


@dataclass
class SoftmaxRule:
    type: Literal[RecommenderRuleTypes.Softmax]
    temperature: float

    def __call__(
            self,
            tree: trees.MoveAndValueTree,
            random_generator: random.Random
    ) -> chess.Move:
        values = [tree.root_node.minmax_evaluation.subjective_value_of(node.minmax_evaluation) for node in
                  tree.root_node.moves_children.values()]

        softmax_ = list(softmax(values, self.temperature))
        print(values)
        print('SOFTMAX', self.temperature, [i / sum(softmax_) for i in softmax_],
              sum([i / sum(softmax_) for i in softmax_]))

        move_as_list = random_generator.choices(
            list(tree.root_node.moves_children.keys()),
            weights=softmax_, k=1)
        best_move: chess.Move = move_as_list[0]
        return best_move


AllRecommendFunctionsArgs = AlmostEqualLogistic | SoftmaxRule


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


def recommend_move_after_exploration_generic(
        recommend_move_after_exploration: AllRecommendFunctionsArgs,
        tree: trees.MoveAndValueTree,
        random_generator: random.Random
) -> chess.Move:
    # if the situation is winning, we ask to play the move that is the most likely
    # to end the game fast by capturing pieces if possible
    is_winning_situation: bool = is_winning(
        node_minmax_evaluation=tree.root_node.minmax_evaluation,
        color=tree.root_node.board.turn
    )
    over: bool = tree.root_node.is_over()
    if is_winning_situation and not over:
        # value of pieces of the opponent before the move
        value_father: int = value_base(
            board=tree.root_node.board,
            color=not tree.root_node.board.turn
        )

        child: AlgorithmNode
        best_value: int | None = None
        best_move: chess.Move | None = None
        for move, child in tree.root_node.moves_children.items():
            # value of pieces of the opponent after that move
            value_child: int = value_base(
                board=child.board,
                color=child.board.turn
            )
            value: int = value_father - value_child

            still_wining_after_move: bool = is_winning(
                node_minmax_evaluation=child.minmax_evaluation,
                color=tree.root_node.board.turn
            )
            if still_wining_after_move and (best_value is None or best_value < value):
                best_value = value
                best_move = move
        assert best_value is not None
        if best_value > 0:
            assert isinstance(best_move, chess.Move)
            return best_move

    # base case
    return recommend_move_after_exploration(tree=tree, random_generator=random_generator)

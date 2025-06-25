"""
This module defines recommender rules for selecting moves in a tree-based move selector.

The recommender rules are implemented as data classes that define a `__call__` method. The `__call__` method takes a
`MoveAndValueTree` object and a random generator, and returns a recommended chess move.

The available recommender rule types are defined in the `RecommenderRuleTypes` enum.

The module also defines a `RecommenderRule` protocol that all recommender rule classes must implement.

Example usage:
    rule = AlmostEqualLogistic(type=RecommenderRuleTypes.AlmostEqualLogistic, temperature=0.5)
    move = rule(tree, random_generator)
"""

import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Protocol

import chipiron.players.move_selector.treevalue.trees as trees
from chipiron.environments.chess.move.imove import moveKey
from chipiron.players.boardevaluators.basic_evaluation import value_base
from chipiron.players.move_selector.treevalue.nodes.algorithm_node.algorithm_node import (
    AlgorithmNode,
)
from chipiron.players.move_selector.treevalue.nodes.itree_node import ITreeNode
from chipiron.players.move_selector.treevalue.nodes.utils import is_winning
from chipiron.utils.logger import chipiron_logger
from chipiron.utils.small_tools import softmax


class RecommenderRuleTypes(str, Enum):
    """
    Enum class that defines the available recommender rule types.
    """

    AlmostEqualLogistic = "almost_equal_logistic"
    Softmax = "softmax"


# theses are functions but i still use dataclasses instead
# of partial to be able to easily construct from yaml files using dacite


@dataclass
class AlmostEqualLogistic:
    """
    Recommender rule that selects the best move allowing for random choice for almost equally valued moves.
    """

    type: Literal[RecommenderRuleTypes.AlmostEqualLogistic]
    temperature: float

    def __call__(
        self, tree: trees.MoveAndValueTree, random_generator: random.Random
    ) -> moveKey:
        """
        Selects the best move from the tree, allowing for random choice for almost equally valued moves.

        Args:
            tree (trees.MoveAndValueTree): The move and value tree.
            random_generator (random.Random): The random generator.

        Returns:
            chess.Move: The recommended chess move.
        """
        # TODO this should be given at construction but postponed for now because of dataclasses
        # find the best first move allowing for random choice for almost equally valued moves.
        best_root_moves: list[moveKey] = (
            tree.root_node.minmax_evaluation.get_all_of_the_best_moves(
                how_equal="almost_equal_logistic"
            )
        )
        chipiron_logger.info(f"We have as bests: {[best for best in best_root_moves]}")
        best_move = random_generator.choice(best_root_moves)

        # assert isinstance(best_move, IMove)
        return best_move


@dataclass
class SoftmaxRule:
    """
    Recommender rule that selects the best move using the softmax function.
    """

    type: Literal[RecommenderRuleTypes.Softmax]
    temperature: float

    def __call__(
        self, tree: trees.MoveAndValueTree, random_generator: random.Random
    ) -> moveKey:
        """
        Selects the best move from the tree using the softmax function.

        Args:
            tree (trees.MoveAndValueTree): The move and value tree.
            random_generator (random.Random): The random generator.

        Returns:
            chess.Move: The recommended chess move.
        """
        # todo maybe there is a way to code this withtout the assert using the childrensorted value? or smth else
        values = []
        for node in tree.root_node.moves_children.values():
            assert isinstance(node, AlgorithmNode)
            value = tree.root_node.minmax_evaluation.subjective_value_of(
                node.minmax_evaluation
            )
            values.append(value)

        softmax_ = list(softmax(values, self.temperature))
        print(
            "SOFTMAX",
            self.temperature,
            [i / sum(softmax_) for i in softmax_],
            sum([i / sum(softmax_) for i in softmax_]),
        )

        move_as_list = random_generator.choices(
            list(tree.root_node.moves_children.keys()), weights=softmax_, k=1
        )
        best_move: moveKey = move_as_list[0]
        return best_move


AllRecommendFunctionsArgs = AlmostEqualLogistic | SoftmaxRule


@dataclass
class RecommenderRule(Protocol):
    """
    Protocol that all recommender rule classes must implement.
    """

    type: RecommenderRuleTypes

    def __call__(
        self, tree: trees.MoveAndValueTree, random_generator: random.Random
    ) -> moveKey:
        """
        Selects the best move from the tree.

        Args:
            tree (trees.MoveAndValueTree): The move and value tree.
            random_generator (random.Random): The random generator.

        Returns:
            chess.Move: The recommended chess move.
        """
        ...


def recommend_move_after_exploration_generic(
    recommend_move_after_exploration: AllRecommendFunctionsArgs,
    tree: trees.MoveAndValueTree,
    random_generator: random.Random,
) -> moveKey:
    """
    Recommends a move after exploration based on a generic rule.

    Args:
        recommend_move_after_exploration (AllRecommendFunctionsArgs): The recommend_move_after_exploration function.
        tree (trees.MoveAndValueTree): The move and value tree.
        random_generator (random.Random): The random number generator.

    Returns:
        chess.Move: The recommended move.

    """

    # if the situation is winning, we ask to play the move that is the most likely
    # to end the game fast by capturing pieces if possible
    is_winning_situation: bool = is_winning(
        node_minmax_evaluation=tree.root_node.minmax_evaluation,
        color=tree.root_node.board.turn,
    )
    over: bool = tree.root_node.is_over()
    if is_winning_situation and not over:
        # value of pieces of the opponent before the move
        value_father: int = value_base(
            board=tree.root_node.board, color=not tree.root_node.board.turn
        )

        child: ITreeNode[Any] | None
        best_value: int | None = None
        best_move: moveKey | None = None
        for move, child in tree.root_node.moves_children.items():
            assert isinstance(child, AlgorithmNode)

            # value of pieces of the opponent after that move
            value_child: int = value_base(board=child.board, color=child.board.turn)
            value: int = value_father - value_child

            still_wining_after_move: bool = is_winning(
                node_minmax_evaluation=child.minmax_evaluation,
                color=tree.root_node.board.turn,
            )
            if still_wining_after_move and (best_value is None or best_value < value):
                best_value = value
                best_move = move
        assert best_value is not None
        if best_value > 0:
            assert best_move is not None
            return best_move

    # base case
    return recommend_move_after_exploration(
        tree=tree, random_generator=random_generator
    )

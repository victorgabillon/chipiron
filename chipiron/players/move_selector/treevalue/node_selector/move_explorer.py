"""
This module contains the MoveExplorer class and its subclasses.
MoveExplorer is responsible for exploring moves in a game tree.
"""

import random
from enum import Enum

from chipiron.environments.chess.move.imove import moveKey
from chipiron.players.move_selector.treevalue.node_selector.notations_and_statics import (
    zipf_picks_random,
)
from chipiron.players.move_selector.treevalue.nodes.algorithm_node import AlgorithmNode


class SamplingPriorities(str, Enum):
    """
    Enumeration class representing the sampling priorities for move exploration.

    Attributes:
        NO_PRIORITY (str): No priority for move sampling.
        PRIORITY_BEST (str): Priority for the best move.
        PRIORITY_TWO_BEST (str): Priority for the two best moves.
    """

    NO_PRIORITY = "no_priority"
    PRIORITY_BEST = "priority_best"
    PRIORITY_TWO_BEST = "priority_two_best"


class MoveExplorer:
    """
    MoveExplorer is responsible for exploring moves in a game tree.
    It provides a method to sample a child node to explore.
    """

    priority_sampling: SamplingPriorities

    def __init__(self, priority_sampling: SamplingPriorities):
        """
        Initializes a MoveExplorer instance.

        Args:
            priority_sampling (SamplingPriorities): The priority sampling strategy to use.
        """
        self.priority_sampling = priority_sampling


class ZipfMoveExplorer(MoveExplorer):
    """
    ZipfMoveExplorer is a subclass of MoveExplorer that uses the Zipf distribution for sampling.
    """

    def __init__(
        self, priority_sampling: SamplingPriorities, random_generator: random.Random
    ) -> None:
        """
        Initializes a ZipfMoveExplorer instance.

        Args:
            priority_sampling (SamplingPriorities): The priority sampling strategy to use.
            random_generator (random.Random): The random number generator to use.
        """
        super().__init__(priority_sampling)
        self.random_generator = random_generator

    def sample_move_to_explore(
        self, tree_node_to_sample_from: AlgorithmNode
    ) -> moveKey:
        """
        Samples a child node to explore from the given tree node.

        Args:
            tree_node_to_sample_from (AlgorithmNode): The tree node to sample from.

        Returns:
            AlgorithmNode: The sampled child node to explore.
        """
        sorted_not_over_moves: list[moveKey] = (
            tree_node_to_sample_from.minmax_evaluation.sort_moves_not_over()
        )

        move = zipf_picks_random(
            ordered_list_elements=sorted_not_over_moves,
            random_generator=self.random_generator,
        )
        return move

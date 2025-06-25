"""
This module contains the implementation of the RecurZipfBase class, which is a node selector for a move selector tree.

The RecurZipfBase class is responsible for selecting the next node to explore in a move selector tree based on the RecurZipf algorithm.

Classes:
- RecurZipfBase: The RecurZipfBase Node selector.

"""

import random
import typing
from dataclasses import dataclass
from typing import Literal

from chipiron.environments.chess.move.imove import moveKey
from chipiron.players.move_selector.treevalue import trees
from chipiron.players.move_selector.treevalue.node_selector.move_explorer import (
    ZipfMoveExplorer,
)
from chipiron.players.move_selector.treevalue.node_selector.node_selector_types import (
    NodeSelectorType,
)
from chipiron.players.move_selector.treevalue.node_selector.opening_instructions import (
    OpeningInstructions,
    OpeningInstructor,
    create_instructions_to_open_all_moves,
)
from chipiron.players.move_selector.treevalue.nodes.algorithm_node.algorithm_node import (
    AlgorithmNode,
)
from chipiron.players.move_selector.treevalue.nodes.utils import (
    best_node_sequence_from_node,
)

from ..move_explorer import SamplingPriorities

if typing.TYPE_CHECKING:
    import chipiron.players.move_selector.treevalue.tree_manager as tree_man


@dataclass
class RecurZipfBaseArgs:
    """
    Arguments for the RecurZipfBase node selector.

    Attributes:
        move_explorer_priority (SamplingPriorities): The priority for move exploration.
    """

    type: Literal[NodeSelectorType.RecurZipfBase]
    move_explorer_priority: SamplingPriorities


class RecurZipfBase:
    """The RecurZipfBase Node selector"""

    opening_instructor: OpeningInstructor

    def __init__(
        self,
        args: RecurZipfBaseArgs,
        random_generator: random.Random,
        opening_instructor: OpeningInstructor,
    ) -> None:
        """
        Initializes a new instance of the RecurZipfBase class.

        Args:
        - args (RecurZipfBaseArgs): The arguments for the RecurZipfBase node selector.
        - random_generator (random.Random): The random number generator.
        - opening_instructor (OpeningInstructor): The opening instructor.

        """
        self.opening_instructor = opening_instructor
        self.move_explorer = ZipfMoveExplorer(
            args.move_explorer_priority, random_generator
        )
        self.random_generator = random_generator

    def choose_node_and_move_to_open(
        self,
        tree: trees.MoveAndValueTree,
        latest_tree_expansions: "tree_man.TreeExpansions",
    ) -> OpeningInstructions:
        """
        Chooses the next node to explore and the move to open.

        Args:
        - tree (trees.MoveAndValueTree): The move selector tree.
        - latest_tree_expansions (tree_man.TreeExpansions): The latest tree expansions.

        Returns:
        - OpeningInstructions: The instructions for opening the selected move.

        """
        # todo maybe proportions and proportions can be valuesorted dict with smart updates

        opening_instructions: OpeningInstructions
        best_node_sequence = best_node_sequence_from_node(tree.root_node)
        if best_node_sequence:
            last_node_in_best_line = best_node_sequence[-1]
            assert isinstance(last_node_in_best_line, AlgorithmNode)
            if (
                last_node_in_best_line.board.is_attacked(
                    not last_node_in_best_line.tree_node.player_to_move
                )
                and not last_node_in_best_line.minmax_evaluation.is_over()
            ):
                # print('best line is underattacked')
                if self.random_generator.random() > 0.5:
                    # print('best line is underattacked and i do')
                    all_moves_to_open: list[moveKey] = (
                        self.opening_instructor.all_moves_to_open(
                            node_to_open=last_node_in_best_line.tree_node
                        )
                    )
                    opening_instructions = create_instructions_to_open_all_moves(
                        moves_to_play=all_moves_to_open,
                        node_to_open=last_node_in_best_line,
                    )
                    return opening_instructions

        wandering_node = tree.root_node

        while wandering_node.minmax_evaluation.moves_not_over:
            assert not wandering_node.is_over()
            move = self.move_explorer.sample_move_to_explore(
                tree_node_to_sample_from=wandering_node
            )
            next_node = wandering_node.moves_children[move]
            assert isinstance(next_node, AlgorithmNode)
            wandering_node = next_node

        all_moves_to_open = self.opening_instructor.all_moves_to_open(
            node_to_open=wandering_node.tree_node
        )
        opening_instructions = create_instructions_to_open_all_moves(
            moves_to_play=all_moves_to_open, node_to_open=wandering_node
        )

        return opening_instructions

    def __str__(self) -> str:
        """
        Returns a string representation of the RecurZipfBase node selector.

        Returns:
        - str: The string representation of the RecurZipfBase node selector.

        """
        return "RecurZipfBase"

"""
factory for sequool node selector
"""

import random
from dataclasses import dataclass
from functools import partial
from typing import Literal

import chipiron.players.move_selector.treevalue.trees as trees
from chipiron.players.move_selector.treevalue.node_selector.node_selector_types import (
    NodeSelectorType,
)
from chipiron.players.move_selector.treevalue.node_selector.opening_instructions import (
    OpeningInstructor,
)

from .sequool import (
    ConsiderNodesFromHalfMoves,
    HalfMoveSelector,
    RandomAllSelector,
    Sequool,
    StaticNotOpenedSelector,
    consider_nodes_from_all_lesser_half_moves_in_descendants,
    consider_nodes_from_all_lesser_half_moves_in_sub_stree,
    consider_nodes_only_from_half_moves_in_descendants,
)


@dataclass
class SequoolArgs:
    """
    Arguments for the Sequool node selector.

    Attributes:
        recursive_selection_on_all_nodes (bool): Flag indicating whether to perform recursive selection on all nodes.
        random_depth_pick (bool): Flag indicating whether to randomly pick a depth for selection.
        consider_all_lesser_half_move (bool): Flag indicating whether to consider all lesser half moves.
    """

    type: Literal[NodeSelectorType.Sequool]
    recursive_selection_on_all_nodes: bool
    random_depth_pick: bool
    consider_all_lesser_half_move: bool


def create_sequool(
    opening_instructor: OpeningInstructor,
    args: SequoolArgs,
    random_generator: random.Random,
) -> Sequool:
    """
    Create a sequool node selector object.

    Args:
        opening_instructor: An opening instructor object.
        args: Dictionary of arguments.
        random_generator: Random generator object.

    Returns:
        A sequool node selector object.

    """
    all_nodes_not_opened = trees.Descendants()
    half_move_selector: HalfMoveSelector
    if args.recursive_selection_on_all_nodes:
        half_move_selector = RandomAllSelector()
    else:
        half_move_selector = StaticNotOpenedSelector(
            all_nodes_not_opened=all_nodes_not_opened
        )

    consider_nodes_from_half_moves: ConsiderNodesFromHalfMoves
    if args.recursive_selection_on_all_nodes:
        consider_nodes_from_half_moves = (
            consider_nodes_from_all_lesser_half_moves_in_sub_stree
        )
    else:
        if args.consider_all_lesser_half_move:
            consider_nodes_from_all_lesser_half_moves = partial(
                consider_nodes_from_all_lesser_half_moves_in_descendants,
                descendants=all_nodes_not_opened,
            )
            consider_nodes_from_half_moves = consider_nodes_from_all_lesser_half_moves
        else:
            consider_nodes_only_from_half_moves = partial(
                consider_nodes_only_from_half_moves_in_descendants,
                descendants=all_nodes_not_opened,
            )
            consider_nodes_from_half_moves = consider_nodes_only_from_half_moves

    sequool: Sequool = Sequool(
        opening_instructor=opening_instructor,
        all_nodes_not_opened=all_nodes_not_opened,
        recursif=args.recursive_selection_on_all_nodes,
        half_move_selector=half_move_selector,
        random_depth_pick=args.random_depth_pick,
        random_generator=random_generator,
        consider_nodes_from_half_moves=consider_nodes_from_half_moves,
    )

    return sequool

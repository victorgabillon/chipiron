"""
factory for sequool node selector
"""
import random

import chipiron.players.move_selector.treevalue.trees as trees
from .sequool2 import Sequool, StaticNotOpenedSelector, ExplorationNodeSelector
from chipiron.players.move_selector.treevalue.node_selector.opening_instructions import OpeningInstructor
from dataclasses import dataclass
from ..node_selector_args import NodeSelectorArgs


@dataclass
class SequoolArgs(NodeSelectorArgs):
    recursive_selection_on_all_nodes: bool
    random_depth_pick: bool


def create_sequool(
        opening_instructor: OpeningInstructor,
        args: SequoolArgs,
        random_generator: random.Random
) -> Sequool:
    """

    Args:
        opening_instructor: an opening instructor object
        args: dictionary of arguments

    Returns: a sequool node selector object

    """
    all_nodes_not_opened = trees.Descendants()
    node_candidates_selector: ExplorationNodeSelector
    if args.recursive_selection_on_all_nodes:
        node_candidates_selector
    else:
        node_candidates_selector: ExplorationNodeSelector = StaticNotOpenedSelector(
            all_nodes_not_opened=all_nodes_not_opened
        )

    sequool: Sequool = Sequool(
        opening_instructor=opening_instructor,
        all_nodes_not_opened=all_nodes_not_opened,
        recursif=args.recursive_selection_on_all_nodes,
        node_candidates_selector=node_candidates_selector,
        random_depth_pick=args.random_depth_pick,
        random_generator=random_generator
    )

    return sequool

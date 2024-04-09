"""
stopping criterion
"""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from . import node_selector as node_sel
from .trees import MoveAndValueTree


@runtime_checkable
class DepthToExpendP(Protocol):

    def get_current_depth_to_expand(self) -> int:
        ...


class StoppingCriterionTypes(str, Enum):
    DepthLimit: str = 'depth_limit'
    TreeMoveLimit: str = 'tree_move_limit'


@dataclass
class StoppingCriterionArgs:
    type: StoppingCriterionTypes


class StoppingCriterion:
    """
    The general stopping criterion
    """

    def should_we_continue(
            self,
            tree: MoveAndValueTree
    ) -> bool:
        """
        Asking should we continue

        Returns:
            boolean of should we continue
        """
        if tree.is_over():
            return False
        return True

    def respectful_opening_instructions(
            self,
            opening_instructions: node_sel.OpeningInstructions,
            tree: MoveAndValueTree
    ) -> node_sel.OpeningInstructions:
        """
        Ensures the opening request do not exceed the stopping criterion


        """
        return opening_instructions

    def get_string_of_progress(
            self,
            tree: MoveAndValueTree
    ) -> str:
        return ''


@dataclass
class TreeMoveLimitArgs(StoppingCriterionArgs):
    tree_move_limit: int


class TreeMoveLimit(StoppingCriterion):
    """
    The stopping criterion based on a tree move limit
    """
    tree_move_limit: int

    def __init__(
            self,
            tree_move_limit: int
    ) -> None:
        self.tree_move_limit = tree_move_limit

    def should_we_continue(
            self,
            tree: MoveAndValueTree
    ) -> bool:
        continue_base = super().should_we_continue(tree=tree)
        if not continue_base:
            return continue_base
        return tree.move_count < self.tree_move_limit

    def get_string_of_progress(
            self,
            tree: MoveAndValueTree
    ) -> str:
        """
        compute the string that display the progress in the terminal
        Returns:
            a string that display the progress in the terminal
        """
        return f'========= tree move counting: {tree.move_count} out of {self.tree_move_limit}' \
               f' |  {tree.move_count / self.tree_move_limit:.0%}'

    def respectful_opening_instructions(
            self,
            opening_instructions: node_sel.OpeningInstructions,
            tree: MoveAndValueTree
    ) -> node_sel.OpeningInstructions:
        """
        Ensures the opening request do not exceed the stopping criterion


        """
        opening_instructions_subset: node_sel.OpeningInstructions = node_sel.OpeningInstructions()
        opening_instructions.pop_items(
            popped=opening_instructions_subset,
            how_many=self.tree_move_limit - tree.move_count
        )
        return opening_instructions_subset


@dataclass
class DepthLimitArgs(StoppingCriterionArgs):
    depth_limit: int


class DepthLimit(StoppingCriterion):
    """
    The stopping criterion based on a depth limit
    """

    depth_limit: int
    node_selector: DepthToExpendP

    def __init__(
            self,
            depth_limit: int,
            node_selector: DepthToExpendP
    ) -> None:
        self.depth_limit = depth_limit
        self.node_selector = node_selector

    def should_we_continue(
            self,
            tree: MoveAndValueTree
    ) -> bool:
        continue_base = super().should_we_continue(tree=tree)
        if not continue_base:
            return continue_base
        return self.node_selector.get_current_depth_to_expand() < self.depth_limit

    def get_string_of_progress(self, tree: MoveAndValueTree) -> str:
        """
        compute the string that display the progress in the terminal
        Returns:
            a string that display the progress in the terminal
        """
        return '========= tree move counting: ' + str(tree.move_count) + ' | Depth: ' + str(
            self.node_selector.get_current_depth_to_expand()) + ' out of ' + str(self.depth_limit)


AllStoppingCriterionArgs = TreeMoveLimitArgs | DepthLimitArgs


def create_stopping_criterion(
        args: AllStoppingCriterionArgs,
        node_selector: node_sel.NodeSelector
) -> StoppingCriterion:
    """
    creating the stopping criterion
    Args:
        args:
        node_selector:

    Returns:
        A stopping criterion

    """
    stopping_criterion: StoppingCriterion

    match args.type:
        case StoppingCriterionTypes.DepthLimit:
            assert (isinstance(node_selector, DepthToExpendP))
            assert isinstance(args, DepthLimitArgs)
            stopping_criterion = DepthLimit(
                depth_limit=args.depth_limit,
                node_selector=node_selector
            )
        case StoppingCriterionTypes.TreeMoveLimit:
            assert isinstance(args, TreeMoveLimitArgs)

            stopping_criterion = TreeMoveLimit(tree_move_limit=args.tree_move_limit)
        case other:
            raise ValueError(f'stopping criterion builder: can not find {other} in file {__name__}')

    return stopping_criterion

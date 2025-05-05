"""
This module defines stopping criteria for a move selector in a game tree.

The stopping criteria determine when the move selector should stop exploring the game tree and make a decision.

The module includes the following classes:

- StoppingCriterion: The general stopping criterion class.
- TreeMoveLimit: A stopping criterion based on a tree move limit.
- DepthLimit: A stopping criterion based on a depth limit.

It also includes helper classes and functions for creating and managing stopping criteria.
"""

import abc
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Literal, Protocol, runtime_checkable

from chipiron.players.move_selector.treevalue import node_selector as node_sel
from chipiron.players.move_selector.treevalue.trees import MoveAndValueTree


@runtime_checkable
class DepthToExpendP(Protocol):
    """
    Protocol for objects that provide the current depth to expand.

    This protocol defines a single method `get_current_depth_to_expand` that should be implemented by classes
    that want to provide the current depth to expand.

    Attributes:
        None

    Methods:
        get_current_depth_to_expand: Returns the current depth to expand as an integer.

    Examples:
        >>> class MyDepthToExpend(DepthToExpendP):
        ...     def get_current_depth_to_expand(self) -> int:
        ...         return 5
        ...
        >>> obj = MyDepthToExpend()
        >>> obj.get_current_depth_to_expand()
        5
    """

    def get_current_depth_to_expand(self) -> int:
        """
        Returns the current depth to expand as an integer.

        Returns:
            The current depth to expand.

        Raises:
            None
        """
        ...


class StoppingCriterionTypes(str, Enum):
    """
    Enum class representing different types of stopping criteria for tree value calculation.
    """

    DepthLimit = "depth_limit"
    TreeMoveLimit = "tree_move_limit"


@dataclass
class StoppingCriterionArgs:
    """
    Represents the arguments for a stopping criterion.

    Attributes:
        type (StoppingCriterionTypes): The type of stopping criterion.
    """

    type: StoppingCriterionTypes


class ProgressMonitorP(Protocol):
    """
    The general stopping criterion Protocol
    """

    def should_we_continue(self, tree: MoveAndValueTree) -> bool:
        """
        Asking should we continue

        Returns:
            boolean of should we continue
        """
        ...

    def respectful_opening_instructions(
        self, opening_instructions: node_sel.OpeningInstructions, tree: MoveAndValueTree
    ) -> node_sel.OpeningInstructions:
        """
        Ensures the opening request do not exceed the stopping criterion


        """
        ...

    def get_string_of_progress(self, tree: MoveAndValueTree) -> str:
        """
        Returns a string representation of the progress made by the stopping criterion.

        Args:
            tree (MoveAndValueTree): The move and value tree.

        Returns:
            str: A string representation of the progress.
        """
        ...

    def get_percent_of_progress(
        self, tree: MoveAndValueTree, notify_function: Callable[[int], None] | None
    ) -> str: ...


class ProgressMonitor:
    """
    The general stopping criterion base class
    """

    def should_we_continue(self, tree: MoveAndValueTree) -> bool:
        """
        Asking should we continue

        Returns:
            boolean of should we continue
        """
        if tree.is_over():
            return False
        return True

    def respectful_opening_instructions(
        self, opening_instructions: node_sel.OpeningInstructions, tree: MoveAndValueTree
    ) -> node_sel.OpeningInstructions:
        """
        Ensures the opening request do not exceed the stopping criterion


        """
        return opening_instructions

    def get_string_of_progress(self, tree: MoveAndValueTree) -> str:
        """
        Returns a string representation of the progress made by the stopping criterion.

        Args:
            tree (MoveAndValueTree): The move and value tree.

        Returns:
            str: A string representation of the progress.
        """
        return ""

    @abc.abstractmethod
    def get_percent_of_progress(self, tree: MoveAndValueTree) -> int: ...

    def notify_percent_progress(
        self,
        tree: MoveAndValueTree,
        notify_percent_function: Callable[[int], None] | None,
    ) -> None:
        percent_progress: int = self.get_percent_of_progress(tree=tree)

        if notify_percent_function is not None:
            notify_percent_function(percent_progress)


@dataclass
class TreeMoveLimitArgs:
    """Arguments for the tree move limit stopping criterion."""

    type: Literal[StoppingCriterionTypes.TreeMoveLimit]
    tree_move_limit: int


class TreeMoveLimit(ProgressMonitor):
    """
    The stopping criterion based on a tree move limit
    """

    tree_move_limit: int

    def __init__(self, tree_move_limit: int) -> None:
        self.tree_move_limit = tree_move_limit

    def should_we_continue(self, tree: MoveAndValueTree) -> bool:
        continue_base: bool = super().should_we_continue(tree=tree)

        should_we: bool
        if not continue_base:
            should_we = continue_base
        else:
            should_we = tree.move_count < self.tree_move_limit
        return should_we

    def respectful_opening_instructions(
        self, opening_instructions: node_sel.OpeningInstructions, tree: MoveAndValueTree
    ) -> node_sel.OpeningInstructions:
        """
        Ensures the opening request do not exceed the stopping criterion


        """
        opening_instructions_subset: node_sel.OpeningInstructions = (
            node_sel.OpeningInstructions()
        )
        opening_instructions.pop_items(
            popped=opening_instructions_subset,
            how_many=self.tree_move_limit - tree.move_count,
        )
        return opening_instructions_subset

    def get_string_of_progress(self, tree: MoveAndValueTree) -> str:
        """
        compute the string that display the progress in the terminal

        Returns:
            a string that display the progress in the terminal
        """
        return (
            f"========= tree move counting: {tree.move_count} out of {self.tree_move_limit}"
            f" |  {tree.move_count / self.tree_move_limit:.0%}"
        )

    def get_percent_of_progress(
        self,
        tree: MoveAndValueTree,
    ) -> int:
        percent: int = int(tree.move_count / self.tree_move_limit * 100)
        return percent


@dataclass
class DepthLimitArgs:
    """
    Arguments for the depth limit stopping criterion.

    Attributes:
        depth_limit (int): The maximum depth allowed for the search.
    """

    type: Literal[StoppingCriterionTypes.DepthLimit]
    depth_limit: int


class DepthLimit(ProgressMonitor):
    """
    The stopping criterion based on a depth limit
    """

    depth_limit: int
    node_selector: DepthToExpendP

    def __init__(self, depth_limit: int, node_selector: DepthToExpendP) -> None:
        """
        Initializes a StoppingCriterion object.

        Args:
            depth_limit (int): The maximum depth to search in the tree.
            node_selector (DepthToExpendP): The node selector used to determine which nodes to expand.

        Returns:
            None
        """
        self.depth_limit = depth_limit
        self.node_selector = node_selector

    def should_we_continue(self, tree: MoveAndValueTree) -> bool:
        """
        Determines whether the search should continue expanding nodes in the tree.

        Args:
            tree (MoveAndValueTree): The tree containing the moves and their corresponding values.

        Returns:
            bool: True if the search should continue, False otherwise.
        """
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
        return (
            "========= tree move counting: "
            + str(tree.move_count)
            + " | Depth: "
            + str(self.node_selector.get_current_depth_to_expand())
            + " out of "
            + str(self.depth_limit)
        )

    def get_percent_of_progress(
        self,
        tree: MoveAndValueTree,
    ) -> int:
        # todo this percent is not precise
        percent: int = int(
            self.node_selector.get_current_depth_to_expand() / self.depth_limit * 100
        )
        return percent


AllStoppingCriterionArgs = TreeMoveLimitArgs | DepthLimitArgs


def create_stopping_criterion(
    args: AllStoppingCriterionArgs, node_selector: node_sel.NodeSelector
) -> ProgressMonitor:
    """
    creating the stopping criterion

    Args:
        args:
        node_selector:

    Returns:
        A stopping criterion

    """
    stopping_criterion: ProgressMonitor

    match args.type:
        case StoppingCriterionTypes.DepthLimit:
            assert isinstance(node_selector, DepthToExpendP)
            assert isinstance(args, DepthLimitArgs)
            stopping_criterion = DepthLimit(
                depth_limit=args.depth_limit, node_selector=node_selector
            )
        case StoppingCriterionTypes.TreeMoveLimit:
            assert isinstance(args, TreeMoveLimitArgs)

            stopping_criterion = TreeMoveLimit(tree_move_limit=args.tree_move_limit)
        case other:
            raise ValueError(
                f"stopping criterion builder: can not find {other} in file {__name__}"
            )

    return stopping_criterion

"""
Module that contains the classes for the exploration data of a tree node.
"""

import typing
from dataclasses import dataclass, field
from typing import Any

from chipiron.utils.small_tools import Interval

if typing.TYPE_CHECKING:
    from chipiron.players.move_selector.treevalue.nodes.itree_node import ITreeNode
    from chipiron.players.move_selector.treevalue.nodes.tree_node import TreeNode


@dataclass
class NodeExplorationData:
    """
    Represents the exploration data for a tree node.

    Attributes:
        tree_node (TreeNode): The tree node associated with the exploration data.
        index (float | None): The index value associated with the node. Defaults to None.

    Methods:
        dot_description(): Returns a string representation of the exploration data for dot visualization.
    """

    tree_node: "TreeNode[ITreeNode[Any]]"
    index: float | None = None

    def dot_description(self) -> str:
        """
        Returns a string representation of the dot description for the index.

        Returns:
            str: The dot description of the index.
        """
        return f"index:{self.index}"


@dataclass
class RecurZipfQuoolExplorationData(NodeExplorationData):
    """
    Represents the exploration data for a tree node with recursive zipf-quool factor.

    Attributes:
        zipf_factored_proba (float | None): The probability associated with the node, factored by zipf-quool factor.
            Defaults to None.

    Methods:
        dot_description(): Returns a string representation of the exploration data for dot visualization.
    """

    # the 'proba' associated by recursively multiplying 1/rank of the node with the max zipf_factor of the parents
    zipf_factored_proba: float | None = None

    def dot_description(self) -> str:
        """
        Returns a string representation of the index and zipf_factored_proba values.

        Returns:
            str: A string representation of the index and zipf_factored_proba values.
        """
        return f"index:{self.index} zipf_factored_proba:{self.zipf_factored_proba}"


@dataclass
class MinMaxPathValue(NodeExplorationData):
    """
    Represents the exploration data for a tree node with minimum and maximum path values.

    Attributes:
        min_path_value (float | None): The minimum path value associated with the node. Defaults to None.
        max_path_value (float | None): The maximum path value associated with the node. Defaults to None.

    Methods:
        dot_description(): Returns a string representation of the exploration data for dot visualization.
    """

    min_path_value: float | None = None
    max_path_value: float | None = None

    def dot_description(self) -> str:
        return f"min_path_value: {self.min_path_value}, max_path_value: {self.max_path_value}"


@dataclass
class IntervalExplo(NodeExplorationData):
    """
    Represents the exploration data for a tree node with an interval.

    Attributes:
        interval (Interval | None): The interval associated with the node. Defaults to None.

    Methods:
        dot_description(): Returns a string representation of the exploration data for dot visualization.
    """

    interval: Interval | None = field(default_factory=Interval)

    def dot_description(self) -> str:
        """
        Returns a string representation of the interval values.

        If the interval is None, returns 'None'.
        Otherwise, returns a string in the format 'min_interval_value: {min_value}, max_interval_value: {max_value}'.

        Returns:
            str: A string representation of the interval values.
        """
        if self.interval is None:
            return "None"
        else:
            return f"min_interval_value: {self.interval.min_value}, max_interval_value: {self.interval.max_value}"


@dataclass
class MaxDepthDescendants(NodeExplorationData):
    """
    Represents the exploration data for a tree node with maximum depth of descendants.
    """

    max_depth_descendants: int = 0

    def update_from_child(self, child_max_depth_descendants: int) -> bool:
        """
        Updates the max_depth_descendants value based on the child's max_depth_descendants.

        Args:
            child_max_depth_descendants (int): The max_depth_descendants value of the child node.

        Returns:
            bool: True if the max_depth_descendants value has changed, False otherwise.
        """
        previous_index = self.max_depth_descendants
        new_index: int = max(
            self.max_depth_descendants, child_max_depth_descendants + 1
        )
        self.max_depth_descendants = new_index
        has_index_changed: bool = new_index != previous_index

        return has_index_changed

    def dot_description(self) -> str:
        """
        Returns a string representation of the dot description for the node indices.

        Returns:
            str: The dot description for the node indices.
        """
        return f"max_depth_descendants: {self.max_depth_descendants}"

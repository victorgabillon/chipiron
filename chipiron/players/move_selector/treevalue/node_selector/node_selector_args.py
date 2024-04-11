"""
This module defines the NodeSelectorArgs class, which represents the arguments for a node selector.

The NodeSelectorArgs class is a dataclass that contains a single attribute:
- type: The type of the node selector, represented by the NodeSelectorType enum.

Example usage:
    args = NodeSelectorArgs(type=NodeSelectorType.BEST)
"""

from dataclasses import dataclass

from .node_selector_types import NodeSelectorType


@dataclass
class NodeSelectorArgs:
    """
    Represents the arguments for a node selector.
    """

    type: NodeSelectorType

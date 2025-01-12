"""
This module provides classes and functions for selecting nodes in a tree structure.

The module includes the following components:
- `create`: A factory function for creating node selectors.
- `AllNodeSelectorArgs`: A class that represents all possible arguments for node selectors.
- `NodeSelector`: A class that represents a node selector.
- `NodeSelectorArgs`: A class that represents the arguments for a node selector.
- `NodeSelectorType`: An enumeration of different types of node selectors.
- `OpeningInstructions`: A class that represents opening instructions for node selectors.
- `OpeningInstruction`: A class that represents an opening instruction for node selectors.
- `OpeningType`: An enumeration of different types of opening instructions.

To use this module, import it and use the provided classes and functions as needed.
"""

from .factory import AllNodeSelectorArgs, create
from .node_selector import NodeSelector
from .node_selector_args import NodeSelectorArgs
from .node_selector_types import NodeSelectorType
from .opening_instructions import OpeningInstruction, OpeningInstructions, OpeningType

__all__ = [
    "OpeningInstructions",
    "OpeningInstruction",
    "AllNodeSelectorArgs",
    "OpeningType",
    "NodeSelector",
    "create",
    "NodeSelectorArgs",
    "NodeSelectorType",
]

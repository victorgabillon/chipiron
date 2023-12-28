from dataclasses import dataclass
from .node_selector_types import NodeSelectorType

@dataclass
class NodeSelectorArgs:
    type: NodeSelectorType

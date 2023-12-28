from .all_node_evaluators import NodeEvaluatorTypes
from dataclasses import dataclass


@dataclass
class NodeEvaluatorArgs:
    type: NodeEvaluatorTypes

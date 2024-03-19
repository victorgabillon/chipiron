from dataclasses import dataclass

from .all_node_evaluators import NodeEvaluatorTypes


@dataclass
class NodeEvaluatorArgs:
    type: NodeEvaluatorTypes
    syzygy_evaluation: bool
    representation: str  # this should maybe be optional and involves creates wrapper? but i am lazy at the moment

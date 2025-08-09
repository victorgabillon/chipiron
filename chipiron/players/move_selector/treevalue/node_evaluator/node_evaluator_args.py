"""
This module defines the `NodeEvaluatorArgs` class, which represents the arguments for a node evaluator.

Classes:
- NodeEvaluatorArgs: Represents the arguments for a node evaluator.

"""

from dataclasses import dataclass
from typing import Literal

from chipiron.players.boardevaluators.evaluation_scale import EvaluationScale
from chipiron.players.boardevaluators.neural_networks.input_converters.ModelInputRepresentationType import (
    InternalTensorRepresentationType,
)

from .all_node_evaluators import NodeEvaluatorTypes


@dataclass
class NodeEvaluatorArgs:
    """
    Represents the arguments for a node evaluator.

    Attributes:
    - type: The type of the node evaluator.
    - syzygy_evaluation: A boolean indicating whether syzygy evaluation is enabled.
    - representation: The representation of the node evaluator.

    """

    # Whether to use syzygy table for evaluation.
    syzygy_evaluation: bool

    # The internal representation type used by the node evaluator.
    internal_representation_type: InternalTensorRepresentationType

    # The evaluation scale used by the node evaluator. (Default values when nodes are found to be over)
    evaluation_scale: EvaluationScale


@dataclass
class BasicEvaluationNodeEvaluatorArgs(NodeEvaluatorArgs):
    """
    Represents the arguments for a node evaluator.

    """

    type: Literal[NodeEvaluatorTypes.BasicEvaluation]

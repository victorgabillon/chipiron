"""
This module defines the `NodeEvaluatorArgs` class, which represents the arguments for a node evaluator.

Classes:
- NodeEvaluatorArgs: Represents the arguments for a node evaluator.

"""

from dataclasses import dataclass

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

    type: NodeEvaluatorTypes
    syzygy_evaluation: bool
    internal_representation_type: InternalTensorRepresentationType = (
        InternalTensorRepresentationType.NO
    )  # this should maybe be optional and involves creates wrapper? but i am lazy at the moment

"""
This module defines the `NodeEvaluatorArgs` class, which represents the arguments for a node evaluator.

Classes:
- NodeEvaluatorArgs: Represents the arguments for a node evaluator.

"""

from dataclasses import dataclass

from coral.neural_networks.input_converters.ModelInputRepresentationType import (
    InternalTensorRepresentationType,
)

from chipiron.players.boardevaluators import master_board_evaluator


@dataclass
class NodeEvaluatorArgs:
    """
    Represents the arguments for a node evaluator.

    Attributes:
    - type: The type of the node evaluator.
    - syzygy_evaluation: A boolean indicating whether syzygy evaluation is enabled.
    - representation: The representation of the node evaluator.

    """

    # The internal representation type used by the node evaluator.
    internal_representation_type: InternalTensorRepresentationType

    # The arguments for the master board evaluator.
    master_board_evaluator: master_board_evaluator.MasterBoardEvaluatorArgs

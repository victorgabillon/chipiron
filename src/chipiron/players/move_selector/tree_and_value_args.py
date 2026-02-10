"""Tree-and-value move selector args with game-specific evaluator inputs."""

from dataclasses import dataclass
from typing import Literal

from anemone import TreeAndValuePlayerArgs as AnemoneTreeArgs

from chipiron.players.boardevaluators import master_board_evaluator
from chipiron.players.boardevaluators.neural_networks.input_converters.model_input_representation_type import (
    InternalTensorRepresentationType,
)
from chipiron.players.move_selector.move_selector_types import MoveSelectorTypes

TREE_AND_VALUE_LITERAL_STRING: Literal["TreeAndValue"] = "TreeAndValue"


@dataclass
class NodeEvaluatorArgs:
    """Represents the arguments for a node evaluator.

    Attributes:
    - type: The type of the node evaluator.
    - syzygy_evaluation: A boolean indicating whether syzygy evaluation is enabled.
    - representation: The representation of the node evaluator.

    """

    # The internal representation type used by the node evaluator.
    internal_representation_type: InternalTensorRepresentationType

    # The arguments for the master board evaluator.
    master_board_evaluator: master_board_evaluator.MasterBoardEvaluatorArgs


@dataclass(frozen=True)
class TreeAndValueAppArgs:
    """Generic wrapper for tree-and-value settings plus evaluator args."""

    anemone_args: AnemoneTreeArgs
    evaluator_args: NodeEvaluatorArgs
    accelerate_when_winning: bool = False
    type: Literal[MoveSelectorTypes.TREE_AND_VALUE] = MoveSelectorTypes.TREE_AND_VALUE

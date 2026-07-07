"""Configuration dataclasses for supervised chess neural-network training."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from coral.board_evaluation import (
    PointOfView,
)
from coral.neural_networks.models.multi_layer_perceptron import (
    MultiLayerPerceptronArgs,
)
from coral.neural_networks.neural_net_architecture_args import (
    NeuralNetArchitectureArgs,
)
from coral.neural_networks.nn_model_type import (
    ActivationFunctionType,
    NNModelType,
)
from coral.neural_networks.output_converters.model_output_type import (
    ModelOutputType,
)

from chipiron.environments.types import GameKind
from chipiron.players.boardevaluators.neural_networks.input_converters.model_input_representation_type import (
    ModelInputRepresentationType,
)

if TYPE_CHECKING:
    from chipiron.utils import MyPath
else:
    MyPath = str


class SupervisedTrainingConfigError(ValueError):
    """Raised when supervised training configuration is inconsistent."""

    def __init__(
        self,
        reuse_existing_model: bool,
        nn_parameters_file_if_reusing_existing_one: MyPath | None,
    ) -> None:
        """Initialize the error with inconsistent trainer arguments."""
        msg = (
            "Problem because you are asking for a reuse of existing model without specifying a"
            f" param file as we have: reuse_existing_model {reuse_existing_model}"
            f" nn_param_file_if_not_reusing_existing_one {nn_parameters_file_if_reusing_existing_one}"
        )
        super().__init__(msg)


class OptimizerType(Enum):
    """Optimizer names supported by supervised chess training config."""

    SGD = "sgd"


@dataclass(frozen=True, slots=True)
class GameInputArgs:
    """Gameinputargs implementation."""

    game_kind: GameKind
    representation: ModelInputRepresentationType


@dataclass
class SupervisedTrainingArgs:
    """Arguments for supervised chess neural-network training.

    Attributes:
        reuse_existing_trainer (bool): Whether to reuse an existing trainer.
        starting_lr (float): The starting learning rate.
        momentum_op (float): The momentum value.
        scheduler_step_size (int): The step size for the scheduler.
        scheduler_gamma (float): The gamma value for the scheduler.
        saving_intermediate_copy (bool): Whether to save intermediate copies.

    """

    neural_network_architecture_args: NeuralNetArchitectureArgs = field(
        default_factory=lambda: NeuralNetArchitectureArgs(
            model_type_args=MultiLayerPerceptronArgs(
                type=NNModelType.MULTI_LAYER_PERCEPTRON,
                number_neurons_per_layer=[5, 1],
                list_of_activation_functions=[
                    ActivationFunctionType.TANGENT_HYPERBOLIC
                ],
            ),
            model_output_type=ModelOutputType(point_of_view=PointOfView.PLAYER_TO_MOVE),
        )
    )
    game_input: GameInputArgs = field(
        default_factory=lambda: GameInputArgs(
            game_kind=GameKind.CHESS,
            representation=ModelInputRepresentationType.PIECE_DIFFERENCE,
        )
    )
    nn_parameters_file_if_reusing_existing_one: MyPath | None = None
    specific_saving_folder: MyPath | None = None
    reuse_existing_model: bool = False
    reuse_existing_trainer: bool = False
    starting_lr: float = 0.1
    momentum_op: float = 0.9
    scheduler_step_size: int = 1
    scheduler_gamma: float = 0.5
    saving_intermediate_copy: bool = True

    batch_size_train: int = 32
    batch_size_test: int = 10
    saving_interval: int = 1000
    saving_intermediate_copy_interval: int = 10000
    min_interval_lr_change: int = 1000000
    min_lr: float = 0.001

    epochs_number: int = 100

    def __post_init__(self) -> None:
        """Run post-init."""
        if (
            self.reuse_existing_model
            and self.nn_parameters_file_if_reusing_existing_one is None
        ):
            raise SupervisedTrainingConfigError(
                self.reuse_existing_model,
                self.nn_parameters_file_if_reusing_existing_one,
            )


# Compatibility aliases kept for older imports/YAML migration.
# Prefer SupervisedTrainingArgs and SupervisedTrainingConfigError in new code.
NNTrainerArgs = SupervisedTrainingArgs
NNTrainerConfigError = SupervisedTrainingConfigError

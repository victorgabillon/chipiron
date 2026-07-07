"""Module to create and save neural network trainers and their parameters."""

import pickle
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, cast, no_type_check

import torch
from coral.board_evaluation import (
    PointOfView,
)
from coral.chi_nn import ChiNN
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
from torch import optim

from chipiron.environments.types import GameKind
from chipiron.learningprocesses.nn_trainer.checkpoint_helpers import (
    get_folder_training_copies_path_from,
    get_optimizer_file_path_from,
    get_scheduler_file_path_from,
)
from chipiron.learningprocesses.nn_trainer.checkpoint_helpers import (
    safe_nn_architecture_save as safe_nn_architecture_save,
)
from chipiron.learningprocesses.nn_trainer.checkpoint_helpers import (
    safe_nn_param_save as safe_nn_param_save,
)
from chipiron.learningprocesses.nn_trainer.checkpoint_helpers import (
    safe_nn_trainer_save as safe_nn_trainer_save,
)
from chipiron.learningprocesses.nn_trainer.nn_trainer import NNPytorchTrainer
from chipiron.players.boardevaluators.neural_networks.input_converters.model_input_representation_type import (
    ModelInputRepresentationType,
)
from chipiron.utils import MyPath
from chipiron.utils.small_tools import mkdir_if_not_existing

if TYPE_CHECKING:
    from collections.abc import Iterable


class NNTrainerConfigError(ValueError):
    """Raised when NN trainer configuration is inconsistent."""

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


SerializableType = (
    str
    | int
    | float
    | bool
    | None
    | dict[str, Any]
    | list[Any]
    | set[Any]
    | frozenset[Any]
)


@dataclass(frozen=True, slots=True)
class GameInputArgs:
    """Gameinputargs implementation."""

    game_kind: GameKind
    representation: ModelInputRepresentationType


@dataclass
class NNTrainerArgs:
    """Arguments for the NNTrainer class.

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
            raise NNTrainerConfigError(
                self.reuse_existing_model,
                self.nn_parameters_file_if_reusing_existing_one,
            )


def create_nn_trainer(
    args: NNTrainerArgs, nn: ChiNN, saving_folder: MyPath
) -> NNPytorchTrainer:
    """Create an instance of NNPytorchTrainer based on the provided arguments and neural network.

    Args:
        args (NNTrainerArgs): The arguments for the NNTrainer.
        nn (ChiNN): The neural network to be trained.

    Returns:
        NNPytorchTrainer: An instance of NNPytorchTrainer.

    """
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    if args.reuse_existing_trainer:
        file_optimizer_path = get_optimizer_file_path_from(folder_path=saving_folder)
        with open(file_optimizer_path, "rb") as file_optimizer:
            optimizer = pickle.load(file_optimizer)

        file_scheduler_path = get_scheduler_file_path_from(folder_path=saving_folder)
        with open(file_scheduler_path, "rb") as file_scheduler:
            scheduler = pickle.load(file_scheduler)

    else:
        optimizer = optim.SGD(
            nn.parameters(),
            lr=args.starting_lr,
            momentum=args.momentum_op,
            weight_decay=0.000,
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma
        )

    if args.saving_intermediate_copy:
        folder_path_training_copies = get_folder_training_copies_path_from(
            saving_folder
        )
        mkdir_if_not_existing(folder_path_training_copies)

    return NNPytorchTrainer(net=nn, optimizer=optimizer, scheduler=scheduler)


@no_type_check
def serialize_for_yaml(obj: Any) -> SerializableType:
    """Recursively converts Enums and other non-serializable objects.

    into basic types for safe YAML dumping.
    """
    # Handle None and primitives
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return cast("SerializableType", obj)

    # Handle Enums
    if isinstance(obj, Enum):
        return serialize_for_yaml(obj.value)

    # Handle dictionaries
    if isinstance(obj, dict):
        dict_obj = cast("dict[Any, Any]", obj)
        dict_result: dict[str, SerializableType] = {}
        for key, value in dict_obj.items():
            str_key: str = str(key)
            serialized_value: SerializableType = serialize_for_yaml(value)
            dict_result[str_key] = serialized_value
        return dict_result

    # Handle lists and tuples
    if isinstance(obj, (list, tuple)):
        iterable_obj: Iterable[Any] = cast("Iterable[Any]", obj)
        list_result: list[SerializableType] = [
            serialize_for_yaml(item) for item in iterable_obj
        ]
        return list_result

    # Handle frozensets
    if isinstance(obj, frozenset):
        frozenset_obj = cast("frozenset[Any]", obj)
        frozenset_result: frozenset[SerializableType] = frozenset(
            serialize_for_yaml(item) for item in frozenset_obj
        )
        return frozenset_result

    # Handle objects with __dict__
    if hasattr(obj, "__dict__"):
        return serialize_for_yaml(vars(obj))

    # Fallback: convert to string
    return str(obj)

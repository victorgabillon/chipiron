"""Module for test evaluate models."""

import dataclasses
import typing

from coral.neural_networks.factory import (
    NeuralNetModelsAndArchitecture,
)
from coral.neural_networks.neural_net_architecture_args import (
    NeuralNetArchitectureArgs,
)

from chipiron.scripts.evaluate_models.evaluate_models import evaluate_models
from chipiron.utils.small_tools import resolve_package_path

print(
    "model_type_args annotation:",
    NeuralNetArchitectureArgs.__annotations__["model_type_args"],
)
T = NeuralNetArchitectureArgs.__annotations__["model_type_args"]
print("is dataclass?", dataclasses.is_dataclass(T))
print("origin:", typing.get_origin(T), "args:", typing.get_args(T))

test_models_to_evaluate_: list[NeuralNetModelsAndArchitecture] = [
    NeuralNetModelsAndArchitecture.build_from_folder_path(
        folder_path=resolve_package_path(
        "package://scripts/learn_nn_supervised/board_evaluators_common_training_data/nn_pytorch/prelu_no_bug"
    ))
]


def test_evaluate_model() -> None:
    """Test evaluate model."""
    report_path = resolve_package_path(
        "package://scripts/evaluate_models/tests/test_evaluation_report.yaml"
    )
    print("report path:", report_path)

    with open(report_path, "w", encoding="utf-8"):
        pass

    m = NeuralNetModelsAndArchitecture.build_from_folder_path(
        folder_path=resolve_package_path("package://scripts/learn_nn_supervised/board_evaluators_common_training_data/nn_pytorch/prelu_no_bug")
    )
    print("ARCH ARGS TYPE:", type(m.nn_architecture_args))
    print("ARCH ARGS:", m.nn_architecture_args)
    print("WEIGHTS:", m.model_weights_file_name)

    evaluate_models(
        models_to_evaluate=test_models_to_evaluate_,
        evaluation_report_file=resolve_package_path("package://scripts/evaluate_models/tests/test_evaluation_report.yaml"),
        dataset_file_name=resolve_package_path("package://scripts/learn_nn_supervised/tests/small_dataset.pi"),
    )


if __name__ == "__main__":
    test_evaluate_model()

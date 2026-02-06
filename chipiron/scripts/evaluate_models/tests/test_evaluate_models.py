"""Module for test evaluate models."""

from coral.neural_networks.factory import (
    NeuralNetModelsAndArchitecture,
)

from chipiron.scripts.evaluate_models.evaluate_models import evaluate_models

from coral.neural_networks.neural_net_architecture_args import (
    NeuralNetArchitectureArgs,
)

import dataclasses
import typing
print("model_type_args annotation:", NeuralNetArchitectureArgs.__annotations__["model_type_args"])
T = NeuralNetArchitectureArgs.__annotations__["model_type_args"]
print("is dataclass?", dataclasses.is_dataclass(T))
print("origin:", typing.get_origin(T), "args:", typing.get_args(T))

test_models_to_evaluate_: list[NeuralNetModelsAndArchitecture] = [
    NeuralNetModelsAndArchitecture.build_from_folder_path(
        folder_path="chipiron/scripts/learn_nn_supervised/board_evaluators_common_training_data/nn_pytorch/prelu_no_bug"
    )
]


def test_evaluate_model() -> None:
    """Test evaluate model."""
    open(
        "chipiron/scripts/evaluate_models/tests/test_evaluation_report.yaml", "w"
    ).close()

    m = NeuralNetModelsAndArchitecture.build_from_folder_path(
        folder_path="chipiron/scripts/learn_nn_supervised/board_evaluators_common_training_data/nn_pytorch/prelu_no_bug"
    )
    print("ARCH ARGS TYPE:", type(m.nn_architecture_args))
    print("ARCH ARGS:", m.nn_architecture_args)
    print("WEIGHTS:", m.model_weights_file_name)

    evaluate_models(
        models_to_evaluate=test_models_to_evaluate_,
        evaluation_report_file="chipiron/scripts/evaluate_models/tests/test_evaluation_report.yaml",
        dataset_file_name="chipiron/scripts/learn_nn_supervised/tests/small_dataset.pi",
    )


if __name__ == "__main__":
    test_evaluate_model()

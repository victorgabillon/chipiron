from chipiron.players.boardevaluators.neural_networks.factory import (
    NeuralNetModelsAndArchitecture,
)
from chipiron.scripts.evaluate_models.evaluate_models import evaluate_models
from chipiron.utils import path

test_models_to_evaluate_: list[NeuralNetModelsAndArchitecture] = [
    NeuralNetModelsAndArchitecture.build_from_folder_path(
        folder_path="chipiron/scripts/learn_nn_supervised/board_evaluators_common_training_data/nn_pytorch/prelu_no_bug"
    )
]


def test_evaluate_model() -> None:
    open(
        "chipiron/scripts/evaluate_models/tests/test_evaluation_report.yaml", "w"
    ).close()
    evaluate_models(
        models_to_evaluate=test_models_to_evaluate_,
        evaluation_report_file="chipiron/scripts/evaluate_models/tests/test_evaluation_report.yaml",
        dataset_file_name="chipiron/scripts/learn_nn_supervised/tests/small_dataset.pi",
    )


if __name__ == "__main__":
    test_evaluate_model()

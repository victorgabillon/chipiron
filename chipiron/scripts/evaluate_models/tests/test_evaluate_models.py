from chipiron.scripts.evaluate_models.evaluate_models import evaluate_models
from chipiron.utils import path

test_folders_of_models_to_evaluate: list[path] = [
    "data/players/board_evaluators/nn_pytorch/nn_pp2d2_2_prelu/param_prelu",
]


def test_evaluate_model() -> None:
    open(
        "chipiron/scripts/evaluate_models/tests/test_evaluation_report.yaml", "w"
    ).close()
    evaluate_models(
        folders_of_models_to_evaluate=test_folders_of_models_to_evaluate,
        evaluation_report_file="chipiron/scripts/evaluate_models/tests/test_evaluation_report.yaml",
        dataset_file_name="chipiron/scripts/learn_nn_supervised/tests/small_dataset.pi",
    )


if __name__ == "__main__":
    test_evaluate_model()

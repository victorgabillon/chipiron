import datetime
import os
import time
from dataclasses import asdict, dataclass
from typing import Any

import dacite
import torch
import yaml
from torch.utils.data import DataLoader

from chipiron.learningprocesses.nn_trainer.nn_trainer import (
    compute_test_error_on_dataset,
)
from chipiron.players.boardevaluators.datasets.datasets import (
    FenAndValueDataSet,
    process_stockfish_value,
)
from chipiron.players.boardevaluators.neural_networks import NNBoardEvaluator
from chipiron.players.boardevaluators.neural_networks.factory import (
    NeuralNetModelsAndArchitecture,
    create_nn_board_eval_from_folder_path_and_existing_model,
    create_nn_board_eval_from_nn_parameters_file_and_existing_model,
    get_nn_param_file_path_from,
)
from chipiron.utils import path
from chipiron.utils.chi_nn import ChiNN


@dataclass
class ModelEvaluation:
    """
    Represents a model evaluation.
    """

    evaluation: float
    time_of_evaluation: datetime.datetime
    number_of_model_parameters: int


EvaluatedModels = dict[str, ModelEvaluation]


def compute_model_hask_key(model_and_archi: NeuralNetModelsAndArchitecture) -> str:
    return (
        str(model_and_archi.model_weights_file_name)
        + model_and_archi.nn_architecture_args.filename()
    )


def count_parameters(model: ChiNN) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_models(
    models_to_evaluate: list[NeuralNetModelsAndArchitecture],
    evaluation_report_file: path = "chipiron/scripts/evaluate_models/evaluation_report.yaml",
    dataset_file_name: path = "data/datasets/goodgames_plusvariation_stockfish_eval_test",
) -> None:
    """
    Evaluates the models in the list models_to_evaluate.

    Raises:
        Exception: If there is an error loading the evaluation report.
    """
    print("Evaluating models...")
    # Load the evaluation report
    evaluated_models: dict[str, ModelEvaluation] = {}
    with open(evaluation_report_file, "r") as stream:
        try:
            evaluated_models_temp: dict[str, dict[Any, Any]] | None
            evaluated_models_temp = yaml.safe_load(stream)
            if evaluated_models_temp is None:
                evaluated_models_temp = {}
            evaluated_model_path: path
            evaluated_model_evaluation_dict: dict[Any, Any]
            for (
                evaluated_model_path,
                evaluated_model_evaluation_dict,
            ) in evaluated_models_temp.items():
                model_evaluation_: ModelEvaluation = dacite.from_dict(
                    data_class=ModelEvaluation, data=evaluated_model_evaluation_dict
                )
                evaluated_models[evaluated_model_path] = model_evaluation_
            print(
                "Here is the content of the previous evaluation report: ",
                evaluated_models,
            )
        except yaml.YAMLError as exc:
            print(exc)
            raise Exception("Error loading the evaluation report")

    # Evaluate the models if necessary
    data_loader_stockfish_boards_test = None
    model_to_evaluate: NeuralNetModelsAndArchitecture
    for model_to_evaluate in models_to_evaluate:
        print(f"\nEvaluating model: {model_to_evaluate}")

        model_hash_key: str = compute_model_hask_key(model_to_evaluate)

        # Check if the model has already been evaluated
        should_evaluate: bool
        if model_hash_key in evaluated_models:
            model_evaluation: ModelEvaluation = evaluated_models[model_hash_key]
            print(
                "This model has already been evaluated. Now checking if the file has been modified since last evaluation..."
            )
            model_params_path_pt: path
            model_params_path_pt = model_to_evaluate.model_weights_file_name

            modification_time = os.path.getmtime(model_params_path_pt)
            readable_time = time.ctime(modification_time)
            print(
                f"Last modification time: {readable_time} and last evaluation time {model_evaluation.time_of_evaluation} "
            )
            # Check if the model has been modified since last evaluation
            if modification_time > model_evaluation.time_of_evaluation.timestamp():
                # The model has been modified since last evaluation
                print(
                    "The model has been modified since last evaluation. Asking for evaluation..."
                )
                should_evaluate = True
            else:
                # The model has not been modified since last evaluation
                print(
                    "The model has not been modified since last evaluation. No need to evaluate."
                )
                should_evaluate = False
        else:
            print("This model has not been evaluated yet. Asking for evaluation...")
            should_evaluate = True

        # Check if the model should be evaluated
        if should_evaluate:
            # Evaluate the model
            print("Evaluating model...")

            criterion = torch.nn.L1Loss()

            nn_board_evaluator: NNBoardEvaluator
            nn_board_evaluator = (
                create_nn_board_eval_from_nn_parameters_file_and_existing_model(
                    model_weights_file_name=model_to_evaluate.model_weights_file_name,
                    nn_architecture_args=model_to_evaluate.nn_architecture_args,
                )
            )

            stockfish_boards_test = FenAndValueDataSet(
                file_name=dataset_file_name,
                preprocessing=False,
                transform_board_function=nn_board_evaluator.board_to_input_convert,
                transform_dataset_value_to_white_value_function=process_stockfish_value,
                transform_white_value_to_model_output_function=nn_board_evaluator.output_and_value_converter.from_value_white_to_model_output,
            )

            stockfish_boards_test.load()

            data_loader_stockfish_boards_test = DataLoader(
                stockfish_boards_test,
                batch_size=10000,
                shuffle=False,
                num_workers=1,
            )
            print(f"Size of test set: {len(data_loader_stockfish_boards_test)}")

            eval = compute_test_error_on_dataset(
                net=nn_board_evaluator.net,
                criterion=criterion,
                data_test=data_loader_stockfish_boards_test,
                number_of_tests=len(data_loader_stockfish_boards_test),
            )
            number_of_model_parameters: int = count_parameters(nn_board_evaluator.net)
            model_evaluation = ModelEvaluation(
                evaluation=eval,
                time_of_evaluation=datetime.datetime.now(),
                number_of_model_parameters=number_of_model_parameters,
            )
            evaluated_models[model_hash_key] = model_evaluation
            print("Model evaluated!")

    evaluated_model_path_: path
    evaluated_model_evaluation: ModelEvaluation
    evaluated_models_final_dict: dict[path, dict[Any, Any]] = {}
    for evaluated_model_path_, evaluated_model_evaluation in evaluated_models.items():
        evaluated_models_final_dict[evaluated_model_path_] = asdict(
            evaluated_model_evaluation
        )
    with open(
        "chipiron/scripts/evaluate_models/evaluation_report.yaml", "w"
    ) as outfile:
        yaml.dump(evaluated_models_final_dict, outfile, default_flow_style=False)

    print("All evaluations done!")


if __name__ == "__main__":
    models_to_evaluate_: list[NeuralNetModelsAndArchitecture] = [
        NeuralNetModelsAndArchitecture.build_from_folder_path(
            folder_path="chipiron/scripts/learn_nn_supervised/board_evaluators_common_training_data/nn_pytorch/prelu_no_bug"
        ),
        NeuralNetModelsAndArchitecture.build_from_folder_path(
            folder_path="chipiron/scripts/learn_nn_supervised/board_evaluators_common_training_data/nn_pytorch/nn_p1_new"
        ),
        NeuralNetModelsAndArchitecture.build_from_folder_path(
            folder_path="chipiron/scripts/learn_nn_supervised/board_evaluators_common_training_data/nn_pytorch/nn_pp2_new"
        ),
        # NeuralNetModelsAndArchitecture.build_from_folder_path(
        #     folder_path="chipiron/scripts/learn_nn_supervised/board_evaluators_common_training_data/nn_pytorch/transformerone"),
    ]

    evaluate_models(models_to_evaluate=models_to_evaluate_)

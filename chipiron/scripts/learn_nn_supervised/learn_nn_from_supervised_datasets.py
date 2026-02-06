"""Describe script learns a neural network (NN) from a supervised dataset of board and evaluation pairs.

The script takes care of setting up the data loader from the evaluation files, creating the NN,
and running the learning process.

Usage:
- Instantiate the `LearnNNScript` class with a `base_script` object.
- Call the `run()` method to start the learning process.
- Call the `terminate()` method to finish the script.

Example:
    base_script = Script()
    learn_script = LearnNNScript(base_script)
    learn_script.run()
    learn_script.terminate()

"""

import copy
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import mlflow
import mlflow.pytorch
import torch
from coral.chi_nn import ChiNN
from coral.neural_networks import NNBWStateEvaluator
from coral.neural_networks.factory import (
    create_nn_state_eval_from_architecture_args,
    create_nn_state_eval_from_nn_parameters_file_and_existing_model,
)
from mlflow.models.signature import (
    ModelSignature,
    infer_signature,  # pyright: ignore[reportUnknownVariableType]
)
from torch.utils.data import DataLoader
from torchinfo import summary  # pyright: ignore[reportUnknownVariableType]

import chipiron.utils.path_variables
from chipiron.environments.chess.types import ChessState
from chipiron.learningprocesses.nn_trainer.factory import (
    NNTrainerArgs,
    create_nn_trainer,
    safe_nn_architecture_save,
    safe_nn_param_save,
    safe_nn_trainer_save,
)
from chipiron.players.boardevaluators.datasets.datasets import (
    DataSetArgs,
    FenAndValueData,
    FenAndValueDataSet,
    custom_collate_fn_fen_and_value,
    process_stockfish_value,  # pyright: ignore[reportUnknownVariableType]
)
from chipiron.players.boardevaluators.neural_networks.chipiron_nn_args import (
    ChipironNNArgs,
    create_content_to_input_convert,
    create_content_to_input_from_model_weights,
    save_chipiron_nn_args,
)
from chipiron.scripts.script import Script
from chipiron.scripts.script_args import BaseScriptArgs
from chipiron.utils import path
from chipiron.utils.logger import chipiron_logger

if TYPE_CHECKING:
    from chipiron.learningprocesses.nn_trainer.nn_trainer import NNPytorchTrainer

logging.basicConfig(level=logging.WARNING)


@dataclass
class LearnNNScriptArgs:
    """Represents the arguments for the LearnNNScript.

    Attributes:
        nn_trainer_args (NNTrainerArgs): The arguments for the NNTrainer.

    """

    base_script_args: BaseScriptArgs = field(default_factory=BaseScriptArgs)
    nn_trainer_args: NNTrainerArgs = field(default_factory=NNTrainerArgs)
    dataset_args: DataSetArgs = field(
        default_factory=lambda: DataSetArgs(
            train_file_name="data/datasets/goodgames_plusvariation_stockfish_eval_train_t.1_merge.pi",
            test_file_name="data/datasets/goodgames_plusvariation_stockfish_eval_test",
            preprocessing_data_set=False,
        )
    )


class LearnNNScript:
    """Script that learns a NN from a supervised dataset pairs of board and evaluation.

    Args:
        base_script (Script): The base script object.

    Attributes:
        base_script (Script): The base script object.
        nn (ChiNN): The neural network object.
        args (LearnNNScriptArgs): The script arguments.

    """

    args_dataclass_name: type[LearnNNScriptArgs] = LearnNNScriptArgs

    base_experiment_output_folder = os.path.join(
        Script.base_experiment_output_folder,
        "learn_nn_supervised/learn_nn_supervised_outputs",
    )

    base_script: Script[LearnNNScriptArgs]
    nn: ChiNN
    args: LearnNNScriptArgs
    nn_board_evaluator: NNBWStateEvaluator[ChessState]
    saving_folder: path

    def __init__(
        self,
        base_script: Script[LearnNNScriptArgs],
    ) -> None:
        """Initialize the LearnNNFromSupervisedDatasets class.

        Args:
            base_script (Script): The base script object.

        Returns:
            None

        """
        self.start_time = time.time()

        # Setting up the dataloader from the evaluation files
        self.base_script = base_script

        # Calling the init of Script that takes care of a lot of stuff, especially parsing the arguments into self.args
        self.args: LearnNNScriptArgs = self.base_script.initiate(
            experiment_output_folder=self.base_experiment_output_folder,
        )

        # taking care of random
        chipiron.set_seeds(seed=self.args.base_script_args.seed)

        if self.args.nn_trainer_args.reuse_existing_model:
            assert (
                self.args.nn_trainer_args.nn_parameters_file_if_reusing_existing_one
                is not None
            )
            content_to_input_convert = create_content_to_input_from_model_weights(
                self.args.nn_trainer_args.nn_parameters_file_if_reusing_existing_one
            )
            self.nn_board_evaluator = create_nn_state_eval_from_nn_parameters_file_and_existing_model(
                model_weights_file_name=self.args.nn_trainer_args.nn_parameters_file_if_reusing_existing_one,
                nn_architecture_args=self.args.nn_trainer_args.neural_network_architecture_args,
                content_to_input_convert=content_to_input_convert,
            )
        else:
            chipiron_nn_args = ChipironNNArgs(
                version=1,
                game_kind=self.args.nn_trainer_args.game_input.game_kind,
                input_representation=self.args.nn_trainer_args.game_input.representation.value,
            )
            content_to_input_convert = create_content_to_input_convert(chipiron_nn_args)
            self.nn_board_evaluator = create_nn_state_eval_from_architecture_args(
                nn_architecture_args=self.args.nn_trainer_args.neural_network_architecture_args,
                content_to_input_convert=content_to_input_convert,
            )

        if self.args.nn_trainer_args.specific_saving_folder is not None:
            self.saving_folder = self.args.nn_trainer_args.specific_saving_folder
        else:
            assert self.args.base_script_args.experiment_output_folder is not None
            self.saving_folder = self.args.base_script_args.experiment_output_folder

        self.nn_trainer: NNPytorchTrainer = create_nn_trainer(
            args=self.args.nn_trainer_args,
            nn=self.nn_board_evaluator.net,
            saving_folder=self.saving_folder,
        )

        self.stockfish_boards_train = FenAndValueDataSet(
            file_name=self.args.dataset_args.train_file_name,
            preprocessing=self.args.dataset_args.preprocessing_data_set,
            transform_board_function=self.nn_board_evaluator.content_to_input_convert,
            transform_dataset_value_to_white_value_function=process_stockfish_value,  # pyright: ignore[reportUnknownArgumentType]
            transform_white_value_to_model_output_function=self.nn_board_evaluator.output_and_value_converter.from_value_white_to_model_output,
        )

        assert self.args.dataset_args.test_file_name is not None
        self.stockfish_boards_test = FenAndValueDataSet(
            file_name=self.args.dataset_args.test_file_name,
            preprocessing=self.args.dataset_args.preprocessing_data_set,
            transform_board_function=self.nn_board_evaluator.content_to_input_convert,
            transform_dataset_value_to_white_value_function=process_stockfish_value,  # pyright: ignore[reportUnknownArgumentType]
            transform_white_value_to_model_output_function=self.nn_board_evaluator.output_and_value_converter.from_value_white_to_model_output,
        )

        start_time = time.time()
        self.stockfish_boards_train.load()
        chipiron_logger.info("--- LOAD %s seconds --- ", time.time() - start_time)
        self.stockfish_boards_test.load()

        self.data_loader_stockfish_boards_train = DataLoader[FenAndValueData](
            self.stockfish_boards_train,
            batch_size=self.args.nn_trainer_args.batch_size_train,
            shuffle=True,
            num_workers=1,
            collate_fn=custom_collate_fn_fen_and_value,
        )

        self.data_loader_stockfish_boards_test = DataLoader(
            self.stockfish_boards_test,
            batch_size=self.args.nn_trainer_args.batch_size_test,
            shuffle=True,
            num_workers=1,
            collate_fn=custom_collate_fn_fen_and_value,
        )

        if self.args.base_script_args.testing:
            # mlflow.set_tracking_uri(
            #    uri=chipiron.utils.path_variables.ML_FLOW_URI_PATH_TEST
            # )
            with tempfile.TemporaryDirectory() as tmpdir:
                mlflow.set_tracking_uri(f"file:{tmpdir}")  # <- Key line!
        else:
            mlflow.set_tracking_uri(uri=chipiron.utils.path_variables.ML_FLOW_URI_PATH)

    def print_and_log_metrics(
        self, count_train_step: int, training_loss: float, test_error: float
    ) -> None:
        """Print and log training metrics to console and MLflow.

        Args:
            count_train_step (int): Current training step count.
            training_loss (float): Current training loss value.
            test_error (float): Current test error value.

        Returns:
            None

        """
        chipiron_logger.info(
            "count_train_step: %s, training loss: %s, lr: %s, time_elapsed: %s",
            count_train_step,
            training_loss,
            self.nn_trainer.scheduler.get_last_lr(),
            time.time() - self.start_time,
        )
        mlflow.log_metric(
            "training_loss",
            training_loss,
            step=count_train_step,
        )
        mlflow.log_metric(
            "test_error",
            test_error,
            step=count_train_step,
        )
        mlflow.log_metric(
            "lr",
            self.nn_trainer.scheduler.get_last_lr()[-1],
            step=count_train_step,
        )

    def run(self) -> None:
        """Run the learning of the NN.

        This method performs the training of the neural network. It iterates over the training data batches,
        computes the training loss, and updates the learning rate if necessary. It also prints the training
        loss and learning rate at regular intervals, and saves the learning process.

        Returns:
            None

        """
        chipiron_logger.info("Starting to learn the NN")

        with mlflow.start_run():
            params = asdict(self.args) | asdict(
                self.args.nn_trainer_args.neural_network_architecture_args
            )
            mlflow.log_params(params)

            # Log model summary.
            model_summary_file_name: str = os.path.join(
                self.saving_folder,
                "model_summary.txt",
            )
            with open(model_summary_file_name, "w", encoding="utf-8") as f:
                f.write(str(summary(self.nn_board_evaluator.net)))
            mlflow.log_artifact(model_summary_file_name)

            count_train_step = 0
            sum_loss_train: float = 0.0
            sum_loss_train_print: float = 0.0
            previous_dict: Any | None = None
            previous_train_loss: float | None = None

            fens_and_values_sample_batch: FenAndValueData
            for _ in range(self.args.nn_trainer_args.epochs_number):
                for __, fens_and_values_sample_batch in enumerate(
                    self.data_loader_stockfish_boards_train
                ):
                    # printing info to console
                    if count_train_step % 10000 == 0 and count_train_step > 0:
                        training_loss: float = sum_loss_train_print / 10000
                        sum_loss_train_print = 0
                        test_error: float = (
                            self.nn_trainer.compute_test_error_on_dataset(
                                data_test=self.data_loader_stockfish_boards_test
                            )
                        )
                        self.print_and_log_metrics(
                            count_train_step=count_train_step,
                            training_loss=training_loss,
                            test_error=test_error,
                        )

                    if (
                        count_train_step % 2000 == 0
                        and count_train_step > 0
                        and self.args.base_script_args.testing
                    ):
                        break

                    # every self.args['min_interval_lr_change'] steps we check for possibly decreasing the learning rate
                    if (
                        count_train_step
                        % self.args.nn_trainer_args.min_interval_lr_change
                        == 0
                        and count_train_step > 0
                    ):
                        # condition to decrease the learning rate
                        if (
                            previous_train_loss is not None
                            and sum_loss_train > previous_train_loss
                            and self.nn_trainer.scheduler.get_last_lr()[-1]
                            > self.args.nn_trainer_args.min_lr
                        ):
                            self.nn_trainer.scheduler.step()
                            chipiron_logger.info(
                                "decaying the learning rate to %s",
                                self.nn_trainer.scheduler.get_last_lr(),
                            )

                        chipiron_logger.info("count_train_step %s", count_train_step)
                        chipiron_logger.info(
                            "training loss: %s, sum_loss_train: %s",
                            sum_loss_train
                            / self.args.nn_trainer_args.min_interval_lr_change,
                            sum_loss_train,
                        )
                        chipiron_logger.info(
                            "previous_train_loss %s", previous_train_loss
                        )
                        chipiron_logger.info(
                            "learning rate %s", self.nn_trainer.scheduler.get_last_lr()
                        )

                        if previous_dict is not None:
                            diff_weighs = sum(
                                (x - y).abs().sum()
                                for x, y in zip(
                                    previous_dict.values(),
                                    self.nn_board_evaluator.net.state_dict().values(),
                                    strict=False,
                                )
                            )
                            print("diff_weights", diff_weighs)
                        previous_dict = copy.deepcopy(
                            self.nn_board_evaluator.net.state_dict()
                        )

                        previous_train_loss = sum_loss_train
                        sum_loss_train = 0.0

                    # MAIN: the training bit
                    count_train_step += 1
                    loss_train = self.nn_trainer.train(
                        fens_and_values_sample_batch.get_input_layer(),
                        fens_and_values_sample_batch.get_target_value(),
                    )
                    sum_loss_train += float(loss_train)
                    sum_loss_train_print += float(loss_train)

                    # saving the learning process
                    self.saving_things_to_file(
                        count_train_step=count_train_step,
                        x_train=fens_and_values_sample_batch.get_input_layer(),
                    )

    def saving_things_to_file(
        self, count_train_step: int, x_train: torch.Tensor
    ) -> None:
        """Save the neural network parameters and trainer to file.

        Args:
            count_train_step (int): The current training step count.

        Returns:
            None

        """
        if count_train_step % self.args.nn_trainer_args.saving_interval == 0:
            safe_nn_param_save(
                nn=self.nn_board_evaluator.net,
                nn_param_folder_name=self.saving_folder,
                file_name=self.args.nn_trainer_args.neural_network_architecture_args.filename(),
            )
            safe_nn_architecture_save(
                nn_architecture_args=self.args.nn_trainer_args.neural_network_architecture_args,
                nn_param_folder_name=self.saving_folder,
            )
            save_chipiron_nn_args(
                ChipironNNArgs(
                    version=1,
                    game_kind=self.args.nn_trainer_args.game_input.game_kind,
                    input_representation=self.args.nn_trainer_args.game_input.representation.value,
                ),
                folder_path=self.saving_folder,
            )
            safe_nn_trainer_save(self.nn_trainer, self.saving_folder)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            signature: ModelSignature = infer_signature(
                x_train.numpy(),
                self.nn_board_evaluator.net(x_train.to(device).detach())
                .cpu()
                .detach()
                .numpy(),
            )
            mlflow.pytorch.log_model(
                self.nn_board_evaluator.net,
                "model",
                signature=signature,
                conda_env=mlflow.pytorch.get_default_conda_env(),
            )

        if (
            self.args.nn_trainer_args.saving_intermediate_copy
            and count_train_step
            % self.args.nn_trainer_args.saving_intermediate_copy_interval
            == 0
        ):
            safe_nn_param_save(
                nn=self.nn_board_evaluator.net,
                nn_param_folder_name=self.saving_folder,
                training_copy=True,
            )

    def terminate(self) -> None:
        """Finishing the script. Profiling or timing."""
        self.base_script.terminate()

    @classmethod
    def get_args_dataclass_name(cls) -> type[LearnNNScriptArgs]:
        """Return the dataclass type that holds the arguments for the script.

        Returns:
            type: The dataclass type for the script's arguments.

        """
        return LearnNNScriptArgs

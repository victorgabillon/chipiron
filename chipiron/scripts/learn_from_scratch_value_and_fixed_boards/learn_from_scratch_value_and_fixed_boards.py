import os
import time
from dataclasses import asdict, dataclass, field

import mlflow
from torchinfo import summary

import chipiron
from chipiron.learningprocesses.nn_trainer.factory import NNTrainerArgs
from chipiron.players.boardevaluators.neural_networks.neural_net_board_eval_args import (
    NeuralNetArchitectureArgs,
)
from chipiron.players.boardevaluators.neural_networks.nn_board_evaluator import (
    NNBoardEvaluator,
)
from chipiron.scripts.script import Script
from chipiron.scripts.script_args import BaseScriptArgs
from chipiron.utils.chi_nn import ChiNN


@dataclass
class LearnNNFromScratchScriptArgs:
    """
    Represents the arguments for the LearnNNFromScratchScript.


    """

    base_script_args: BaseScriptArgs = field(default_factory=BaseScriptArgs)
    nn_trainer_args: NNTrainerArgs = field(default_factory=NNTrainerArgs)

    epochs_number: int = 1


class LearnNNFromScratchScript:
    """
    Script that learns a NN from a supervised dataset pairs of board and evaluation

    Args:
        base_script (Script): The base script object.

    Attributes:
        base_script (Script): The base script object.
        nn (ChiNN): The neural network object.
        args (LearnNNFromScratchScriptArgs): The script arguments.

    """

    args_dataclass_name: type[LearnNNFromScratchScriptArgs] = (
        LearnNNFromScratchScriptArgs
    )

    base_experiment_output_folder = os.path.join(
        Script.base_experiment_output_folder,
        "learn_from_scratch_value_and_fixed_boards/learn_from_scratch_outputs",
    )

    base_script: Script
    nn: ChiNN
    args: LearnNNFromScratchScriptArgs
    nn_board_evaluator: NNBoardEvaluator
    nn_architecture_args: NeuralNetArchitectureArgs

    def __init__(
        self,
        base_script: Script,
    ) -> None:
        """
        Initializes the LearnNNFromSupervisedDatasets class.

        Args:
            base_script (Script): The base script object.

        Returns:
            None
        """

        self.start_time = time.time()

        # Setting up the dataloader from the evaluation files
        self.base_script = base_script

        # Calling the init of Script that takes care of a lot of stuff, especially parsing the arguments into self.args
        self.args: LearnNNFromScratchScriptArgs = self.base_script.initiate(
            args_dataclass_name=LearnNNFromScratchScriptArgs,
            experiment_output_folder=self.base_experiment_output_folder,
        )

        # taking care of random
        chipiron.set_seeds(seed=self.args.base_script_args.seed)

    def run(self) -> None:
        """Running the learning of the NN from scratch."""

        print("Starting to learn the NN from scratch")

        with mlflow.start_run():

            params = asdict(self.args) | asdict(self.nn_architecture_args)
            mlflow.log_params(params)

            # Log model summary.
            model_summary_file_name: str = os.path.join(
                self.args.nn_trainer_args.neural_network_folder_path,
                "model_summary.txt",
            )
            with open(model_summary_file_name, "w") as f:
                f.write(str(summary(self.nn_board_evaluator.net)))
            mlflow.log_artifact(model_summary_file_name)

            # count_train_step = 0
            # sum_loss_train: float = 0.0
            # sum_loss_train_print: float = 0.0
            # previous_dict: Any | None = None
            # previous_train_loss: float | None = None
            for i in range(self.args.epochs_number):
                ...
                # play

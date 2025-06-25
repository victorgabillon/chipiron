import logging
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import mlflow
import pandas
from torch.utils.data import DataLoader
from torchinfo import summary

import chipiron
from chipiron.environments.chess.board.utils import FenPlusHistory
from chipiron.learningprocesses.nn_trainer.factory import NNTrainerArgs
from chipiron.players import PlayerArgs
from chipiron.players.boardevaluators.datasets.datasets import (
    DataSetArgs,
    FenAndValueDataSet,
)
from chipiron.players.boardevaluators.neural_networks.factory import (
    create_nn_board_eval_from_architecture_args,
    create_nn_board_eval_from_nn_parameters_file_and_existing_model,
)
from chipiron.players.boardevaluators.neural_networks.neural_net_board_eval_args import (
    NeuralNetArchitectureArgs,
)
from chipiron.players.boardevaluators.neural_networks.nn_board_evaluator import (
    NNBoardEvaluator,
)
from chipiron.players.boardevaluators.table_base import SyzygyTable
from chipiron.players.boardevaluators.table_base.factory import create_syzygy
from chipiron.players.factory import create_player
from chipiron.players.move_selector.move_selector_types import MoveSelectorTypes
from chipiron.players.move_selector.random import Random
from chipiron.players.player_ids import PlayerConfigTag
from chipiron.scripts.chipiron_args import ImplementationArgs
from chipiron.scripts.script import Script
from chipiron.scripts.script_args import BaseScriptArgs
from chipiron.utils import path
from chipiron.utils.chi_nn import ChiNN
from chipiron.utils.logger import chipiron_logger, suppress_logging


@dataclass
class LearnNNFromScratchScriptArgs:
    """
    Represents the arguments for the LearnNNFromScratchScript.


    """

    epochs_number_with_respect_to_evaluating_player: int = 10
    number_of_evaluating_player_per_loop: int = 10
    number_of_gradient_descent_per_loop: int = 10
    starting_boards_are_non_labelled: bool = True

    dataset_args: DataSetArgs = field(
        default_factory=lambda: DataSetArgs(
            train_file_name="data/datasets/modified_fen_test.pkl",
            test_file_name=None,
            preprocessing_data_set=False,
        )
    )

    evaluating_player_args: PlayerArgs | PlayerConfigTag = field(
        default_factory=lambda: PlayerArgs(
            name="Random",
            main_move_selector=Random(type=MoveSelectorTypes.Random),
            syzygy_play=False,
        )
    )

    base_script_args: BaseScriptArgs = field(default_factory=BaseScriptArgs)
    nn_trainer_args: NNTrainerArgs = field(default_factory=NNTrainerArgs)
    implementation_args: ImplementationArgs = field(default_factory=ImplementationArgs)


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

    base_script: Script[LearnNNFromScratchScriptArgs]
    nn: ChiNN
    args: LearnNNFromScratchScriptArgs
    nn_board_evaluator: NNBoardEvaluator
    nn_architecture_args: NeuralNetArchitectureArgs
    saving_folder: path

    def __init__(
        self,
        base_script: Script[LearnNNFromScratchScriptArgs],
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
            experiment_output_folder=self.base_experiment_output_folder,
        )

        if self.args.nn_trainer_args.reuse_existing_model:
            assert (
                self.args.nn_trainer_args.nn_parameters_file_if_reusing_existing_one
                is not None
            )
            self.nn_board_evaluator = create_nn_board_eval_from_nn_parameters_file_and_existing_model(
                model_weights_file_name=self.args.nn_trainer_args.nn_parameters_file_if_reusing_existing_one,
                nn_architecture_args=self.args.nn_trainer_args.neural_network_architecture_args,
            )
        else:
            self.nn_board_evaluator = create_nn_board_eval_from_architecture_args(
                nn_architecture_args=self.args.nn_trainer_args.neural_network_architecture_args
            )

        if self.args.nn_trainer_args.specific_saving_folder is not None:
            self.saving_folder = self.args.nn_trainer_args.specific_saving_folder
        else:
            assert self.args.base_script_args.experiment_output_folder is not None
            self.saving_folder = self.args.base_script_args.experiment_output_folder

        # taking care of random
        chipiron.set_seeds(seed=self.args.base_script_args.seed)

        if self.args.starting_boards_are_non_labelled:

            def transform_dataset_value_to_white_value_function(
                row: pandas.Series,
            ) -> float:
                return 0.0

        else:

            def transform_dataset_value_to_white_value_function(
                row: pandas.Series,
            ) -> float:
                assert isinstance(row["value"], float)
                return row["value"]

        self.boards_dataset = FenAndValueDataSet(
            file_name=self.args.dataset_args.train_file_name,
            preprocessing=self.args.dataset_args.preprocessing_data_set,
            transform_board_function=self.nn_board_evaluator.board_to_input_convert,
            transform_dataset_value_to_white_value_function=transform_dataset_value_to_white_value_function,
            transform_white_value_to_model_output_function=self.nn_board_evaluator.output_and_value_converter.from_value_white_to_model_output,
        )
        self.boards_dataset.load()

        self.data_loader_value_function = DataLoader(
            self.boards_dataset,
            batch_size=self.args.nn_trainer_args.batch_size_train,
            shuffle=True,
            num_workers=1,
        )

        self.index_evaluating_player_data: int = 0

        if self.args.base_script_args.testing:
            mlflow.set_tracking_uri(
                uri=chipiron.utils.path_variables.ML_FLOW_URI_PATH_TEST
            )
        else:
            mlflow.set_tracking_uri(uri=chipiron.utils.path_variables.ML_FLOW_URI_PATH)

        random_generator = random.Random(self.args.base_script_args.seed)
        syzygy_table: SyzygyTable[Any] | None = create_syzygy(
            use_rust=self.args.implementation_args.use_rust_boards,
        )
        assert isinstance(
            self.args.evaluating_player_args, PlayerArgs
        )  # because of the magic automatic parsing transcription, can we remove this assert somehow?
        self.player = create_player(
            args=self.args.evaluating_player_args,
            syzygy=syzygy_table,
            random_generator=random_generator,
            implementation_args=self.args.implementation_args,
            universal_behavior=self.args.base_script_args.universal_behavior,
        )

    def run(self) -> None:
        """Running the learning of the NN from scratch."""

        print("Starting to learn the NN from scratch")

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
            with open(model_summary_file_name, "w") as f:
                f.write(str(summary(self.nn_board_evaluator.net)))
            mlflow.log_artifact(model_summary_file_name)

            for i in range(self.args.epochs_number_with_respect_to_evaluating_player):
                # update the dataset with a tree search with the current model
                self.update_dataset_value_with_evaluating_player()

                # update the model perform a batch gradient descent with the updated dataset
                self.learn_model_some_steps()

    def learn_model_some_steps(self) -> None: ...

    def update_dataset_value_with_evaluating_player(self) -> None:

        index_range_evaluating_player: int
        for index_range_evaluating_player in range(
            self.args.number_of_evaluating_player_per_loop
        ):
            index_evaluating_player_data_temp: int = (
                self.index_evaluating_player_data + index_range_evaluating_player
            )

            board_fen_to_recompute_value = self.boards_dataset.get_unprocessed(
                idx=index_evaluating_player_data_temp
            )["fen"]

            assert isinstance(self.boards_dataset.data, pandas.DataFrame)
            # print(
            #    "debug",
            #    board_fen_to_recompute_value,
            #    len(self.boards_dataset.data),
            #    index_evaluating_player_data_temp,
            # )

            with suppress_logging(
                chipiron_logger, level=logging.WARNING
            ):  # no logging during the tree search
                self.player.select_move(
                    fen_plus_history=FenPlusHistory(
                        current_fen=board_fen_to_recompute_value
                    ),
                    seed_int=self.args.base_script_args.seed,
                )

            self.index_evaluating_player_data += 1

        # record_the_dtaa
        # set in a
        # file

    def terminate(self) -> None:
        """
        Finishing the script. Profiling or timing.
        """
        pass

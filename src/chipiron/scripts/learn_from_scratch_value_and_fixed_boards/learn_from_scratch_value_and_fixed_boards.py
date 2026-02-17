"""Learn a neural network from scratch from a supervised dataset of boards and values,.

and from a fixed set of non-labelled boards.
"""
# pylint: disable=duplicate-code

import logging
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import cast

import mlflow
import pandas as pd
from atomheart.board.utils import FenPlusHistory
from coral.chi_nn import ChiNN
from coral.neural_networks.factory import (
    create_nn_state_eval_from_architecture_args,
    create_nn_state_eval_from_nn_parameters_file_and_existing_model,
)
from coral.neural_networks.neural_net_architecture_args import (
    NeuralNetArchitectureArgs,
)
from coral.neural_networks.nn_state_evaluator import (
    NNBWStateEvaluator,
)
from torch.utils.data import DataLoader
from torchinfo import summary  # pyright: ignore[reportUnknownVariableType]

import chipiron
from chipiron.environments.chess.types import ChessState
from chipiron.learningprocesses.nn_trainer.factory import NNTrainerArgs
from chipiron.players import PlayerArgs
from chipiron.players.adapters.chess_syzygy_oracle import (
    ChessSyzygyPolicyOracle,
    ChessSyzygyTerminalOracle,
    ChessSyzygyValueOracle,
)
from chipiron.players.boardevaluators.datasets.datasets import (
    DataSetArgs,
    FenAndValueDataSet,
)
from chipiron.players.boardevaluators.neural_networks.chipiron_nn_args import (
    ChipironNNArgs,
    create_content_to_input_convert,
    create_content_to_input_from_model_weights,
)
from chipiron.players.boardevaluators.table_base.factory import (
    AnySyzygyTable,
    create_syzygy,
)
from chipiron.players.factory import create_chess_player
from chipiron.players.move_selector.random_args import RandomSelectorArgs
from chipiron.players.player_ids import PlayerConfigTag
from chipiron.scripts.chipiron_args import ImplementationArgs
from chipiron.scripts.script import Script
from chipiron.scripts.script_args import BaseScriptArgs
from chipiron.utils import MyPath
from chipiron.utils.logger import chipiron_logger, suppress_logging
from chipiron.utils.path_variables import ML_FLOW_URI_PATH


@dataclass
class LearnNNFromScratchScriptArgs:
    """Represents the arguments for the LearnNNFromScratchScript."""

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
            main_move_selector=RandomSelectorArgs(),
            syzygy_play=False,
        )
    )

    base_script_args: BaseScriptArgs = field(default_factory=BaseScriptArgs)
    nn_trainer_args: NNTrainerArgs = field(default_factory=NNTrainerArgs)
    implementation_args: ImplementationArgs = field(default_factory=ImplementationArgs)


class LearnNNFromScratchScript:
    """Script that learns a NN from a supervised dataset pairs of board and evaluation.

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
    nn_board_evaluator: NNBWStateEvaluator[ChessState]
    nn_architecture_args: NeuralNetArchitectureArgs
    saving_folder: MyPath

    def __init__(
        self,
        base_script: Script[LearnNNFromScratchScriptArgs],
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
        self.args: LearnNNFromScratchScriptArgs = self.base_script.initiate(
            experiment_output_folder=self.base_experiment_output_folder,
        )

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

        # taking care of random
        chipiron.set_seeds(seed=self.args.base_script_args.seed)

        def transform_dataset_value_to_white_value_function(
            row: pd.Series,
        ) -> float:
            # Cast to help type checker understand the expected type
            value = cast("float", row["value"])
            return float(value)

        self.boards_dataset = FenAndValueDataSet(
            file_name=self.args.dataset_args.train_file_name,
            preprocessing=self.args.dataset_args.preprocessing_data_set,
            transform_board_function=self.nn_board_evaluator.content_to_input_convert,
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
            with tempfile.TemporaryDirectory() as tmpdir:
                mlflow.set_tracking_uri(f"file:{tmpdir}")  # <- Key line!
        else:
            mlflow.set_tracking_uri(uri=ML_FLOW_URI_PATH)

        random_generator = random.Random(self.args.base_script_args.seed)
        syzygy_table: AnySyzygyTable | None = create_syzygy(
            use_rust=self.args.implementation_args.use_rust_boards,
        )

        assert not isinstance(self.args.evaluating_player_args, PlayerConfigTag), (
            "PlayerConfigTag is not supported in this script."
        )
        chess_args = self.args.evaluating_player_args
        self.player = create_chess_player(
            args=chess_args,
            terminal_oracle=ChessSyzygyTerminalOracle(syzygy_table)
            if syzygy_table is not None
            else None,
            value_oracle=ChessSyzygyValueOracle(syzygy_table)
            if syzygy_table is not None
            else None,
            policy_oracle=ChessSyzygyPolicyOracle(syzygy_table)
            if syzygy_table is not None
            else None,
            random_generator=random_generator,
            implementation_args=self.args.implementation_args,
            universal_behavior=self.args.base_script_args.universal_behavior,
        )

    def run(self) -> None:
        """Run the learning of the NN from scratch."""
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
            with open(model_summary_file_name, "w", encoding="utf-8") as f:
                f.write(str(summary(self.nn_board_evaluator.net)))
            mlflow.log_artifact(model_summary_file_name)

            for _i in range(self.args.epochs_number_with_respect_to_evaluating_player):
                # update the dataset with a tree search with the current model
                self.update_dataset_value_with_evaluating_player()

                # update the model perform a batch gradient descent with the updated dataset
                self.learn_model_some_steps()

    def learn_model_some_steps(self) -> None:
        """Perform a few steps of training on the model."""

    def update_dataset_value_with_evaluating_player(self) -> None:
        """Update the dataset with the values from the evaluating player."""
        index_range_evaluating_player: int
        for index_range_evaluating_player in range(
            self.args.number_of_evaluating_player_per_loop
        ):
            index_evaluating_player_data_temp: int = (
                self.index_evaluating_player_data + index_range_evaluating_player
            )

            board_fen_to_recompute_value: str = cast(
                "str",
                self.boards_dataset.get_unprocessed(
                    idx=index_evaluating_player_data_temp
                )["fen"],
            )

            assert isinstance(self.boards_dataset.data, pd.DataFrame)

            with suppress_logging(
                chipiron_logger, level=logging.WARNING
            ):  # no logging during the tree search
                self.player.select_move(
                    state_snapshot=FenPlusHistory(
                        current_fen=board_fen_to_recompute_value
                    ),
                    seed=self.args.base_script_args.seed,
                )

            self.index_evaluating_player_data += 1

        # record_the_dtaa
        # set in a
        # file

    def terminate(self) -> None:
        """Finishing the script. Profiling or timing."""

    @classmethod
    def get_args_dataclass_name(cls) -> type[LearnNNFromScratchScriptArgs]:
        """Return the dataclass type that holds the arguments for the script.

        Returns:
            type: The dataclass type for the script's arguments.

        """
        return LearnNNFromScratchScriptArgs

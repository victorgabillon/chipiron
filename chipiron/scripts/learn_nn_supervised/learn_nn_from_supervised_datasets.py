"""
This script learns a neural network (NN) from a supervised dataset of board and evaluation pairs.

The script takes care of setting up the data loader from the evaluation files, creating the NN, and running the learning process.

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
import os
import time
from dataclasses import dataclass, field
from typing import Any

from torch.utils.data import DataLoader

from chipiron.learningprocesses.nn_trainer.factory import (
    NNTrainerArgs,
    create_nn_trainer,
    safe_nn_param_save,
    safe_nn_trainer_save,
    safe_nn_architecture_save,
)
from chipiron.learningprocesses.nn_trainer.nn_trainer import NNPytorchTrainer
from chipiron.players.boardevaluators.datasets.datasets import FenAndValueDataSet
from chipiron.players.boardevaluators.neural_networks.factory import (
    create_nn,
    create_nn_from_folder_path_and_existing_model,
    get_architecture_args_from_file,
)
from chipiron.players.boardevaluators.neural_networks.input_converters.TensorRepresentationType import (
    InternalTensorRepresentationType,
    get_default_internal_representation,
)
from chipiron.players.boardevaluators.neural_networks.input_converters.factory import (
    RepresentationFactory,
)
from chipiron.players.boardevaluators.neural_networks.input_converters.representation_364_bti import (
    RepresentationBTI,
)
from chipiron.players.boardevaluators.neural_networks.input_converters.representation_factory_factory import (
    create_board_representation_factory,
)
from chipiron.players.boardevaluators.neural_networks.neural_net_board_eval_args import (
    NeuralNetArchitectureArgs,
)
from chipiron.scripts.script import Script
from chipiron.scripts.script_args import BaseScriptArgs
from chipiron.utils.chi_nn import ChiNN


@dataclass
class LearnNNScriptArgs:
    """
    Represents the arguments for the LearnNNScript.

    Attributes:
        nn_trainer_args (NNTrainerArgs): The arguments for the NNTrainer.
        stockfish_boards_train_file_name (str): The file name for the training dataset.
        stockfish_boards_test_file_name (str): The file name for the test dataset.
        preprocessing_data_set (bool): Whether to preprocess the dataset.
        batch_size_train (int): The batch size for training.
        batch_size_test (int): The batch size for testing.
        saving_interval (int): The interval for saving the model.
        saving_intermediate_copy_interval (int): The interval for saving intermediate copies of the model.
        min_interval_lr_change (int): The minimum interval for changing the learning rate.
        min_lr (float): The minimum learning rate.
    """

    base_script_args: BaseScriptArgs = field(default_factory=BaseScriptArgs)
    nn_trainer_args: NNTrainerArgs = field(default_factory=NNTrainerArgs)
    stockfish_boards_train_file_name: str = (
        "data/datasets/goodgames_plusvariation_stockfish_eval_train_t.1_merge.pi"
    )
    stockfish_boards_test_file_name: str = (
        "data/datasets/goodgames_plusvariation_stockfish_eval_test"
    )
    preprocessing_data_set: bool = False
    batch_size_train: int = 32
    batch_size_test: int = 10
    saving_interval: int = 1000
    saving_intermediate_copy_interval: int = 10000
    min_interval_lr_change: int = 1000000
    min_lr: float = 0.001
    epochs_number: int = 1
    test: bool = False  # hack to test fast, change at some point


class LearnNNScript:
    """
    Script that learns a NN from a supervised dataset pairs of board and evaluation

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

    base_script: Script
    nn: ChiNN
    args: LearnNNScriptArgs

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
        # Setting up the dataloader from the evaluation files
        self.base_script = base_script

        # Calling the init of Script that takes care of a lot of stuff, especially parsing the arguments into self.args
        self.args: LearnNNScriptArgs = self.base_script.initiate(
            args_dataclass_name=LearnNNScriptArgs,
            experiment_output_folder=self.base_experiment_output_folder,
        )

        if self.args.nn_trainer_args.reuse_existing_model:
            self.nn, self.nn_architecture_args = (
                create_nn_from_folder_path_and_existing_model(
                    folder_path=self.args.nn_trainer_args.neural_network_folder_path
                )
            )

        else:
            assert (
                self.args.nn_trainer_args.nn_architecture_file_if_not_reusing_existing_one
                is not None
            )
            self.nn_architecture_args: NeuralNetArchitectureArgs = (
                get_architecture_args_from_file(
                    architecture_file_name=self.args.nn_trainer_args.nn_architecture_file_if_not_reusing_existing_one
                )
            )
            self.nn = create_nn(nn_type=self.nn_architecture_args.model_type)
            self.nn.init_weights()
        self.nn.print_param()
        self.nn_trainer: NNPytorchTrainer = create_nn_trainer(
            args=self.args.nn_trainer_args, nn=self.nn
        )

        internal_tensor_representation_type: InternalTensorRepresentationType = (
            get_default_internal_representation(
                tensor_representation_type=self.nn_architecture_args.tensor_representation_type
            )
        )
        board_representation_factory: RepresentationFactory[Any] | None = (
            create_board_representation_factory(
                board_representation_factory_type=internal_tensor_representation_type
            )
        )
        assert board_representation_factory is not None
        board_to_input = RepresentationBTI(
            representation_factory=board_representation_factory
        )

        self.stockfish_boards_train = FenAndValueDataSet(
            file_name=self.args.stockfish_boards_train_file_name,
            preprocessing=self.args.preprocessing_data_set,
            transform_board_function=board_to_input.convert,
            transform_value_function="stockfish",
        )

        self.stockfish_boards_test = FenAndValueDataSet(
            file_name=self.args.stockfish_boards_test_file_name,
            preprocessing=self.args.preprocessing_data_set,
            transform_board_function=board_to_input.convert,
            transform_value_function="stockfish",
        )

        start_time = time.time()
        self.stockfish_boards_train.load()
        print("--- LOADdd %s seconds ---" % (time.time() - start_time))
        self.stockfish_boards_test.load()

        self.data_loader_stockfish_boards_train = DataLoader(
            self.stockfish_boards_train,
            batch_size=self.args.batch_size_train,
            shuffle=True,
            num_workers=1,
        )

        self.data_loader_stockfish_boards_test = DataLoader(
            self.stockfish_boards_test,
            batch_size=self.args.batch_size_test,
            shuffle=True,
            num_workers=1,
        )

    def run(self) -> None:
        """Running the learning of the NN.

        This method performs the training of the neural network. It iterates over the training data batches,
        computes the training loss, and updates the learning rate if necessary. It also prints the training
        loss and learning rate at regular intervals, and saves the learning process.

        Returns:
            None
        """
        print("Starting to learn the NN")
        count_train_step = 0
        sum_loss_train: float = 0.0
        sum_loss_train_print: float = 0.0
        previous_dict: Any | None = None
        previous_train_loss: float | None = None
        for i in range(self.args.epochs_number):
            for i_batch, sample_batched in enumerate(
                self.data_loader_stockfish_boards_train
            ):

                # printing info to console
                if count_train_step % 10000 == 0 and count_train_step > 0:
                    print(
                        "count_train_step",
                        count_train_step,
                        "training loss",
                        sum_loss_train_print / 10000,
                        "lr",
                        self.nn_trainer.scheduler.get_last_lr(),
                    )
                    sum_loss_train_print = 0
                    self.nn_trainer.compute_test_error_on_dataset(
                        data_test=self.data_loader_stockfish_boards_test
                    )

                if (
                    count_train_step % 20000 == 0
                    and count_train_step > 0
                    and self.args.test
                ):
                    break

                # every self.args['min_interval_lr_change'] steps we check for possibly decreasing the learning rate
                if (
                    count_train_step % self.args.min_interval_lr_change == 0
                    and count_train_step > 0
                ):

                    # condition to decrease the learning rate
                    if (
                        previous_train_loss is not None
                        and sum_loss_train > previous_train_loss
                        and self.nn_trainer.scheduler.get_last_lr()[-1]
                        > self.args.min_lr
                    ):
                        self.nn_trainer.scheduler.step()
                        print(
                            "decaying the learning rate to",
                            self.nn_trainer.scheduler.get_last_lr(),
                        )

                    print("count_train_step", count_train_step)
                    print(
                        "training loss",
                        sum_loss_train / self.args.min_interval_lr_change,
                        sum_loss_train,
                    )
                    print("previous_train_loss", previous_train_loss)
                    print("learning rate", self.nn_trainer.scheduler.get_last_lr())

                    if previous_dict is not None:
                        diff_weighs = sum(
                            (x - y).abs().sum()
                            for x, y in zip(
                                previous_dict.values(), self.nn.state_dict().values()
                            )
                        )
                        print("diff_weights", diff_weighs)
                    previous_dict = copy.deepcopy(self.nn.state_dict())

                    previous_train_loss = sum_loss_train
                    sum_loss_train = 0.0

                # MAIN: the training bit
                count_train_step += 1
                loss_train = self.nn_trainer.train(sample_batched[0], sample_batched[1])
                sum_loss_train += float(loss_train)
                sum_loss_train_print += float(loss_train)

                # saving the learning process
                self.saving_things_to_file(count_train_step)

    def saving_things_to_file(self, count_train_step: int) -> None:
        """
        Saves the neural network parameters and trainer to file.

        Args:
            count_train_step (int): The current training step count.

        Returns:
            None
        """
        if count_train_step % self.args.saving_interval == 0:
            safe_nn_param_save(
                self.nn, self.args.nn_trainer_args.neural_network_folder_path
            )
            safe_nn_architecture_save(
                nn_architecture_args=self.nn_architecture_args,
                nn_param_folder_name=self.args.nn_trainer_args.neural_network_folder_path,
            )
            safe_nn_trainer_save(
                self.nn_trainer, self.args.nn_trainer_args.neural_network_folder_path
            )
        if (
            self.args.nn_trainer_args.saving_intermediate_copy
            and count_train_step % self.args.saving_intermediate_copy_interval == 0
        ):
            safe_nn_param_save(
                self.nn,
                self.args.nn_trainer_args.neural_network_folder_path,
                training_copy=True,
            )

    def terminate(self) -> None:
        """
        Finishing the script. Profiling or timing.
        """
        pass

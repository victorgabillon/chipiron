import copy
import os
import time
from dataclasses import dataclass, field
from typing import Any

import dacite
from torch.utils.data import DataLoader

from chipiron.learningprocesses.nn_trainer.factory import create_nn_trainer, safe_nn_param_save, safe_nn_trainer_save
from chipiron.players.boardevaluators.datasets.datasets import FenAndValueDataSet
from chipiron.players.boardevaluators.neural_networks import NeuralNetBoardEvalArgs
from chipiron.players.boardevaluators.neural_networks.factory import create_nn
from chipiron.players.boardevaluators.neural_networks.input_converters.factory import Representation364Factory
from chipiron.players.boardevaluators.neural_networks.input_converters.representation_364_bti import \
    Representation364BTI
from chipiron.scripts.script import Script, ScriptArgs


@dataclass
class LearnNNScriptArgs(ScriptArgs):
    neural_network: NeuralNetBoardEvalArgs = field(
        default_factory=lambda: NeuralNetBoardEvalArgs(
            nn_type='pp2d2_2_leaky',
            nn_param_folder_name='foo'
        )
    )
    create_nn_file: bool = True
    config_file_name: str = 'scripts/learn_nn_supervised/exp_options.yaml'
    #  'stockfish_boards_train_file_name': '/home/victor/goodgames_plusvariation_stockfish_eval_train_10000000',
    stockfish_boards_train_file_name: str = 'data/datasets/goodgames_plusvariation_stockfish_eval_train_t.1_merge'
    stockfish_boards_test_file_name: str = 'data/datasets/goodgames_plusvariation_stockfish_eval_test'
    preprocessing_data_set: bool = False
    batch_size_train: int = 32
    batch_size_test: int = 10
    saving_interval: int = 1000
    saving_intermediate_copy: bool = True
    saving_intermediate_copy_interval: int = 10000
    min_interval_lr_change: int = 1000000
    starting_lr: float = .1
    min_lr: float = .001
    momentum_op: float = .9
    scheduler_step_size: float = 1
    scheduler_gamma: float = .5
    reuse_existing_trainer: bool = False


class LearnNNScript:
    """
    Script that learns a NN from a supervised dataset pairs of board and evaluation

    """

    base_experiment_output_folder = os.path.join(Script.base_experiment_output_folder,
                                                 'learn_nn_supervised/learn_nn_supervised_outputs')

    base_script: Script

    def __init__(
            self,
            base_script: Script,
    ) -> None:
        """ Setting up the dataloader from the evaluation files"""

        self.base_script = base_script

        # Calling the init of Script that takes care of a lot of stuff, especially parsing the arguments into self.args
        args_dict: dict[str, Any] = self.base_script.initiate(self.base_experiment_output_folder)

        # Converting the args in the standardized dataclass
        self.args: LearnNNScriptArgs = dacite.from_dict(data_class=LearnNNScriptArgs,
                                                        data=args_dict)

        self.nn = create_nn(args=self.args.neural_network,
                            create_file=self.args.create_nn_file)
        self.nn.print_param()
        self.nn_trainer = create_nn_trainer(args=self.args,
                                            nn=self.nn)

        board_representation_factory = Representation364Factory()
        board_to_input = Representation364BTI(representation_factory=board_representation_factory)

        self.stockfish_boards_train = FenAndValueDataSet(
            file_name=self.args.stockfish_boards_train_file_name,
            preprocessing=self.args.preprocessing_data_set,
            transform_board_function=board_to_input.convert,
            transform_value_function='stockfish')

        self.stockfish_boards_test = FenAndValueDataSet(
            file_name=self.args.stockfish_boards_test_file_name,
            preprocessing=self.args.preprocessing_data_set,
            transform_board_function=board_to_input.convert,
            transform_value_function='stockfish')

        start_time = time.time()
        self.stockfish_boards_train.load()
        print("--- LOADdd %s seconds ---" % (time.time() - start_time))
        self.stockfish_boards_test.load()

        self.data_loader_stockfish_boards_train = DataLoader(self.stockfish_boards_train,
                                                             batch_size=self.args.batch_size_train,
                                                             shuffle=True, num_workers=1)

        self.data_loader_stockfish_boards_test = DataLoader(self.stockfish_boards_test,
                                                            batch_size=self.args.batch_size_test,
                                                            shuffle=True, num_workers=1)

    def run(self):
        """ Running the learning of the NN"""
        print('Starting to learn the NN')
        count_train_step = 0
        sum_loss_train: float = 0.
        sum_loss_train_print: float = 0.
        previous_dict: Any | None = None
        previous_train_loss: float | None = None
        for i in range(100):
            for i_batch, sample_batched in enumerate(self.data_loader_stockfish_boards_train):

                # printing info to console
                if count_train_step % 10000 == 0 and count_train_step > 0:
                    print('count_train_step', count_train_step, 'training loss', sum_loss_train_print / 10000, 'lr',
                          self.nn_trainer.scheduler.get_last_lr())
                    sum_loss_train_print = 0
                    self.compute_test_error()

                # every self.args['min_interval_lr_change'] steps we check for possibly decreasing the learning rate
                if count_train_step % self.args.min_interval_lr_change == 0 and count_train_step > 0:

                    # condition to decrease the learning rate
                    if previous_train_loss is not None and sum_loss_train > previous_train_loss \
                            and self.nn_trainer.scheduler.get_last_lr() > self.args.min_lr:
                        self.nn_trainer.scheduler.step()
                        print('decaying the learning rate to', self.nn_trainer.scheduler.get_last_lr())

                    print('count_train_step', count_train_step)
                    print('training loss', sum_loss_train / self.args.min_interval_lr_change, sum_loss_train)
                    print('previous_train_loss', previous_train_loss)
                    print('learning rate', self.nn_trainer.scheduler.get_last_lr())

                    if previous_dict is not None:
                        diff_weighs = sum(
                            (x - y).abs().sum() for x, y in zip(previous_dict.values(), self.nn.state_dict().values()))
                        print('diff_weighs', diff_weighs)
                    previous_dict = copy.deepcopy(self.nn.state_dict())

                    previous_train_loss = sum_loss_train
                    sum_loss_train = 0.

                # MAIN: the training bit
                count_train_step += 1
                loss_train = self.nn_trainer.train(sample_batched[0], sample_batched[1])
                sum_loss_train += float(loss_train)
                sum_loss_train_print += float(loss_train)

                # saving the learning process
                self.saving_things_to_file(count_train_step)

    def compute_test_error(self):
        sum_loss_test = 0
        count_test = 0
        for i in range(100):
            sample_batched_test = next(iter(self.data_loader_stockfish_boards_test))
            loss_test = self.nn_trainer.test(sample_batched_test[0], sample_batched_test[1])
            sum_loss_test += loss_test
            count_test += 1
        print('test error', float(sum_loss_test / float(count_test)))

    def saving_things_to_file(self, count_train_step):
        if count_train_step % self.args.saving_interval == 0:
            safe_nn_param_save(self.nn, self.args.neural_network)
            safe_nn_trainer_save(self.nn_trainer, self.args.neural_network)
        if self.args.saving_intermediate_copy \
                and count_train_step % self.args.saving_intermediate_copy_interval == 0:
            safe_nn_param_save(self.nn, self.args.neural_network, training_copy=True)

    def terminate(self) -> None:
        """
        Finishing the script. Profiling or timing.
        """
        pass

from src.players.boardevaluators.neural_networks.nn_trainer import NNPytorchTrainer
from src.players.boardevaluators.neural_networks.factory import create_nn
import time
from scripts.script import Script
from src.players.boardevaluators.datasets.datasets import FenAndValueDataSet
from torch.utils.data import DataLoader
import copy


class LearnNNScript(Script):
    """
    Script that learns a NN from a supervised dataset pairs of boards and evaluation

    """
    default_param_dict = Script.default_param_dict | \
                         {'nn_type': 'NetPP2',
                          'config_file_name': 'chipiron/scripts/one_match/exp_options.yaml',
                           }

    def __init__(self):
        super().__init__()
        nn = create_nn(arg=,create_file=)
        nn = create_nn(args['nn_type'], args['nn_weights_path'])
        script = LearnNNScript()

        self.nn = NetPP2D2_2('', 'nn_pp2d2_2/paramTEST.pt')
        self.nn.load_from_file_or_init_weights(authorisation_to_create_file=True)
        self.nn.print_param()
        self.nn_trainer = NNPytorchTrainer(self.nn)
        self.stockfish_boards_train = FenAndValueDataSet(
            file_name='/home/victor/goodgames_plusvariation_stockfish_eval_train_10000000',
            preprocessing=True,
            transform_board_function=self.nn.transform_board_function,
            transform_value_function='stockfish')

        self.stockfish_boards_test = FenAndValueDataSet(
            file_name='/home/victor/goodgames_plusvariation_stockfish_eval_test',
            preprocessing=True,
            transform_board_function=self.nn.transform_board_function,
            transform_value_function='stockfish')

        start_time = time.time()
        self.stockfish_boards_train.load()
        print("--- LOAD %s seconds ---" % (time.time() - start_time))
        self.stockfish_boards_test.load()

        batch_size_train = 32
        self.data_loader_stockfish_boards_train = DataLoader(self.stockfish_boards_train, batch_size=batch_size_train,
                                                             shuffle=True, num_workers=1)
        self.data_loader_stockfish_boards_test = DataLoader(self.stockfish_boards_test, batch_size=10,
                                                            shuffle=True, num_workers=1)

    def run(self):
        print('run')
        count = 0
        big_count = 0
        big_count_loss = 0
        sum_loss = 0
        previous_dict = None
        size_last_losses = 100
        last_train_losses = [0] * size_last_losses
        for i in range(100):
            for i_batch, sample_batched in enumerate(self.data_loader_stockfish_boards_train):
                # print('rrrr',i_batch)
                range_loss = 10000
                if count % range_loss == 0 and count > 0:
                    last_train_losses[big_count % size_last_losses] = sum_loss
                    big_count += 1
                    big_count_loss += 1
                    print('P', count)
                    print('training loss', sum_loss / range_loss, sum_loss)
                    print('last_train_losses', last_train_losses)
                    print('decaying the learning rate to', self.nn_trainer.scheduler.get_last_lr(),
                          big_count_loss > size_last_losses and sum_loss > last_train_losses[
                              (big_count_loss + 1) % size_last_losses], big_count_loss > size_last_losses,
                          sum_loss > last_train_losses[(big_count_loss + 1) % size_last_losses], big_count_loss,
                          size_last_losses, sum_loss, last_train_losses[(big_count_loss + 1) % size_last_losses])

                    if big_count_loss > size_last_losses and sum_loss > last_train_losses[
                        (big_count_loss + 1) % size_last_losses]:
                        self.nn_trainer.scheduler.step()
                        print('decaying the learning rate to', self.nn_trainer.scheduler.get_last_lr())
                        big_count_loss = 0

                    if previous_dict is not None:
                        diff_weigts = sum(
                            (x - y).abs().sum() for x, y in zip(previous_dict.values(), self.nn.state_dict().values()))
                        print('diff_weigts', diff_weigts)
                    previous_dict = copy.deepcopy(self.nn.state_dict())
                    sum_loss = 0
                    sum_loss_test = 0
                    count_test = 0
                    for i in range(100):
                        sample_batched_test = next(iter(self.data_loader_stockfish_boards_test))
                        loss_test = self.nn_trainer.test(sample_batched_test[0], sample_batched_test[1])
                        # print('$$',loss_test)

                        sum_loss_test += loss_test
                        count_test += 1
                    print('test error', float(sum_loss_test / float(count_test)))

                count += 1

                loss = self.nn_trainer.train(sample_batched[0], sample_batched[1])
                #  print('%%%',loss)
                sum_loss += float(loss)

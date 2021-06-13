from src.players.boardevaluators.neural_networks.nn_trainer import NNPytorchTrainer
from src.players.boardevaluators.neural_networks.nn_pp2 import NetPP2
from src.players.boardevaluators.neural_networks.nn_pp2d2 import NetPP2D2

from scripts.script import Script
from src.players.boardevaluators.datatsets.datasets import FenAndValueDataSet
from torch.utils.data import DataLoader
import random
import copy


class LearnNNScript(Script):

    def __init__(self):
        super().__init__()
        self.nn = NetPP2D2('', 'nn_pp2d2/param.pt')
        self.nn.load_or_init_weights()
        self.nn.print_param()
        self.nn_trainer = NNPytorchTrainer(self.nn)
        self.stockfish_boards_train = FenAndValueDataSet(
            file_name='/home/victor/goodgames_plusvariation_stockfish_eval_train',
            preprocessing=True,
            transform_board_function=self.nn.transform_board_function,
            transform_value_function='stockfish')

        self.stockfish_boards_test = FenAndValueDataSet(
            file_name='/home/victor/goodgames_plusvariation_stockfish_eval_test',
            preprocessing=True ,
            transform_board_function=self.nn.transform_board_function,
            transform_value_function='stockfish')

        self.stockfish_boards_train.load()
        self.stockfish_boards_test.load()

        batch_size_train = 124
        self.data_loader_stockfish_boards_train = DataLoader(self.stockfish_boards_train, batch_size=batch_size_train,
                                                       shuffle=True, num_workers=1)
        self.data_loader_stockfish_boards_test = DataLoader(self.stockfish_boards_test, batch_size=1,
                                                       shuffle=False, num_workers=1)

    def run(self):
        print('run')
        count = 0
        sum_loss = 0
        previous_dict = None
        for i in range(1000000):
            for i_batch, sample_batched in enumerate(self.data_loader_stockfish_boards_train):

                range_loss =  10000
                if count % range_loss == 0:
                    print('P', count)
                    print('training loss', sum_loss/range_loss, sum_loss)
                    sum_loss = 0
                    if previous_dict is not None:
                        diff_weigts = sum(
                            (x - y).abs().sum() for x, y in zip(previous_dict.values(), self.nn.state_dict().values()))
                        print('diff_weigts', diff_weigts)
                    previous_dict = copy.deepcopy(self.nn.state_dict())

                    sum_loss_test = 0
                    count_test = 0
                    for i_batch_test, sample_batched_test in enumerate(self.data_loader_stockfish_boards_test):
                        loss_test = self.nn_trainer.test(sample_batched_test[0], sample_batched_test[1])

                        sum_loss_test += loss_test
                        count_test +=1
                    print('test error', float(sum_loss_test/float(count_test)))

                count += 1

                loss = self.nn_trainer.train(sample_batched[0], sample_batched[1])
              #  print('%%%',loss)
                sum_loss += loss






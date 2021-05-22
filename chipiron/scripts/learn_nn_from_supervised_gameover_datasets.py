from src.players.boardevaluators.neural_networks.nn_trainer import NNPytorchTrainer
from src.players.boardevaluators.neural_networks.nn_pp1 import NetPP1
from src.players.boardevaluators.neural_networks.nn_p1 import NetP1
from scripts.script import Script
from src.players.boardevaluators.neural_networks.datasets import NextBoards, ClassifiedBoards,StockfishEvalsBoards
from torch.utils.data import DataLoader
import random


class LearnNNScript(Script):

    def __init__(self):
        super().__init__()
        self.nn = NetPP1('', 'nn_pp1/paramtest2.pt')
        self.nn.load_or_init_weights()
        self.nn.print_param()
        self.nn_trainer = NNPytorchTrainer(self.nn)
      #  self.classified_boards = ClassifiedBoards(self.nn.transform_board_function)
      #  self.next_boards = NextBoards(self.nn.transform_board_function)
        self.stockfish_boards = StockfishEvalsBoards(self.nn.transform_board_function)

      #  self.data_loader_next_boards = DataLoader(self.next_boards, batch_size=1,
        #                                          shuffle=True, num_workers=0)
        self.data_loader_stockfish_boards = DataLoader(self.stockfish_boards, batch_size=512,
                                                  shuffle=True, num_workers=6)
       # self.data_loader_classified_boards = DataLoader(self.classified_boards, batch_size=1,
         #                                               shuffle=True, num_workers=0)

    def run(self):
        print('run')
        count = 0
        sum_loss = 0
        for i in range(1000000):
            # for i_batch, sample_batched in enumerate(self.data_loader_classified_boards):
            # for i_batch, sample_batched in enumerate(
            #         zip(self.data_loader_classified_boards, self.data_loader_next_boards)):
            for i_batch, sample_batched in enumerate(self.data_loader_stockfish_boards):

                if count % 1000 == 0:
                    print('P', count)
                    print('sum_loss', sum_loss)
                    sum_loss = 0
                count += 1

                if random.random() < 2:
                   # print('d', sample_batched)
                  #  print('%%%',sample_batched[1],sample_batched[0])
                    loss = self.nn_trainer.train(sample_batched[0], sample_batched[1])
                    sum_loss += loss

              #  if random.random() < -.001:
               #     self.nn_trainer.train_next_boards(sample_batched[1][0], torch.tensor(sample_batched[1][1]))

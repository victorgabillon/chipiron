from players.boardevaluators.neural_networks.nn_trainer import NNPytorchTrainer
from players.boardevaluators.neural_networks.nn_pp1 import NetPP1
from players.boardevaluators.neural_networks.nn_p1 import NetP1
from players.boardevaluators.neural_networks.nn_p2 import NetP2
from players.boardevaluators.neural_networks.datasets import  NextBoards, ClassifiedBoards
from torch.utils.data import DataLoader
import torch
import random

class LearnNNScript:

    def __init__(self):
        self.nn = NetPP1('', 'nn_pp1/param.pt')
        self.nn.load_or_init_weights()
        self.nn.print_param()
        self.nn_trainer = NNPytorchTrainer(self.nn)
        self.classified_boards = ClassifiedBoards(self.nn.transform_board_function)
        self.next_boards = NextBoards(self.nn.transform_board_function)

        self.data_loader_next_boards = DataLoader(self.next_boards, batch_size=1,
                                                  shuffle=True, num_workers=0)
        self.data_loader_classified_boards = DataLoader(self.classified_boards, batch_size=1,
                                                       shuffle=True, num_workers= 0)

    def run(self):
        print('run')
        count = 0
        for i in range(1000):
            #for i_batch, sample_batched in enumerate(self.data_loader_classified_boards):
            for i_batch, sample_batched in enumerate(zip(self.data_loader_classified_boards, self.data_loader_next_boards)):

                count +=1
                if count %1000 ==0:
                    print('P', count)

                #print('ooo', i_batch)
                #print('sample', sample_batched)
              #  print('samplea', sample_batched[0])
              #  print('sampleas', sample_batched[1])

                if random.random()<2:
                    self.nn_trainer.train(sample_batched[0][0], torch.tensor(sample_batched[0][1]))
                if random.random()<.001:
                    self.nn_trainer.train_next_boards(sample_batched[1][0], torch.tensor(sample_batched[1][1]))

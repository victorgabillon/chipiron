from src.players.boardevaluators.neural_networks.nn_trainer import NNPytorchTrainer
from src.players.boardevaluators.neural_networks.nn_pp1 import NetPP1
from src.players.boardevaluators.neural_networks.datasets import ClassifiedBoards
from torch.utils.data import DataLoader
import torch
import random

class LearnNNExploreScript:

    def __init__(self):
        self.nn = NetPP1('', 'nn_pp1/param_class_learn_full_board_mix.pt')
        self.nn.load_or_init_weights()
        self.nn.print_param()
        self.nn_trainer = NNPytorchTrainer(self.nn)
        self.states = ClassifiedBoards()
        self.data_loader_states = DataLoader(self.states, batch_size=1,
                                                  shuffle=True, num_workers=0)


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

                self.nn_trainer.train(sample_batched[0][0], torch.tensor(sample_batched[0][1]))
                if random.random()<-1:
                    self.nn_trainer.train_next_boards(sample_batched[1][0], torch.tensor(sample_batched[1][1]))

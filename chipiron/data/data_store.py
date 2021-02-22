import pickle
import random
import numpy as np
import pandas as pd

def load_replay_buffer(folder):
    file_name = 'runs/Players/BoardEvaluators/NN1_pytorch/' + folder + '/replay_buffer.rbuf'
    try:
        with open(file_name, 'rb') as fileNNR:
            print('%%%%%%', folder)
            return pickle.load(fileNNR)

    except EnvironmentError:
        return ReplayBuffer(file_name)


class ReplayBufferManager:

    def __init__(self, folder):
        self.folder = folder
        self.replay_buffer = load_replay_buffer(folder)

    def save(self):
        file_name = 'runs/Players/BoardEvaluators/NN1_pytorch/' + self.folder + '/replay_buffer.rbuf'
        try:
            with open(file_name, 'wb') as fileNNW:
                pickle.dump(self.replay_buffer, fileNNW)
        except KeyboardInterrupt:
            with open(file_name, 'wb') as fileNNW:
                pickle.dump(self.replay_buffer, fileNNW)
            exit(-1)


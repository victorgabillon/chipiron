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


class ReplayBuffer:

    def __init__(self, file_):
        self.count = 0
        self.max_count = 100000
        self.buffer = {}
        self.positives = 0
        self.negatives = 0
        self.equalities = 0

    def add_element(self, element):
        if element[1] == 0. and self.negatives < self.positives + 10 and self.negatives < self.equalities + 10:
            self.buffer[self.count] = element
            self.count += 1
            self.count = self.count % self.max_count
        if element[1] == 0.5 and self.equalities < self.positives + 10 and self.equalities < self.negatives + 10:
            self.buffer[self.count] = element
            self.count += 1
            self.count = self.count % self.max_count
        if element[1] == 1 and self.positives< self.negatives + 10 and self.positives < self.equalities + 10:
            self.buffer[self.count] = element
            self.count += 1
            self.count = self.count % self.max_count

    def choose_random_element(self, how_many):
        how_many_real = min(how_many, len(self.buffer))
        return random.choices(list(self.buffer.values()), k=how_many_real)

    def print_info(self):
        print('replay buffer', self.max_count, self.count)
        po = 0
        ne = 0
        eq = 0
        for key, value in self.buffer.items():
            if value[1] == 1:
                po += 1
            if value[1] == 0:
                ne += 1
            if value[1] == .5:
                eq += 1
        self.positives = po
        self.negatives = ne
        self.equalities = eq
        print('po', po, 'ne', ne, 'eq', eq)
        print('poneeq', po + ne + eq)

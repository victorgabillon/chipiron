import random
import torch


class NNPytorchTrainer:

    def __init__(self, net, optimizer, scheduler):
        self.net = net
        self.criterion = torch.nn.L1Loss()
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, input_layer, target_value):
        self.net.train()
        self.optimizer.zero_grad()
        prediction_with_player_to_move_as_white = self.net(input_layer)
        loss = self.criterion(prediction_with_player_to_move_as_white, target_value)
        loss.backward()
        self.optimizer.step()
        return loss

    def test(self, input_layer, target_value):
        self.net.eval()
        prediction_with_player_to_move_as_white = self.net(input_layer)
        loss = self.criterion(prediction_with_player_to_move_as_white, target_value)
        self.net.train()
        return loss

    def train_next_boards(self, input_layer, next_input_layer):
        self.net.eval()
        target_value = - self.net(next_input_layer)

        self.net.train()
        self.optimizer.zero_grad()
        prediction_with_player_to_move_as_white = self.net(input_layer)

        loss = self.criterion(prediction_with_player_to_move_as_white, target_value)
        loss.backward()
        self.optimizer.step()



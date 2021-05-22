import random
import torch
import torch.optim as optim


class NNPytorchTrainer:

    def __init__(self, net):
        self.net = net
        self.criterion = torch.nn.L1Loss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=.0005, momentum=0.9)

    def train(self, input_layer, target_value):
        self.net.train()
       # print('inputlayer',input_layer)
        self.optimizer.zero_grad()


        prediction_with_player_to_move_as_white = self.net(input_layer)
        #print('prediction_with_player_to_move_as_white, target_value',prediction_with_player_to_move_as_white, target_value)
        loss = self.criterion(prediction_with_player_to_move_as_white, target_value)
        #print('loss',loss, type(loss))

        loss.backward()
        self.optimizer.step()

        if random.random() < .001:
          #  print('dinnnff', prediction_with_player_to_move_as_white, target_value, loss)
            self.net.print_param()
            self.net.safe_save()
        return loss

    def train_next_boards(self, input_layer, next_input_layer):

        self.net.eval()
        target_value = - self.net(next_input_layer)

        self.net.train()
        self.optimizer.zero_grad()
        prediction_with_player_to_move_as_white = self.net(input_layer)
        #print('~~#',prediction_with_player_to_move_as_white, target_value)
        #self.net.print_input(input_layer)
        #self.net.print_input(next_input_layer)

        #print('~~s#',input_layer, next_input_layer)
        #print('sdsd',input_layer- next_input_layer)

        loss = self.criterion(prediction_with_player_to_move_as_white, target_value)
        loss.backward()
        self.optimizer.step()

        if random.random() < .001:
            pass
            print('diff', prediction_with_player_to_move_as_white, target_value, loss)
            self.net.print_param()
            self.net.safe_save()

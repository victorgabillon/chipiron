import torch
import torch.nn as nn
import torch.optim as optim
import random
from src.players.boardevaluators.neural_networks.nn_pytorch import BoardNet
from src.players.boardevaluators.neural_networks.board_to_tensor import transform_board_pieces_square


class NetPP1(BoardNet):
    def __init__(self):
        super(BoardNet, self).__init__()

        self.transform_board = transform_board_pieces_square

        self.fc1 = nn.Linear(768, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        return x

    def init_weights(self, file):
        ran = torch.rand(768) * 0.001 + 0.03
        ran = ran.unsqueeze(0)
        self.fc1.weight = torch.nn.Parameter(ran)
        nn.init.constant(self.net.fc1.bias, 0.0)
        torch.save(self.net.state_dict(), file)
        for param in self.net.parameters():
            print(param.data)


class NN4Pytorch:

    def __init__(self, path_to_origin_folder, file_path):
        self.param_file = file_path
        self.path_to_origin_folder = path_to_origin_folder
        self.net = Net()
        # self.jit_net = torch.jit.script(Net())
        self.init_weights()
        self.criterion = torch.nn.L1Loss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=.005, momentum=0.9)

    def init_weights(self):

        try:
            with open(
                    self.path_to_origin_folder + 'chipiron/runs/players/boardevaluators/NN1_pytorch/' + self.param_file,
                    'rb') as fileNNR:
                self.net.load_state_dict(torch.load(fileNNR))

        except EnvironmentError:
            with open('chipiron/runs/players/boardevaluators/NN1_pytorch/' + self.param_file, 'wb') as fileNNW:
                with torch.no_grad():
                    ran = torch.rand(768) * 0.001 + 0.03
                    ran = ran.unsqueeze(0)
                    self.net.fc1.weight = torch.nn.Parameter(ran)
                    nn.init.constant(self.net.fc1.bias, 0.0)
                    torch.save(self.net.state_dict(), fileNNW)
                    for param in self.net.parameters():
                        print(param.data)



    def train_one_example(self, input_layer, target_value, target_input_layer):

        if target_value is None:
            assert (target_input_layer is not None)
            self.net.eval()
            # print('**', target_input_layer)
            real_target_value = 1 - self.net(target_input_layer)
            self.net.train()
        else:
            assert (target_input_layer is None)
            real_target_value = target_value

        self.net.train()
        self.optimizer.zero_grad()
        prediction_with_player_to_move_as_white = self.net(input_layer)
        target_min_max_value_player_to_move = torch.tensor([real_target_value])
        loss = self.criterion(prediction_with_player_to_move_as_white, target_min_max_value_player_to_move)
        loss.backward()
        self.optimizer.step()

        # Save new params
        new_state_dict = {}
        for key in self.net.state_dict():
            new_state_dict[key] = self.net.state_dict()[key].clone()

        # print('after')
        if random.random() < 0.01:
            self.print_param()
        # print('after')

        try:
            with open('chipiron/runs/players/boardevaluators/NN1_pytorch/' + self.param_file, 'wb') as fileNNW:
                # print('ddfff', fileNNW)
                torch.save(self.net.state_dict(), fileNNW)
        except KeyboardInterrupt:
            with open('chipiron/runs/players/boardevaluators/NN1_pytorch/' + self.param_file, 'wb') as fileNNW:
                # print('ddfff', fileNNW)
                torch.save(self.net.state_dict(), fileNNW)
            exit(-1)


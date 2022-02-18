import torch
import torch.nn as nn
from src.players.boardevaluators.neural_networks.nn_pytorch import BoardNet
from src.players.boardevaluators.neural_networks.board_to_tensor import transform_board_pieces_one_side


class NetP1(BoardNet):
    def __init__(self, path_to_main_folder, relative_path_file):
        super(NetP1, self).__init__(path_to_main_folder, relative_path_file)

        self.transform_board_function = transform_board_pieces_one_side
        self.fc1 = nn.Linear(5, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
       # print('weewwwww',x)

        x = self.fc1(x)
        x = self.tanh(x)
     #   print('weew',x)
        return x

    def init_weights(self, file):
        ran = torch.rand(5) * 0.001 + 0.03
        ran = ran.unsqueeze(0)
        self.fc1.weight = torch.nn.Parameter(ran)
        nn.init.constant(self.fc1.bias, 0.0)
        torch.save(self.state_dict(), file)
        for param in self.parameters():
            print(param.data)

    def print_param(self):
        for layer, param in enumerate(self.parameters()):
            if layer == 0:
                print('pawns', param.data[0, 0])
                print('knights', param.data[0, 1])
                print('bishops', param.data[0, 2])
                print('rook', param.data[0, 3])
                print('queen', param.data[0, 4])
            # print('king', param.data[0, 5])
            else:
                print(param.data)

    def print_input(self, input):

        print('pawns', input[0])
        print('knights', input[1])
        print('bishops', input[2])
        print('rook', input[3])
        print('queen', input[4])
    #  print('king', input[5])

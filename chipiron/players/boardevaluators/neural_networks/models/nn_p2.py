import torch
import torch.nn as nn
from chipiron.utils.chi_nn import ChiNN
from chipiron.players.boardevaluators.neural_networks.board_to_tensor import transform_board_pieces_two_sides


class NetP2(ChiNN):
    def __init__(self, path_to_main_folder, relative_path_file):
        super(NetP2, self).__init__()

        self.transform_board_function = transform_board_pieces_two_sides
        self.fc1 = nn.Linear(10, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        return x

    def init_weights(self, file):
        ran = torch.rand(10) * 0.001 + 0.03
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
                print('pawns-opp', param.data[0, 5])
                print('knights-opp', param.data[0, 6])
                print('bishops-opp', param.data[0, 7])
                print('rook-opp', param.data[0, 8])
                print('queen-opp', param.data[0, 9])
            else:
                print(param.data)


def print_input(self, input):
    print('pawns', input[0])
    print('knights', input[1])
    print('bishops', input[2])
    print('rook', input[3])
    print('queen', input[4])
#  print('king', input[5])
    print('pawns-opp', input[5])
    print('knights-opp', input[6])
    print('bishops-opp', input[7])
    print('rook-opp', input[8])
    print('queen-opp', input[9])
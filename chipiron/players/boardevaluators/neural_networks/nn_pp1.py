import torch
import torch.nn as nn
from players.boardevaluators.neural_networks.nn_pytorch import BoardNet
from players.boardevaluators.neural_networks.board_to_tensor import transform_board_pieces_square


class NetPP1(BoardNet):
    def __init__(self, path_to_main_folder, relative_path_file):
        super(NetPP1, self).__init__(path_to_main_folder, relative_path_file)

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

    def print_param(self):
        for layer, param in enumerate(self.parameters()):
            if layer == 0:
                print('pawns', sum(param.data[0, 64 * 0 + 8: 64 * 0 + 64 - 8]) / (64. - 16.))
                print_piece_param(0, param)
                print('knights', sum(param.data[0, 64 * 1: 64 * 1 + 64]) / 64.)
                print_piece_param(1, param)
                print('bishops', sum(param.data[0, 64 * 2: 64 * 2 + 64]) / 64.)
                print_piece_param(2, param)
                print('rook', sum(param.data[0, 64 * 3: 64 * 3 + 64]) / 64.)
                print_piece_param(3, param)
                print('queen', sum(param.data[0, 64 * 4: 64 * 4 + 64]) / 64.)
                print_piece_param(4, param)
                print('king', sum(param.data[0, 64 * 5: 64 * 5 + 64]) / 64.)
                print_piece_param(5, param)
                print('pawns-opposite', sum(param.data[0, 64 * 6 + 8: 64 * 6 + 64 - 8]) / (64. - 16.))
                print_piece_param(6, param)
                print('knights-opposite', sum(param.data[0, 64 * 7: 64 * 7 + 64]) / 64.)
                print_piece_param(7, param)
                print('bishops-opposite', sum(param.data[0, 64 * 8: 64 * 8 + 64]) / 64.)
                print_piece_param(8, param)
                print('rook-opposite', sum(param.data[0, 64 * 9: 64 * 9 + 64]) / 64.)
                print_piece_param(9, param)
                print('queen-opposite', sum(param.data[0, 64 * 10: 64 * 10 + 64]) / 64.)
                print_piece_param(10, param)
                print('king-opposite', sum(param.data[0, 64 * 11: 64 * 11 + 64]) / 64.)
                print_piece_param(11, param)
            else:
                print(param.data)


def print_piece_param(i, param):
    for r in range(8):
        print(param.data[0, 64 * i + 8 * r: 64 * i + 8 * (r + 1)])

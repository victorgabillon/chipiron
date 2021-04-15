import torch
import torch.nn as nn
from src.players.boardevaluators.neural_networks.nn_pytorch import BoardNet
from src.players.boardevaluators.neural_networks.board_to_tensor import transform_board_pieces_square


class NetPP2(BoardNet):
    def __init__(self, path_to_main_folder, relative_path_file):
        super(NetPP2, self).__init__(path_to_main_folder, relative_path_file)

        self.transform_board_function = transform_board_pieces_square
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
        nn.init.constant(self.fc1.bias, 0.0)
        torch.save(self.state_dict(), file)
        for param in self.parameters():
            print(param.data)

    def print_param(self):
        for layer, param in enumerate(self.parameters()):
            if layer == 0:
                print('pawns', sum(param.data[0, 64 * 0 + 8: 64 * 0 + 64 - 8]) / (64. - 16.))
                print_piece_param(0, param.data)
                print('knights', sum(param.data[0, 64 * 1: 64 * 1 + 64]) / 64.)
                print_piece_param(1, param.data)
                print('bishops', sum(param.data[0, 64 * 2: 64 * 2 + 64]) / 64.)
                print_piece_param(2, param.data)
                print('rook', sum(param.data[0, 64 * 3: 64 * 3 + 64]) / 64.)
                print_piece_param(3, param.data)
                print('queen', sum(param.data[0, 64 * 4: 64 * 4 + 64]) / 64.)
                print_piece_param(4, param.data)
                print('king', sum(param.data[0, 64 * 5: 64 * 5 + 64]) / 64.)
                print_piece_param(5, param.data)
                print('pawns-opposite', sum(param.data[0, 64 * 6 + 8: 64 * 6 + 64 - 8]) / (64. - 16.))
                print_piece_param(6, param.data)
                print('knights-opposite', sum(param.data[0, 64 * 7: 64 * 7 + 64]) / 64.)
                print_piece_param(7, param.data)
                print('bishops-opposite', sum(param.data[0, 64 * 8: 64 * 8 + 64]) / 64.)
                print_piece_param(8, param.data)
                print('rook-opposite', sum(param.data[0, 64 * 9: 64 * 9 + 64]) / 64.)
                print_piece_param(9, param.data)
                print('queen-opposite', sum(param.data[0, 64 * 10: 64 * 10 + 64]) / 64.)
                print_piece_param(10, param.data)
                print('king-opposite', sum(param.data[0, 64 * 11: 64 * 11 + 64]) / 64.)
                print_piece_param(11, param.data)
            else:
                print(param.data)

    def print_input(self,input):

                print('pawns', sum(input[0, 64 * 0 + 8: 64 * 0 + 64 - 8]) / (64. - 16.))
                print_piece_param(0, input)
                print('knights', sum(input[0, 64 * 1: 64 * 1 + 64]) / 64.)
                print_piece_param(1, input)
                print('bishops', sum(input[0, 64 * 2: 64 * 2 + 64]) / 64.)
                print_piece_param(2, input)
                print('rook', sum(input[0, 64 * 3: 64 * 3 + 64]) / 64.)
                print_piece_param(3, input)
                print('queen', sum(input[0, 64 * 4: 64 * 4 + 64]) / 64.)
                print_piece_param(4, input)
                print('king', sum(input[0, 64 * 5: 64 * 5 + 64]) / 64.)
                print_piece_param(5, input)
                print('pawns-opposite', sum(input[0, 64 * 6 + 8: 64 * 6 + 64 - 8]) / (64. - 16.))
                print_piece_param(6, input)
                print('knights-opposite', sum(input[0, 64 * 7: 64 * 7 + 64]) / 64.)
                print_piece_param(7, input)
                print('bishops-opposite', sum(input[0, 64 * 8: 64 * 8 + 64]) / 64.)
                print_piece_param(8, input)
                print('rook-opposite', sum(input[0, 64 * 9: 64 * 9 + 64]) / 64.)
                print_piece_param(9, input)
                print('queen-opposite', sum(input[0, 64 * 10: 64 * 10 + 64]) / 64.)
                print_piece_param(10, input)
                print('king-opposite', sum(input[0, 64 * 11: 64 * 11 + 64]) / 64.)
                print_piece_param(11, input)




def print_piece_param(i, vec):
    for r in range(8):
        print(vec[0, 64 * i + 8 * r: 64 * i + 8 * (r + 1)])


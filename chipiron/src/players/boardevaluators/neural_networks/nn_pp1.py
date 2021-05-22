import torch
import torch.nn as nn
from src.players.boardevaluators.neural_networks.nn_pytorch import BoardNet
from src.players.boardevaluators.neural_networks.board_to_tensor import board_to_tensor_pieces_square, \
    node_to_tensors_pieces_square_fast,get_tensor_from_tensors
import chess


class NetPP1(BoardNet):
    def __init__(self, path_to_main_folder, relative_path_file):
        super(NetPP1, self).__init__(path_to_main_folder, relative_path_file)

        self.transform_board_function = board_to_tensor_pieces_square
        self.fc1 = nn.Linear(384 + 2, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        return x

    def init_weights(self, file):
        ran = torch.rand(384 + 2) * 0.001 + 0.03
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
                print('castling', )
                print(param.data[0, 64 * 6:64 * 6 + 2])
            else:
                print(param.data)



    def compute_representation(self, node, parent_node, board_modifications):
        node_to_tensors_pieces_square_fast(node, parent_node, board_modifications, False)

    def get_nn_input(self, node):
        return get_tensor_from_tensors(node.tensor_white, node.tensor_black, node.tensor_castling_white,
                                       node.tensor_castling_black, node.player_to_move)

def print_input(input):

        print('pawns', sum(input[64 * 0 + 8: 64 * 0 + 64 - 8]) / (64. - 16.))
        print_input_param(0, input)
        print('knights', sum(input[64 * 1: 64 * 1 + 64]) / 64.)
        print_input_param(1, input)
        print('bishops', sum(input[64 * 2: 64 * 2 + 64]) / 64.)
        print_input_param(2, input)
        print('rook', sum(input[64 * 3: 64 * 3 + 64]) / 64.)
        print_input_param(3, input)
        print('queen', sum(input[64 * 4: 64 * 4 + 64]) / 64.)
        print_input_param(4, input)
        print('king', sum(input[64 * 5: 64 * 5 + 64]) / 64.)
        print_input_param(5, input)

def print_piece_param(i, vec):
    for r in range(8):
        print(vec[0, 64 * i + 8 * r: 64 * i + 8 * (r + 1)])


def print_input_param(i, vec):
    for r in range(8):
        print(vec[64 * i + 8 * r: 64 * i + 8 * (r + 1)])

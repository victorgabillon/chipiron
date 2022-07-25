import torch.nn as nn
from chipiron.extra_tools.chi_nn import ChiNN
from chipiron.players.boardevaluators.neural_networks.board_to_tensor import board_to_tensor_pieces_square_two_sides, \
    node_to_tensors_pieces_square_fast, get_tensor_from_tensors_two_sides


class NetPP2D2_2_RRELU(ChiNN):
    def __init__(self  ):
        super(NetPP2D2_2_RRELU, self).__init__( )

        self.transform_board_function = board_to_tensor_pieces_square_two_sides
        self.fc1 = nn.Linear(772, 20)
        self.relu_1 = nn.RReLU()
        self.fc2 = nn.Linear(20, 1)
        self.tanh = nn.Tanh()
        #self.dropout = nn.Dropout(.5)

    def forward(self, x):
        x = self.fc1(x)
       # x = self.dropout(self.relu_1(x))
        x = self.relu_1(x)
        x=self.fc2(x)
        x = self.tanh(x)
        return x

    def init_weights(self, file):
        pass

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
                print('castling', param.data[0, 64 * 12: 64 * 12 + 2])
                print('castlingopposite', param.data[0, 64 * 12 + 2: 64 * 12 + 4])
            else:
                print('other layer', layer, param.data)

    def print_input(self, input):

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

    def compute_representation(self, node, parent_node, board_modifications):
        node_to_tensors_pieces_square_fast(node, parent_node, board_modifications, False)

    def get_nn_input(self, node):
        return get_tensor_from_tensors_two_sides(node.tensor_white, node.tensor_black, node.tensor_castling_white,
                                                 node.tensor_castling_black, node.player_to_move)


def print_piece_param(i, vec):
    for r in range(8):
        print(vec[0, 64 * i + 8 * r: 64 * i + 8 * (r + 1)])

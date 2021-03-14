import torch
import torch.nn as nn
import chess
import torch.optim as optim
import random
from numba import jit


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(768, 1)
        self.tanh = nn.Tanh()

    # self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        #      x = self.dropout(x)
        x = self.fc1(x)
        #print('@',x)
        x = self.tanh(x)
        return x


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
            with open(self.path_to_origin_folder+ 'chipiron/runs/players/boardevaluators/NN1_pytorch/' + self.param_file, 'rb') as fileNNR:
                print('%%%%%%', self.param_file)
                self.net.load_state_dict(torch.load(fileNNR))

        except EnvironmentError:
            print('%%%%%%', self.param_file)

            with open('chipiron/runs/players/boardevaluators/NN1_pytorch/' + self.param_file, 'wb') as fileNNW:
                with torch.no_grad():
                    print('!!!!!!!!')
                    ran = torch.rand(768) * 0.001 + 0.03
                    print('££', ran.size())
                    print('$$', ran)
                    ran = ran.unsqueeze(0)
                    self.net.fc1.weight = torch.nn.Parameter(ran)
                    nn.init.constant(self.net.fc1.bias, 0.0)
                    torch.save(self.net.state_dict(), fileNNW)
                    for param in self.net.parameters():
                        print(param.data)

    def value_white(self, board):
        self.net.eval()
        x = transform_board(board, False)
        # print(board.chess_board, board.chess_board.turn)
        #print('555', x,x[64*3:64*4])

        y_pred = self.net(x)
        # y_pred = self.jit_net.forward(x)
        #print('~sd', y_pred, y_pred[0].item(), type(y_pred[0].item()))
        prediction_with_player_to_move_as_white = y_pred[0].item()

        if board.chess_board.turn == chess.BLACK:
            value_white = -prediction_with_player_to_move_as_white
        else:
            value_white = prediction_with_player_to_move_as_white

        #        assert (-1 <= value_white <= 1)
        # print('@@@', value_white)
        # for param in self.net.parameters():
        #   print(param.data)
        #        assert(2==6)
        return value_white

    #
    # def train_one_example2(self, input_layer, target_value_0_1, target_input_layer):
    # #
    #      self.optimizer.zero_grad()
    #      x = torch.rand(10)
    #      y = self.net(input_layer)
    #      loss = self.criterion(y, real_l(input_layer))
    # #
    #      loss.backward()
    #      self.optimizer.step()
    # #
    #      for param in self.net.parameters():
    #          print(param.data)
    #      print('loss', loss)

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

        # print('~~ train')
        self.net.train()
        # assuming white to play assert?

        # assert (0 <= real_target_value_0_1 <= 1)

        self.optimizer.zero_grad()
        prediction_with_player_to_move_as_white = self.net(input_layer)

        #        assert (0 <= prediction_with_player_to_move_as_white_0_1 <= 1)

        target_min_max_value_player_to_move = torch.tensor([real_target_value])
        # print('learning from these values,  target:', target_min_max_value_player_to_move, 'prediction ',
        #      prediction_with_player_to_move_as_white)

        # calculate loss
        loss = self.criterion(prediction_with_player_to_move_as_white, target_min_max_value_player_to_move)

        # print('pl', loss)
        # print('ff', prediction_with_player_to_move_as_white_0_1, target_value_with_player_to_move_as_white,
        #       prediction_with_player_to_move_as_white_0_1 - target_value_with_player_to_move_as_white)

        #  assert(2==3)
        # print('before')
        # for param in self.net.parameters():
        #     print(param.data)
        # print('before')
        #     print(torch.max(param.data))
        #     print(torch.min(param.data))

        # print('updated prediction', prediction_with_player_to_move_as_white, self.net(input_layer))

        # backprop
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

    def print_param(self):
        for layer, param in enumerate(self.net.parameters()):
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


def transform_board_pieces(board, requires_grad_):
    # normalisation of the board so that it is white turn (possible color inversion if it was black's turn)

    if board.chess_board.turn == chess.BLACK:
        color_turn = board.chess_board.turn
        color_not_turn = chess.WHITE
    else:
        color_turn = chess.WHITE
        color_not_turn = chess.BLACK

    transform = torch.zeros(10, requires_grad=requires_grad_)

    # print('ol', board.chessBoard)
    transform[0] = bin(board.chess_board.pawns & board.chess_board.occupied_co[color_turn]).count('1')
    transform[1] = bin(board.chess_board.knights & board.chess_board.occupied_co[color_turn]).count('1')
    transform[2] = bin(board.chess_board.bishops & board.chess_board.occupied_co[color_turn]).count('1')
    transform[3] = bin(board.chess_board.rooks & board.chess_board.occupied_co[color_turn]).count('1')
    transform[4] = bin(board.chess_board.queens & board.chess_board.occupied_co[color_turn]).count('1')
    transform[5] = -bin(board.chess_board.pawns & board.chess_board.occupied_co[color_not_turn]).count('1')
    transform[6] = -bin(board.chess_board.knights & board.chess_board.occupied_co[color_not_turn]).count('1')
    transform[7] = -bin(board.chess_board.bishops & board.chess_board.occupied_co[color_not_turn]).count('1')
    transform[8] = -bin(board.chess_board.rooks & board.chess_board.occupied_co[color_not_turn]).count('1')
    transform[9] = -bin(board.chess_board.queens & board.chess_board.occupied_co[color_not_turn]).count('1')
    return transform


def transform_board(board, requires_grad_):
    # normalisation of the board so that it is white turn (possible color inversion if it was black's turn)
    inversion = 1
    if board.chess_board.turn == chess.BLACK:
        inversion = -1

    transform = torch.zeros(768, requires_grad=requires_grad_)

    for square in range(64):
        piece = board.chess_board.piece_at(square)
        if piece:
            # print('p', square, piece.color, type(piece.piece_type))
            piece_code = 6 * (piece.color != board.chess_board.turn) + (piece.piece_type - 1)
            # print('dp', 64 * piece_code + square, 2 * piece.color - 1)
            if piece.color == chess.BLACK:
                square_index = chess.square_mirror(square)
            else:
                square_index = square
            index = 64 * piece_code + square_index
            transform[index] = (2 * piece.color - 1) * inversion
        # transform[64 * piece_code + square] = 2 * piece.color - 1
    return transform


# def real_f(x):
#     return torch.sigmoid(x[0] * .7 + x[1] * .2 + x[2] * .3 + x[3] * .4 + x[4] * .9
#                          + x[5] * .7 + x[6] * .2 + x[7] * .3 + x[8] * .4 + x[9] * .9)
#
#
# def real_l(x):
#     return x[0] * 1 + x[1] * 3 + x[2] * 3 + x[3] * 5 + x[4] * 9 + x[5] * 1 + x[6] * 3 + x[7] * 3 + x[8] * 5 + x[9] * 9
#

if __name__ == "__main__":
    nn = NN4Pytorch()

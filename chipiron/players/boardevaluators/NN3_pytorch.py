import yaml
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import torch.optim as optim
import settings
from random import random


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        #      x = self.dropout(x)
        x = self.fc1(x)
        x = self.sigmoid(x)
        return x


class NN3Pytorch:

    def __init__(self, folder):
        self.folder = folder
        self.net = Net()
        # self.jit_net = torch.jit.script(Net())
        self.init_weights()

        if settings.learning_nn_bool:
            self.criterion = torch.nn.L1Loss()
            self.optimizer = optim.SGD(self.net.parameters(), lr=.001, momentum=0.9)
        else:
            # torch.no_grad()
            assert (3 == 4)

    def init_weights(self):

        try:
            with open('runs/players/boardevaluators/NN1_pytorch/' + self.folder + '/param.pt', 'rb') as fileNNR:
                print('%%%%%%', self.folder)
                self.net.load_state_dict(torch.load(fileNNR))

        except EnvironmentError:
            with open('runs/players/boardevaluators/NN1_pytorch/' + self.folder + '/param.pt', 'wb') as fileNNW:
                with torch.no_grad():
                    print('!!!!!!!!')
                    ran = torch.rand(768) * 0.1 + 0.03
                    print('££',ran.size())
                    print('$$', ran)
                    ran = ran.unsqueeze(0)
                    self.net.fc1.weight = torch.nn.Parameter(ran)
                    nn.init.constant(self.net.fc1.bias, 0.0)
                    torch.save(self.net.state_dict(), fileNNW)
                    for param in self.net.parameters():
                        print(param.data)

    def value_white(self, board):
        self.net.eval()
        x = transform_board(board)
        #  print('555',x)
        y_pred = self.net(x)
        # y_pred = self.jit_net.forward(x)
        # print('~', y_pred, y_pred[0].item(), type(y_pred[0].item()))
        prediction_with_player_to_move_as_white = 2 * y_pred[0].item() - 1

        if board.chess_board.turn == chess.BLACK:
            value_white = -prediction_with_player_to_move_as_white
        else:
            value_white = prediction_with_player_to_move_as_white

        #        assert (-1 <= value_white <= 1)
        # print('@@@', value_white)
        # for param in self.net.parameters():
        #     print(param.data)
        return value_white

    def train_one_example(self, board, min_max_value_white, over_event):

        print('~~ train')
        self.net.train()
        # assuming white to play assert?
        x = transform_board(board)

        # print('x', x[45:55])
        # print('##', min_max_value_white)
        assert (-1 <= min_max_value_white <= 1)

        self.optimizer.zero_grad()
        prediction_with_player_to_move_as_white_0_1 = self.net(x)

        assert (0 <= prediction_with_player_to_move_as_white_0_1 <= 1)

        target_value_with_player_to_move_as_white = torch.zeros(1)
        if board.chess_board.turn == chess.BLACK:
            target_value_with_player_to_move_as_white[0] = -min_max_value_white / 2. + .5
        if board.chess_board.turn == chess.WHITE:
            target_value_with_player_to_move_as_white[0] = min_max_value_white / 2. + .5
        print('learning from these values', min_max_value_white, prediction_with_player_to_move_as_white_0_1,
              target_value_with_player_to_move_as_white[0])

        # calculate loss
        loss = self.criterion(prediction_with_player_to_move_as_white_0_1, target_value_with_player_to_move_as_white)
        print('pl', loss)
        # print('ff', prediction_with_player_to_move_as_white_0_1, target_value_with_player_to_move_as_white,
        #       prediction_with_player_to_move_as_white_0_1 - target_value_with_player_to_move_as_white)

        for param in self.net.parameters():
            print(param.data)
            print(torch.max(param.data))
            print(torch.min(param.data))

        print('updated prediction', prediction_with_player_to_move_as_white_0_1, self.net(x))

        # Save init values
        old_state_dict = {}
        for key in self.net.state_dict():
            old_state_dict[key] = self.net.state_dict()[key].clone()

        # backprop
        loss.backward()
        self.optimizer.step()

        # Save new params
        new_state_dict = {}
        for key in self.net.state_dict():
            new_state_dict[key] = self.net.state_dict()[key].clone()

        # Compare params
        for key in old_state_dict:
            if not (old_state_dict[key] == new_state_dict[key]).all():
                print('Diff in {}'.format(key))
        with open('runs/players/boardevaluators/NN1_pytorch/' + self.folder + '/param.pt', 'wb') as fileNNW:
            print('ddfff', fileNNW)
            torch.save(self.net.state_dict(), fileNNW)


def transform_board(board):
    # normalisation of the board so that it is white turn (possible color inversion if it was black's turn)
    inversion = 1
    if board.chess_board.turn == chess.BLACK:
        inversion = -1

    if settings.learning_nn_bool:
        transform = torch.zeros(768)
    else:
        # transform = torch.zeros(768, requires_grad=False)
        assert (3 == 4)

    # print('ol', board.chessBoard)
    for square in range(64):
        piece = board.chess_board.piece_at(square)
        if piece:
            # print('p', square, piece.color, piece.piece_type)
            piece_code = 6 * piece.color_player + (piece.piece_type - 1)
            # print('dp', 64 * piece_code + square, 2 * piece.color - 1)
            transform[64 * piece_code + square] = (2 * piece.color_player - 1) * inversion
        # transform[64 * piece_code + square] = 2 * piece.color - 1
    return transform

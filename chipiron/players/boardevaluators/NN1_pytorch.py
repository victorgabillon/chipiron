import yaml
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import torch.optim as optim
import settings


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(768, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 1)

    def forward(self, x):
        # for param in self.parameters():
        #     print(';;',param.data)
        # print('w#', x)
        x = F.relu(self.fc1(x))
        # print('#c',x)
        x = F.relu(self.fc2(x))
        #  print('s#',x)
        x = self.fc3(x)
        #  print('#',x)
        return torch.sigmoid(x)


class NN1Pytorch:

    def __init__(self, folder):
        self.folder = folder
        self.net = Net()
        # self.jit_net = torch.jit.script(Net())
        self.init_weights()

        if settings.learning_nn_bool:
            self.criterion = nn.BCEWithLogitsLoss()
            self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        else:
            torch.no_grad()



    def init_weights(self):

        with open(r'runs/players/boardevaluators/NN1_pytorch/' + self.folder + '/hyper_param.yaml') as fileParam:
            # print(fileParam.read())
            hyper_param = yaml.load(fileParam, Loader=yaml.FullLoader)
        layer_sizes = hyper_param['layer_sizes']
        print(layer_sizes)

        try:
            with open('runs/players/boardevaluators/NN1_pytorch/' + self.folder + '/param.pt', 'rb') as fileNNR:
                self.net.load_state_dict(torch.load(fileNNR))
                self.net.eval()

        except EnvironmentError:
            with open('runs/players/boardevaluators/NN1_pytorch/' + self.folder + '/param.pt', 'wb') as fileNNW:
                torch.save(self.net.state_dict(), fileNNW)

    def value_white(self, board):
        x = transform_board(board)
        #  print('555',x)
        y_pred = self.net(x)
        # y_pred = self.jit_net.forward(x)
        #print('~', y_pred, y_pred[0].item(), type(y_pred[0].item()))
        prediction_with_player_to_move_as_white = 2*y_pred[0].item()-1

        if board.chess_board.turn == chess.BLACK:
            value_white = -prediction_with_player_to_move_as_white
        else:
            value_white = prediction_with_player_to_move_as_white

        assert(-1<=value_white<=1)

        return value_white

    def train_one_example(self, board, min_max_value_white, over_event):
        # assuming white to play assert?
        x = transform_board(board)
        print('##',min_max_value_white)
        assert(-1<=min_max_value_white<=1)
        prediction_with_player_to_move_as_white = self.net(x)
        assert (0 <= prediction_with_player_to_move_as_white <= 1)
        prediction_with_player_to_move_as_white = 2*prediction_with_player_to_move_as_white-1


        print('opopoop',min_max_value_white,prediction_with_player_to_move_as_white)
        target_value_with_player_to_move_as_white = torch.zeros(1)
        if board.chess_board.turn == chess.BLACK:
            target_value_with_player_to_move_as_white[0] = -min_max_value_white
        if board.chess_board.turn == chess.WHITE:
            target_value_with_player_to_move_as_white[0] = min_max_value_white

        # calculate loss
        loss = self.criterion(prediction_with_player_to_move_as_white, target_value_with_player_to_move_as_white)
        print('pl',loss)
        print('ff',prediction_with_player_to_move_as_white, target_value_with_player_to_move_as_white,prediction_with_player_to_move_as_white- target_value_with_player_to_move_as_white)

        # backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        with open('runs/players/boardevaluators/NN1_pytorch/' + self.folder + '/param.pt', 'wb') as fileNNW:
            torch.save(self.net.state_dict(), fileNNW)

def transform_board(board):
    # normalisation of the board so that it is white turn (possible color inversion if it was black's turn)
    inversion = 1
    if board.chess_board.turn == chess.BLACK:
        inversion = -1

    if settings.learning_nn_bool:
        transform = torch.zeros(768)
    else:
        transform = torch.zeros(768, requires_grad=False)

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

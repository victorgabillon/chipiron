
import torch
import torch.nn as nn
import chess
import torch.optim as optim
import settings


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


class NN4Pytorch:

    def __init__(self, folder):
        self.folder = folder
        self.net = Net()
        # self.jit_net = torch.jit.script(Net())
        self.init_weights()

        if settings.learning_nn_bool:
            self.criterion = torch.nn.L1Loss()
            self.optimizer = optim.SGD(self.net.parameters(), lr=.001, momentum=0.9)
        else:
            pass
            # torch.no_grad()
        #  assert (3 == 4)

    def init_weights(self):

        try:
            with open('chipiron/runs/players/boardevaluators/NN1_pytorch/' + self.folder + '/paramtest.pt', 'rb') as fileNNR:
                print('%%%%%%', self.folder)
                self.net.load_state_dict(torch.load(fileNNR))

        except EnvironmentError:
            with open('chipiron/runs/players/boardevaluators/NN1_pytorch/' + self.folder + '/paramtest.pt', 'wb') as fileNNW:
                with torch.no_grad():
                    print('!!!!!!!!')
                    ran = torch.rand(10) * 0.001 + 0.03
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
        x = transform_board(board)
        #print(board.chess_board, board.chess_board.turn)
        #print('555', x)

        y_pred = self.net(x)
        # y_pred = self.jit_net.forward(x)
        # print('~', y_pred, y_pred[0].item(), type(y_pred[0].item()))
        prediction_with_player_to_move_as_white = 2 * y_pred[0].item() - 1

        if board.chess_board.turn == chess.BLACK:
            value_white = -prediction_with_player_to_move_as_white
        else:
            value_white = prediction_with_player_to_move_as_white

        #        assert (-1 <= value_white <= 1)
        #print('@@@', value_white)
        #for param in self.net.parameters():
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

    def train_one_example(self, input_layer, target_value_0_1, target_input_layer):

        if target_value_0_1 is None:
            assert (target_input_layer is not None)
            real_target_value_0_1 = -self.net(target_input_layer)

        else:
            assert (target_input_layer is None)
            real_target_value_0_1 = target_value_0_1

        print('~~ train')
        self.net.train()
        # assuming white to play assert?

        #        assert (0 <= real_target_value_0_1 <= 1)

        self.optimizer.zero_grad()
        prediction_with_player_to_move_as_white_0_1 = self.net(input_layer)

        #        assert (0 <= prediction_with_player_to_move_as_white_0_1 <= 1)

        target_min_max_value_player_to_move_0_1 = torch.tensor([real_target_value_0_1])
        print('learning from these values,  target:', target_min_max_value_player_to_move_0_1, 'prediction ',
              prediction_with_player_to_move_as_white_0_1)

        # calculate loss
        loss = self.criterion(prediction_with_player_to_move_as_white_0_1, target_min_max_value_player_to_move_0_1)

        #loss = self.criterion(prediction_with_player_to_move_as_white_0_1, real_l(input_layer))

        print('pl', loss)
        # print('ff', prediction_with_player_to_move_as_white_0_1, target_value_with_player_to_move_as_white,
        #       prediction_with_player_to_move_as_white_0_1 - target_value_with_player_to_move_as_white)

        #  assert(2==3)
        print('before')
        for param in self.net.parameters():
            print(param.data)
        print('before')
        #     print(torch.max(param.data))
        #     print(torch.min(param.data))

        print('updated prediction', prediction_with_player_to_move_as_white_0_1, self.net(input_layer))

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

        print('after')
        for param in self.net.parameters():
            print(param.data)
        print('after')

        # Compare params
        for key in old_state_dict:
            if not (old_state_dict[key] == new_state_dict[key]).all():
                print('Diff in {}'.format(key))
        with open('runs/players/boardevaluators/NN1_pytorch/' + self.folder + '/paramtest.pt', 'wb') as fileNNW:
            print('ddfff', fileNNW)
            torch.save(self.net.state_dict(), fileNNW)


def transform_board(board):
    # normalisation of the board so that it is white turn (possible color inversion if it was black's turn)

    if board.chess_board.turn == chess.BLACK:
        color_turn = board.chess_board.turn
        color_not_turn = chess.WHITE
    else:
        color_turn = chess.WHITE
        color_not_turn = chess.BLACK

    if settings.learning_nn_bool:
        transform = torch.zeros(10)
    else:
        transform = torch.zeros(10, requires_grad=False)
        # assert (3 == 4)

    # print('ol', board.chessBoard)
    transform[0] = bin(board.chess_board.pawns & board.chess_board.occupied_co[color_turn]).count('1')
    transform[1] = bin(board.chess_board.knights & board.chess_board.occupied_co[color_turn]).count('1')
    transform[2] = bin(board.chess_board.bishops & board.chess_board.occupied_co[color_turn]).count('1')
    transform[3] = bin(board.chess_board.rooks & board.chess_board.occupied_co[color_turn]).count('1')
    transform[4] = bin(board.chess_board.queens & board.chess_board.occupied_co[color_turn]).count('1')
    transform[5] = - bin(board.chess_board.pawns & board.chess_board.occupied_co[color_not_turn]).count('1')
    transform[6] = -bin(board.chess_board.knights & board.chess_board.occupied_co[color_not_turn]).count('1')
    transform[7] = -bin(board.chess_board.bishops & board.chess_board.occupied_co[color_not_turn]).count('1')
    transform[8] = -bin(board.chess_board.rooks & board.chess_board.occupied_co[color_not_turn]).count('1')
    transform[9] = -bin(board.chess_board.queens & board.chess_board.occupied_co[color_not_turn]).count('1')
    return transform


def real_f(x):
    return torch.sigmoid(x[0] * .7 + x[1] * .2 + x[2] * .3 + x[3] * .4 + x[4] * .9
                         + x[5] * .7 + x[6] * .2 + x[7] * .3 + x[8] * .4 + x[9] * .9)


def real_l(x):
    return x[0] * 1 + x[1] * 3 + x[2] * 3 + x[3] * 5 + x[4] * 9 + x[5] * 1 + x[6] * 3 + x[7] * 3 + x[8] * 5 + x[9] * 9



if __name__ == "__main__":
    nn = NN4Pytorch()
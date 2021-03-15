import torch
import torch.nn as nn
import chess
import random


class BoardNet(nn.Module):
    def __init__(self, path_to_main_folder, relative_path_file):
        super(BoardNet, self).__init__()
        self.path_to_param_file = path_to_main_folder + 'chipiron/runs/players/boardevaluators/NN1_pytorch/' \
                                  + relative_path_file

    def load_or_init_weights(self):
        try:  # load
            with open(self.path_to_param_file, 'rb') as fileNNR:
                self.load_state_dict(torch.load(fileNNR))

        except EnvironmentError:  # init
            with open(self.path_to_param_file, 'wb') as fileNNW:
                with torch.no_grad():
                    self.init_weights(fileNNW)

    def value_white(self, board):
        self.eval()
        x = self.transform_board(board, False)

        y_pred = self(x)
        prediction_with_player_to_move_as_white = y_pred[0].item()

        if board.chess_board.turn == chess.BLACK:
            value_white = -prediction_with_player_to_move_as_white
        else:
            value_white = prediction_with_player_to_move_as_white

        self.print_param()
        return value_white



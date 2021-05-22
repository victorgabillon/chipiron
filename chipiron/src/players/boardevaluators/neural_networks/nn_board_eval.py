import torch
import chess
from src.players.boardevaluators.neural_networks.nn_pp1 import NetPP1


class NNBoardEval:
    def __init__(self, arg):
        if arg['subtype'] == 'pp1':
            self.net = NetPP1('', arg['nn_param_file_name'])
            self.net.load_or_init_weights()
            self.my_scripted_model = torch.jit.script(self.net)

    def compute_representation(self, node, parent_node, board_modifications):
        self.net.compute_representation(node, parent_node, board_modifications)

    def value_white(self, node):
        self.my_scripted_model.eval()
        x = self.net.get_nn_input(node)
        torch.no_grad()
        y_pred = self.my_scripted_model(x)
       # y_pred = self.net(x)

        torch.no_grad()
        prediction_with_player_to_move_as_white = y_pred[0].item()

        if node.board.chess_board.turn == chess.BLACK:
            value_white = -prediction_with_player_to_move_as_white
        else:
            value_white = prediction_with_player_to_move_as_white

        # print(board,value_white)
        # self.print_param()
        return value_white

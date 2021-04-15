import chess
from src.players.boardevaluators.syzygy import Syzygy


class BoardEvaluatorsWrapper:
    # VALUE_WHITE_WHEN_OVER is the value_white default value when the node is over
    # set atm to be symmetric and high be be preferred
    VALUE_WHITE_WHEN_OVER = [VALUE_WHITE_WHEN_OVER_WHITE_WINS, VALUE_WHITE_WHEN_OVER_DRAW,
                             VALUE_WHITE_WHEN_OVER_BLACK_WINS, ] = [1000, 0, -1000]

    def __init__(self, board_evaluator, syzygy):
        self.board_evaluator = board_evaluator
        self.syzygy_evaluator = syzygy
        assert (isinstance(syzygy, Syzygy))

    def value_white(self, node):
        value_white = self.syzygy_value_white(node.board)
        if value_white is None:
            value_white = self.board_evaluator.value_white(node)
        return value_white

    def compute_representation(self, node, parent_node, board_modifications):
        self.board_evaluator.compute_representation(node, parent_node, board_modifications)

    def syzygy_value_white(self, board):
        if self.syzygy_evaluator is None or not self.syzygy_evaluator.fast_in_table(board):
            return None
        else:
            print('deed', self.syzygy_evaluator.fast_in_table(board))
            return self.syzygy_evaluator.value_white(board)

    def check_obvious_over_events(self, node):
        """ updates the node.over object
         if the game is obviously over"""
        game_over = node.board.chess_board.is_game_over()
        if game_over:
            value_as_string = node.board.chess_board.result()
            if value_as_string == '0-1':
                how_over_ = node.over_event.WIN
                who_is_winner_ = chess.BLACK
            elif value_as_string == '1-0':
                how_over_ = node.over_event.WIN
                who_is_winner_ = chess.WHITE
            elif value_as_string == '1/2-1/2':
                how_over_ = node.over_event.DRAW
                who_is_winner_ = node.over_event.NO_KNOWN_WINNER
            node.over_event.becomes_over(how_over=how_over_,
                                         who_is_winner=who_is_winner_)

        elif self.syzygy_evaluator and self.syzygy_evaluator.fast_in_table(node.board):
            print('@@', node.board)
            self.syzygy_evaluator.set_over_event(node)

    def value_white_from_over_event(self, over_event):
        assert over_event.is_over()
        if over_event.is_win():
            assert (not over_event.is_draw())
            if over_event.is_winner(chess.WHITE):
                return self.VALUE_WHITE_WHEN_OVER_WHITE_WINS
            else:
                return self.VALUE_WHITE_WHEN_OVER_BLACK_WINS
        else:  # draw
            assert (over_event.is_draw())
            return self.VALUE_WHITE_WHEN_OVER_DRAW

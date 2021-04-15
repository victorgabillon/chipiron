import chess.syzygy
from src.players.boardevaluators.over_event import OverEvent


class Syzygy:

    def __init__(self, chess_simulator, path_to_chipiron):
        self.chessSimulator = chess_simulator
        self.table_base = chess.syzygy.open_tablebase(path_to_chipiron + "chipiron/syzygy-tables/")

    def fast_in_table(self, board):
        return board.number_of_pieces_on_the_board() < 6

    def in_table(self, board):
        try:
            self.table_base.probe_wdl(board.chess_board)
        except KeyError:
            return False
        return True

    def set_over_event(self, node):
        val = self.val(node.board)

        who_is_winner_ = node.over_event.NO_KNOWN_WINNER
        if val != 0:
            how_over_ = node.over_event.WIN
            if val > 0:
                who_is_winner_ = node.player_to_move
            if val < 0:
                who_is_winner_ = chess.WHITE if node.player_to_move == chess.BLACK else chess.BLACK
        else:
            how_over_ = node.over_event.DRAW

        node.over_event.becomes_over(how_over=how_over_,
                                     who_is_winner=who_is_winner_)

    def val(self, board):
        # tablebase.probe_wdl Returns 2 if the side to move is winning, 0 if the position is a draw and -2 if the side to move is losing.
        val = self.table_base.probe_wdl(board.chess_board)
        return val

    def get_over_tag(self, board):
        val = self.table_base.probe_wdl(board.chess_board)
        if val > 0:
            if board.chess_board.turn == chess.WHITE:
                return OverEvent.TAG_WIN_WHITE
            else:
                return OverEvent.TAG_WIN_BLACK
        elif val == 0:
            return OverEvent.TAG_DRAW
        else:
            if board.chess_board.turn == chess.WHITE:
                return OverEvent.TAG_WIN_BLACK
            else:
                return OverEvent.TAG_WIN_WHITE

    def sting_result(self, board):
        val = self.table_base.probe_wdl(board.chess_board)
        player_to_move = 'white' if board.chess_board.turn == chess.WHITE else 'black'
        if val > 0:
            return 'WIN for player ' + player_to_move
        elif val == 0:
            return 'DRAW'
        else:
            return 'LOSS for player ' + player_to_move

    def dtz(self, board):
        dtz = self.table_base.probe_dtz(board.chess_board)
        return dtz

    def best_move(self, board):
        all_moves = list(board.get_legal_moves())

        # avoid draws by 50 move rules in winning position, # otherwise look
        # for it to make it last and preserve pieces in case of mistake by opponent

        best_value = -1000000000000000000000
        best_move = None
        for move in all_moves:
            next_board = self.chessSimulator.step_create(board, move, 0)
            val_player_next_board = self.val(next_board)
            val_player_node = -val_player_next_board
            dtz_player_next_board = self.dtz(next_board)
            dtz_player_node = -dtz_player_next_board
            if val_player_node > 0:  # winning position
                new_value = board.chess_board.is_zeroing(move) * 100 - dtz_player_node + 1000
            elif val_player_node == 0:
                new_value = - board.chess_board.is_zeroing(move) * 100 + dtz_player_node
            elif val_player_node < 0:
                new_value = - board.chess_board.is_zeroing(move) * 100 + dtz_player_node - 1000
            # print('edeswswswaqq',str(move),val_player_node,new_value ,node.board.chess_board.is_zeroing(move) ,self.dtz(next_board))

            if new_value > best_value:
                best_value = new_value
                best_move = move
        return best_move

    #   val = self.tablebase.probe_dtz(board.chessBoard)

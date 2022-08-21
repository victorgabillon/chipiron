import chess.engine
import chipiron as ch


class Stockfish:

    def __init__(self, options):
        self.engine = chess.engine.SimpleEngine.popen_uci("/home/victor/stockfish_13_linux_x64/stockfish_13_linux_x64")
        self.depth = options['depth']

    def select_move(self, board: ch.chess.board.BoardChi):
        result = self.engine.play(board, chess.engine.Limit(depth=self.depth))
        return result.move

    def print_info(self):
        super().print_info()
        print('type: Stockfish')

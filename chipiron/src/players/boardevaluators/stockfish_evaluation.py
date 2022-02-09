import chess.engine


class Stockfish:

    def __init__(self):
        self.engine = chess.engine.SimpleEngine.popen_uci("chipiron/stockfish/stockfish_13_linux_x64")
        self.depth = 20

    def value_white(self, node):
        info = self.engine.analyse(node, chess.engine.Limit(time=0.1))
        return info["score"]

    def score(self, board):
        info = self.engine.analyse(board, chess.engine.Limit(time=0.1))
        return info["score"]

    def compute_representation(self, node,parent_node,board_modifications):
        pass

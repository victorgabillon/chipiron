import chess
import chess.svg






class DisplayBoards:





    def display(self,board):
            self.chessboardSvg = chess.svg.board(board).encode("UTF-8")
            self.widgetSvg.load(self.chessboardSvg)
import chess
import chess.svg

from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import QApplication, QWidget







class DisplayBoards:





    def display(self,board):
            self.chessboardSvg = chess.svg.board(board).encode("UTF-8")
            self.widgetSvg.load(self.chessboardSvg)
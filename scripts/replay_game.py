import pickle
import sys
from PyQt5.QtWidgets import *
from src.displays import MainWindow
from scripts.script import Script


class ReplayGameScript(Script):

    def __init__(self):
        super().__init__()

    def run(self):
        with open('chipiron/runs/match_outputs/Friday-04-16-2021--17:09:50:556501/games/Game2_W:ZipfSequool-vs-B:RecurZipfBase.obj',
                  'rb') as fileGame:
            chess_board = pickle.load(fileGame)

        chessGui = QApplication(sys.argv)
        window = MainWindow(chess_board)
        window.show()
        chessGui.exec_()

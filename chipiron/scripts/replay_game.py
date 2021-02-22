import pickle
import sys
from PyQt5.QtWidgets import *
from displays.gui_replay_games import MainWindow

with open('runs/runsOutput/Saturday-02-13-2021--20:56:43:932784/games/Game3_W:ZipfSequool-vs-B:RecurZipf.obj',
          'rb') as fileGame:
    chess_board = pickle.load(fileGame)

chessGui = QApplication(sys.argv)
window = MainWindow(chess_board)
window.show()
chessGui.exec_()

import pickle
import sys

from PySide6.QtWidgets import QApplication

from chipiron.displays.gui_replay_games import MainWindow
from chipiron.scripts.script import Script


class ReplayGameScript(Script):

    def __init__(self):
        pass

    def run(self):
        with open(
                'scripts/one_match/outputs/Wednesday-01-31-2024--21:03:54:357326/games/_0_W:Sequool-vs-B:SequoolOnlyPromosing.obj',
                'rb') as fileGame:
            chess_board = pickle.load(fileGame)

        chessGui = QApplication(sys.argv)
        window = MainWindow(chess_board)
        window.show()
        chessGui.exec_()

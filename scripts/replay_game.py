import pickle
import sys
from PySide6.QtWidgets import *
from chipiron.displays.gui_replay_games import MainWindow
from scripts.script import Script
import os

class ReplayGameScript(Script):

    def __init__(self):
        base_experiment_output_folder = None
        base_script: Script

    def run(self):
        with open('scripts/one_match/outputs/Wednesday-01-31-2024--21:03:54:357326/games/_0_W:Sequool-vs-B:SequoolOnlyPromosing.obj',
                  'rb') as fileGame:
            chess_board = pickle.load(fileGame)

        chessGui = QApplication(sys.argv)
        window = MainWindow(chess_board)
        window.show()
        chessGui.exec_()

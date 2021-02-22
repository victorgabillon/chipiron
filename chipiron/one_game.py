
import yaml
import time
import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import chess
from games.play_one_game import PlayOneGame
from players.create_player import create_player
from chessenvironment.chess_environment import ChessEnvironment
from displays.gui import MainWindow
import cProfile, pstats, io
from pstats import SortKey

start_time = time.time()

#pr = cProfile.Profile()
#pr.enable()

with open(r'runs/GameSettings/setting_navo.yaml') as fileGame:
    argsGame = yaml.load(fileGame, Loader=yaml.FullLoader)
    print(argsGame)

with open(r'runs/players/Human.yaml') as filePlayerWhite:
    argsPlayerWhite = yaml.load(filePlayerWhite, Loader=yaml.FullLoader)
    print(argsPlayerWhite)

#with open(r'runs/players/RandomExplorer.yaml') as filePlayerBlack:
with open(r'runs/players/RecurZipf.yaml') as filePlayerBlack:
    argsPlayerBlack = yaml.load(filePlayerBlack, Loader=yaml.FullLoader)
    print(argsPlayerBlack)

chessSimulator = ChessEnvironment()
playerWhite = create_player(argsPlayerWhite, chessSimulator)

playerBlack = create_player(argsPlayerBlack, chessSimulator)
play = PlayOneGame(argsGame, playerWhite, playerBlack, chessSimulator)



#play.play()


chessGui = QApplication(sys.argv)
window = MainWindow(play)
window.show()
chessGui.exec_()



print("--- %s seconds ---" % (time.time() - start_time))

#pr.disable()
#s = io.StringIO()
#sortby = SortKey.CUMULATIVE
#ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
#ps.print_stats()
#print(s.getvalue())
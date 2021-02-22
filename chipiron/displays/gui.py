#! /usr/bin/env python

"""
This module is the execution point of the chess GUI application.
"""

import sys

import chess
import threading

import settings
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import QApplication, QWidget

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import time


class Worker(QRunnable):
    '''
    Worker thread
    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''
        self.fn(*self.args, **self.kwargs)


class MainWindow(QWidget):
    """
    Create a surface for the chessboard.
    """

    def __init__(self, play):
        """
        Initialize the chessboard.
        """
        super().__init__()

        self.play = play
        self.setWindowTitle("Chess GUI")
        self.setGeometry(300, 300, 800, 800)

        self.widgetSvg = QSvgWidget(parent=self)
        self.widgetSvg.setGeometry(10, 10, 600, 600)

        self.closeButton = QPushButton(self)
        self.closeButton.setText("Close")  # text
        self.closeButton.setIcon(QIcon("close.png"))  # icon
        self.closeButton.setShortcut('Ctrl+D')  # shortcut key
        self.closeButton.clicked.connect(self.stopppy)
        self.closeButton.setToolTip("Close the widget")  # Tool tip
        self.closeButton.move(700, 100)

        self.closeButton2 = QPushButton(self)
        self.closeButton2.setText("Player")  # text
        # self.closeButton2.move(700, 200)
        self.closeButton2.setStyleSheet('QPushButton {background-color: white; color: blue;}')
        self.closeButton2.setGeometry(650, 200, 150, 20)

        self.closeButton3 = QPushButton(self)
        self.closeButton3.setText("Player")  # text
        self.closeButton3.setStyleSheet('QPushButton {background-color: black; color: blue;}')
        self.closeButton3.setGeometry(650, 300, 150, 20)

        self.closeButton4 = QPushButton(self)
        self.closeButton4.setText("Score 0-0")  # text
        self.closeButton4.setStyleSheet('QPushButton {background-color: black; color: blue;}')
        self.closeButton4.setGeometry(650, 400, 150, 20)

        self.boardSize = min(self.widgetSvg.width(),
                             self.widgetSvg.height())
        self.coordinates = True
        self.margin = 0.05 * self.boardSize if self.coordinates else 0
        self.squareSize = (self.boardSize - 2 * self.margin) / 8.0
        self.pieceToMove = [None, None]

        # print('333')
        self.threadpool = QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())
        self.startPlayThread()
        time.sleep(.05)
        # self.board = self.play.play_one_game.game.board.chessBoard

        self.checkThreadTimer = QTimer(self)
        self.checkThreadTimer.setInterval(5)  # .5 seconds
        self.checkThreadTimer.timeout.connect(self.drawBoard)
        self.checkThreadTimer.start()

        self.whiteishuman = self.play.play_one_game.player_white.player_name == 'Human'
        self.blackishuman = self.play.play_one_game.player_black.player_name == 'Human'

    def stopppy(self):
        self.threadpool.shutdown(self.worker)
        self.close()

    @pyqtSlot(QWidget)
    def mousePressEvent(self, event):
        """
        Handle left mouse clicks and enable moving chess pieces by
        clicking on a chess piece and then the target square.

        Moves must be made according to the rules of chess because
        illegal moves are suppressed.
        """

        white_plays = self.play.play_one_game.game.who_plays()
        humanwhitetoplay = white_plays and self.whiteishuman
        humanblacktoplay = not white_plays and self.blackishuman
        #  print('eee',white_plays,self.whiteishuman, self.blackishuman,humanwhitetoplay ,  humanblacktoplay)

        if humanblacktoplay or humanwhitetoplay:
            if white_plays:
                player = self.play.play_one_game.player_white
            else:
                player = self.play.play_one_game.player_black

            if event.x() <= self.boardSize and event.y() <= self.boardSize:
                if event.buttons() == Qt.LeftButton:

                    if self.margin < event.x() < self.boardSize - self.margin and self.margin < event.y() < self.boardSize - self.margin:

                        file = int((event.x() - self.margin) / self.squareSize)
                        rank = 7 - int((event.y() - self.margin) / self.squareSize)
                        square = chess.square(file, rank)
                        piece = self.board.piece_at(square)
                        self.coordinates = "{}{}".format(chr(file + 97), str(rank + 1))
                        # print('uuu',file,rank,square,piece,coordinates,self.pieceToMove[0])
                        if self.pieceToMove[0] is not None:
                            try:
                                move = chess.Move.from_uci("{}{}".format(self.pieceToMove[1], self.coordinates))
                                move_promote = chess.Move.from_uci("{}{}q".format(self.pieceToMove[1], self.coordinates))
                                settings.global_lock.acquire()
                                try:
                                    if move in self.board.legal_moves:
                                        # self.board.push(move)
                                        self.play.play_one_game.game.play(move)
                                        self.drawBoard()
                                        player.human_played = True
                                    if move_promote in self.board.legal_moves:
                                        self.choicePromote()
                                        self.play.play_one_game.game.play(self.move_promote_asked)
                                        self.drawBoard()
                                        player.human_played = True
                                        print(move)
                                    else:
                                        print('Looks like a wrong move.', move, self.board.legal_moves)
                                finally:
                                    if settings.global_lock.locked():
                                        settings.global_lock.release()
                            except ValueError:
                                print("Oops!  Doubleclicked?  Try again...")
                            piece = None
                            coordinates = None
                        self.pieceToMove = [piece, self.coordinates]

    def choicePromote(self):

        self.d = QDialog()
        d= self.d
        d.setWindowTitle("Promote to ?")
        d.setWindowModality(Qt.ApplicationModal)


        d.closeButtonQ = QPushButton(d)
        d.closeButtonQ.setText("Queen")  # text
        d.closeButtonQ.setStyleSheet('QPushButton {background-color: white; color: blue;}')
        d.closeButtonQ.setGeometry(150, 100, 150, 20)
        d.closeButtonQ.clicked.connect(self.promoteQueen)

        d.closeButtonR = QPushButton(d)
        d.closeButtonR.setText("Rook")  # text
        d.closeButtonR.setStyleSheet('QPushButton {background-color: white; color: blue;}')
        d.closeButtonR.setGeometry(150, 200, 150, 20)
        d.closeButtonR.clicked.connect(self.promoteRook)

        d.closeButtonB = QPushButton(d)
        d.closeButtonB.setText("Bishop")  # text
        d.closeButtonB.setStyleSheet('QPushButton {background-color: white; color: blue;}')
        d.closeButtonB.setGeometry(150, 300, 150, 20)
        d.closeButtonB.clicked.connect(self.promoteBishop)

        d.closeButtonK = QPushButton(d)
        d.closeButtonK.setText("Knight")  # text
        d.closeButtonK.setStyleSheet('QPushButton {background-color: white; color: blue;}')
        d.closeButtonK.setGeometry(150, 400, 150, 20)
        d.closeButtonK.clicked.connect(self.promoteKnight)

        d.exec_()

    def promoteQueen(self):
       self.move_promote_asked =  chess.Move.from_uci("{}{}q".format(self.pieceToMove[1], self.coordinates))
       self.d.close()

    def promoteRook(self):
       self.move_promote_asked =  chess.Move.from_uci("{}{}r".format(self.pieceToMove[1], self.coordinates))
       self.d.close()

    def promoteBishop(self):
       self.move_promote_asked =  chess.Move.from_uci("{}{}b".format(self.pieceToMove[1], self.coordinates))
       self.d.close()

    def promoteKnight(self):
       self.move_promote_asked =  chess.Move.from_uci("{}{}n".format(self.pieceToMove[1], self.coordinates))
       self.d.close()


    def goplay(self):
        self.play.play_the_match()
    
    def startPlayThread(self):
        print('startPlayThread()')
        self.worker = Worker(self.goplay)
        self.threadpool.start(self.worker)

    def computerMove(self):
        print('computerMove')
        worker = Worker()
        self.threadpool.start(worker)

    def drawBoard(self):
        """
        Draw a chessboard with the starting position and then redraw
        it for every new move.
        """
        if not settings.global_lock.locked():
            settings.global_lock.acquire()
            try:
                self.board = self.play.play_one_game.game.board.chess_board
                self.boardSvg = self.board._repr_svg_().encode("UTF-8")
                self.drawBoardSvg = self.widgetSvg.load(self.boardSvg)

                self.closeButton2.setText('White: ' + self.play.play_one_game.player_white.player_name)  # text
                self.closeButton3.setText('Black: ' + self.play.play_one_game.player_black.player_name)  # text
                self.closeButton4.setText(
                    'Score: ' + str(self.play.match_results.get_player_one_wins()) + '-'
                    + str(self.play.match_results.get_player_two_wins()) + '-'
                    + str(self.play.match_results.get_draws()))  # text


            finally:
                if settings.global_lock.locked():
                    settings.global_lock.release()
            return self.drawBoardSvg


if __name__ == "__main__":
    chessGui = QApplication(sys.argv)
    window = MainWindow(None)
    window.show()
    sys.exit(chessGui.exec_())

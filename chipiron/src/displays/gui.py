#! /usr/bin/env python

"""
This module is the execution point of the chess GUI application.
"""

import chess
from PyQt5.QtCore import Qt
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


class MainWindow(QWidget):
    """
    Create a surface for the chessboard.
    """

    def __init__(self, gui_mailbox, main_thread_mailbox):
        """
        Initialize the chessboard.
        """
        super().__init__()
        self.gui_mailbox = gui_mailbox
        self.main_thread_mailbox = main_thread_mailbox

        self.setWindowTitle("Chess GUI")
        self.setGeometry(300, 300, 1400, 800)

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
        self.closeButton2.setStyleSheet('QPushButton {background-color: white; color: blue;}')
        self.closeButton2.setGeometry(620, 200, 370, 30)

        self.closeButton3 = QPushButton(self)
        self.closeButton3.setText("Player")  # text
        self.closeButton3.setStyleSheet('QPushButton {background-color: black; color: blue;}')
        self.closeButton3.setGeometry(620, 300, 370, 30)

        self.closeButton4 = QPushButton(self)
        self.closeButton4.setText("Score 0-0")  # text
        self.closeButton4.setStyleSheet('QPushButton {background-color: white; color: black;}')
        self.closeButton4.setGeometry(620, 400, 370, 30)

        self.closeButton5 = QPushButton(self)
        self.closeButton5.setText("Round")  # text
        self.closeButton5.setStyleSheet('QPushButton {background-color: white; color: black;}')
        self.closeButton5.setGeometry(620, 500, 370, 30)

        self.closeButton6 = QPushButton(self)
        self.closeButton6.setText("fen")  # text
        self.closeButton6.setStyleSheet('QPushButton {background-color: white; color: black;}')
        self.closeButton6.setGeometry(50, 700, 850, 30)

        self.closeButton7 = QPushButton(self)
        self.closeButton7.setText("Stock Eval")  # text
        self.closeButton7.setStyleSheet('QPushButton {background-color: white; color: black;}')
        self.closeButton7.setGeometry(620, 600, 370, 30)

        self.board_size = min(self.widgetSvg.width(),
                              self.widgetSvg.height())
        self.coordinates = True
        self.margin = 0.05 * self.board_size if self.coordinates else 0
        self.squareSize = (self.board_size - 2 * self.margin) / 8.0
        self.pieceToMove = [None, None]

        self.checkThreadTimer = QTimer(self)
        self.checkThreadTimer.setInterval(5)  # .5 seconds
        self.checkThreadTimer.timeout.connect(self.process_message)
        self.checkThreadTimer.start()

    def stopppy(self):
        # should we send a kill message to the main thread?
        self.close()

    @pyqtSlot(QWidget)
    def mousePressEvent(self, event):
        """
        Handle left mouse clicks and enable moving chess pieces by
        clicking on a chess piece and then the target square.

        Moves must be made according to the rules of chess because
        illegal moves are suppressed.
        """
        if event.x() <= self.board_size and event.y() <= self.board_size:
            if event.buttons() == Qt.LeftButton:

                if self.margin < event.x() < self.board_size - self.margin \
                        and self.margin < event.y() < self.board_size - self.margin:

                    file = int((event.x() - self.margin) / self.squareSize)
                    rank = 7 - int((event.y() - self.margin) / self.squareSize)
                    square = chess.square(file, rank)
                    piece = self.board.piece_at(square)
                    self.coordinates = "{}{}".format(chr(file + 97), str(rank + 1))
                    if self.pieceToMove[0] is not None:
                        try:
                            move = chess.Move.from_uci("{}{}".format(self.pieceToMove[1], self.coordinates))
                            move_promote = chess.Move.from_uci("{}{}q".format(self.pieceToMove[1], self.coordinates))
                            if move in self.board.legal_moves:
                                self.send_move_to_main_thread(self.move)
                            elif move_promote in self.board.legal_moves:
                                self.choice_promote()
                                self.send_move_to_main_thread(self.move_promote_asked)
                            else:
                                print('Looks like a wrong move.', move, self.board.legal_moves)
                        except ValueError:
                            print("Oops!  Doubleclicked?  Try again...")
                        piece = None
                    self.pieceToMove = [piece, self.coordinates]

    def send_move_to_main_thread(self, move):
        self.main_thread_mailbox.put({'type': 'move', 'move': move})

    def choice_promote(self):
        self.d = QDialog()
        d = self.d
        d.setWindowTitle("Promote to ?")
        d.setWindowModality(Qt.ApplicationModal)

        d.closeButtonQ = QPushButton(d)
        d.closeButtonQ.setText("Queen")  # text
        d.closeButtonQ.setStyleSheet('QPushButton {background-color: white; color: blue;}')
        d.closeButtonQ.setGeometry(150, 100, 150, 20)
        d.closeButtonQ.clicked.connect(self.promote_queen)

        d.closeButtonR = QPushButton(d)
        d.closeButtonR.setText("Rook")  # text
        d.closeButtonR.setStyleSheet('QPushButton {background-color: white; color: blue;}')
        d.closeButtonR.setGeometry(150, 200, 150, 20)
        d.closeButtonR.clicked.connect(self.promote_rook)

        d.closeButtonB = QPushButton(d)
        d.closeButtonB.setText("Bishop")  # text
        d.closeButtonB.setStyleSheet('QPushButton {background-color: white; color: blue;}')
        d.closeButtonB.setGeometry(150, 300, 150, 20)
        d.closeButtonB.clicked.connect(self.promote_bishop)

        d.closeButtonK = QPushButton(d)
        d.closeButtonK.setText("Knight")  # text
        d.closeButtonK.setStyleSheet('QPushButton {background-color: white; color: blue;}')
        d.closeButtonK.setGeometry(150, 400, 150, 20)
        d.closeButtonK.clicked.connect(self.promote_knight)

        d.exec_()

    def promote_queen(self):
        self.move_promote_asked = chess.Move.from_uci("{}{}q".format(self.pieceToMove[1], self.coordinates))
        self.d.close()

    def promote_rook(self):
        self.move_promote_asked = chess.Move.from_uci("{}{}r".format(self.pieceToMove[1], self.coordinates))
        self.d.close()

    def promote_bishop(self):
        self.move_promote_asked = chess.Move.from_uci("{}{}b".format(self.pieceToMove[1], self.coordinates))
        self.d.close()

    def promote_knight(self):
        self.move_promote_asked = chess.Move.from_uci("{}{}n".format(self.pieceToMove[1], self.coordinates))
        self.d.close()

    def process_message(self):
        """
        Draw a chessboard with the starting position and then redraw
        it for every new move.
        """

        if not self.gui_mailbox.empty():
            message = self.gui_mailbox.get()

            if message['type'] == 'board':
                self.board = message['board']
                self.draw_board()
            if message['type'] == 'evaluation':
                evaluation = message['evaluation']
                self.update_evaluation(evaluation)
            if message['type'] == 'players_color_to_id':
                players_color_to_id = message['players_color_to_id']
                self.update_players_color_to_id(players_color_to_id)
            if message['type'] == 'match_results':
                match_results = message['match_results']
                self.update_match_stats(match_results)

    def draw_board(self):
        """
        Draw a chessboard with the starting position and then redraw
        it for every new move.
        """

        self.boardSvg = self.board._repr_svg_().encode("UTF-8")
        self.drawBoardSvg = self.widgetSvg.load(self.boardSvg)
        self.closeButton5.setText('Round: ' + str(self.board.fullmove_number))  # text
        self.closeButton6.setText('fen: ' + str(self.board.fen()))  # text
        return self.drawBoardSvg

    def update_players_color_to_id(self, players_color_to_id):
        self.closeButton2.setText('White: ' + players_color_to_id[chess.WHITE])  # text
        self.closeButton3.setText('Black: ' + players_color_to_id[chess.BLACK])  # text

    def update_evaluation(self, evaluation):
        self.closeButton7.setText('eval: ' + str(evaluation))  # text

    def update_match_stats(self, match_info):
        player_one_wins, player_two_wins, draws = match_info.get_simple_result()
        self.closeButton4.setText(
            'Score: ' + str(player_one_wins) + '-'
            + str(player_two_wins) + '-'
            + str(draws))  # text



#! /usr/bin/env python

"""
This module is the execution point of the chess GUI application.
"""

import chess
from PySide6.QtCore import Qt
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from chipiron.games.game.game_playing_status import PlayingStatus
from chipiron.environments.chess.board import BoardChi
from chipiron.utils.communication.gui_messages import GameStatusMessage, BackMessage, EvaluationMessage, \
    MatchResultsMessage
from chipiron.utils.communication.gui_player_message import PlayersColorToPlayerMessage
from chipiron.utils.communication.player_game_messages import BoardMessage, MoveMessage
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import pyqtgraph as pg


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


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
        self.closeButton.move(700, 0)

        # self.play_button = QPushButton(self)
        # self.play_button.setText("Play")  # text
        # self.play_button.setIcon(QIcon("data/gui/play.png"))  # icon
        # self.play_button.clicked.connect(self.play_button_clicked)
        # self.play_button.setToolTip("play the game")  # Tool tip
        # self.play_button.move(700, 100)

        self.pause_button = QPushButton(self)
        self.pause_button.setText("Pause")  # text
        self.pause_button.setIcon(QIcon("data/gui/pause.png"))  # icon
        self.pause_button.clicked.connect(self.pause_button_clicked)
        self.pause_button.setToolTip("pause the game")  # Tool tip
        self.pause_button.move(850, 100)

        self.back_button = QPushButton(self)
        self.back_button.setText("Back")  # text
        self.back_button.setIcon(QIcon("data/gui/back.png"))  # icon
        self.back_button.clicked.connect(self.back_button_clicked)
        self.back_button.setToolTip("back one move")  # Tool tip
        self.back_button.move(1000, 100)

        self.player_white_button = QPushButton(self)
        self.player_white_button.setText("Player")  # text
        self.player_white_button.setIcon(QIcon("data/gui/white_king.png"))  # icon
        self.player_white_button.setStyleSheet('QPushButton {background-color: white; color: black;}')
        self.player_white_button.setGeometry(620, 250, 370, 30)

        self.player_black_button = QPushButton(self)
        self.player_black_button.setText("Player")  # text
        self.player_black_button.setIcon(QIcon("data/gui/black_king.png"))  # icon
        self.player_black_button.setStyleSheet('QPushButton {background-color: black; color: white;}')
        self.player_black_button.setGeometry(620, 300, 370, 30)

        self.tablewidget = QTableWidget(1, 2, self)
        self.tablewidget.setGeometry(1100, 250, 260, 330)

        self.score_button = QPushButton(self)
        self.score_button.setText("Score 0-0")  # text
        self.score_button.setStyleSheet('QPushButton {background-color: white; color: black;}')
        self.score_button.setGeometry(620, 400, 370, 30)

        self.round_button = QPushButton(self)
        self.round_button.setText("Round")  # text
        self.round_button.setStyleSheet('QPushButton {background-color: white; color: black;}')
        self.round_button.setGeometry(620, 500, 370, 30)

        self.fen_button = QPushButton(self)
        self.fen_button.setText("fen")  # text
        self.fen_button.setStyleSheet('QPushButton { background-color: white; color: black;}')
        self.fen_button.setGeometry(50, 700, 1250, 30)

        self.eval_button = QPushButton(self)
        self.eval_button.setText("Stock Eval")  # text
        self.eval_button.setStyleSheet('QPushButton {background-color: white; color: black;}')
        self.eval_button.setGeometry(620, 600, 470, 30)

        self.eval_button_chi = QPushButton(self)
        self.eval_button_chi.setText("Chi Eval")  # text
        self.eval_button_chi.setStyleSheet('QPushButton {background-color: white; color: black;}')
        self.eval_button_chi.setGeometry(620, 650, 470, 30)

        self.eval_button_white = QPushButton(self)
        self.eval_button_white.setText("White Eval")  # text
        self.eval_button_white.setStyleSheet('QPushButton {background-color: white; color: black;}')
        self.eval_button_white.setGeometry(620, 700, 470, 30)

        self.eval_button_black = QPushButton(self)
        self.eval_button_black.setText("Black Eval")  # text
        self.eval_button_black.setStyleSheet('QPushButton {background-color: white; color: black;}')
        self.eval_button_black.setGeometry(620, 750, 470, 30)

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

    def stopppy(self) -> None:
        # should we send a kill message to the main thread?
        self.close()

    def play_button_clicked(self) -> None:
        message: GameStatusMessage = GameStatusMessage(status=PlayingStatus.PLAY)
        self.main_thread_mailbox.put(message)

    def back_button_clicked(self) -> None:
        message: BackMessage = BackMessage()
        self.main_thread_mailbox.put(message)

    def pause_button_clicked(self) -> None:
        message: GameStatusMessage = GameStatusMessage(status=PlayingStatus.PAUSE)
        self.main_thread_mailbox.put(message)

    @Slot(QWidget)
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
                    self.coordinates = f'{chr(file + 97)}{rank + 1}'
                    if self.pieceToMove[0] is not None:
                        try:
                            move = chess.Move.from_uci("{}{}".format(self.pieceToMove[1], self.coordinates))
                            move_promote = chess.Move.from_uci("{}{}q".format(self.pieceToMove[1], self.coordinates))
                            if move in self.board.legal_moves:
                                self.send_move_to_main_thread(move)
                            elif move_promote in self.board.legal_moves:
                                self.choice_promote()
                                self.send_move_to_main_thread(self.move_promote_asked)
                            else:
                                print(f'Looks like a wrong move. {move} {self.board.legal_moves}')
                        except ValueError:
                            print("Oops!  Doubleclicked?  Try again...")
                        piece = None
                    self.pieceToMove = [piece, self.coordinates]

    def send_move_to_main_thread(self, move):
        print('move', type(move), move)
        message: MoveMessage = MoveMessage(move=move, corresponding_board=self.board.fen(), player_name='Human')
        self.main_thread_mailbox.put(item=message)

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
            match message:
                case BoardMessage():
                    message: BoardMessage
                    self.board: BoardChi = message.board
                    self.draw_board()
                    self.display_move_history()
                case EvaluationMessage():
                    message: EvaluationMessage
                    evaluation_stock = message.evaluation_stock
                    evaluation_chipiron = message.evaluation_chipiron
                    evaluation_black = message.evaluation_player_black
                    evaluation_white = message.evaluation_player_white
                    self.update_evaluation(evaluation_stock=evaluation_stock,
                                           evaluation_chipiron=evaluation_chipiron,
                                           evaluation_white=evaluation_white,
                                           evaluation_black=evaluation_black)
                case PlayersColorToPlayerMessage():
                    message: PlayersColorToPlayerMessage
                    players_color_to_player: dict = message.player_color_to_gui_info
                    self.update_players_color_to_id(players_color_to_player)
                case MatchResultsMessage():
                    message: MatchResultsMessage
                    match_results = message.match_results
                    self.update_match_stats(match_results)
                case GameStatusMessage():
                    message: GameStatusMessage
                    play_status: PlayingStatus = message.status
                    self.update_game_play_status(play_status)
                case other:
                    raise ValueError(f'unknown type of message received by gui {other} in {__name__}')

    def display_move_history(self):
        import math
        num_half_move: int = len(self.board.move_stack)
        num_rounds: int = int(math.ceil(num_half_move / 2))
        self.tablewidget.setRowCount(num_rounds)
        self.tablewidget.setHorizontalHeaderLabels(['White', 'Black'])
        for player in range(2):
            for round_ in range(num_rounds):
                half_move = round_ * 2 + player
                if half_move < num_half_move:
                    item = QTableWidgetItem(str(self.board.move_stack[half_move]))
                    self.tablewidget.setItem(round_, player, item)

    def draw_board(self):
        """
        Draw a chessboard with the starting position and then redraw
        it for every new move.
        """

        self.boardSvg = self.board._repr_svg_().encode("UTF-8")
        self.drawBoardSvg = self.widgetSvg.load(self.boardSvg)
        self.round_button.setText('Round: ' + str(self.board.fullmove_number))  # text
        self.fen_button.setText('fen: ' + str(self.board.fen()))  # text
        return self.drawBoardSvg

    def update_players_color_to_id(self, players_color_to_player: dict[chess.COLORS, str]):
        self.player_white_button.setText(' White: ' + players_color_to_player[chess.WHITE])  # text
        self.player_black_button.setText(' Black: ' + players_color_to_player[chess.BLACK])  # text

    def update_evaluation(self, evaluation_stock, evaluation_chipiron, evaluation_white,
                          evaluation_black):
        print('rgrgr', evaluation_stock, evaluation_chipiron, evaluation_white,evaluation_black)
        self.eval_button.setText('eval: ' + str(evaluation_stock))  # text
        self.eval_button_chi.setText('eval: ' + str(evaluation_chipiron))  # text
        self.eval_button_black.setText('eval: ' + str(evaluation_white))  # text
        self.eval_button_white.setText('eval: ' + str(evaluation_black))  # text
    def update_game_play_status(self, play_status: PlayingStatus):
        if play_status == PlayingStatus.PAUSE:
            self.pause_button.setText("Play")  # text
            self.pause_button.setIcon(QIcon("data/gui/play.png"))  # icon
            self.pause_button.clicked.connect(self.play_button_clicked)
            self.pause_button.setToolTip("play the game")  # Tool tip
            self.pause_button.move(850, 100)
        elif play_status == PlayingStatus.PLAY:
            self.pause_button.setText("Pause")  # text
            self.pause_button.setIcon(QIcon("data/gui/pause.png"))  # icon
            self.pause_button.clicked.connect(self.pause_button_clicked)
            self.pause_button.setToolTip("pause the game")  # Tool tip
            self.pause_button.move(850, 100)

    def update_match_stats(self, match_info):
        player_one_wins, player_two_wins, draws = match_info.get_simple_result()
        self.score_button.setText(
            'Score: ' + str(player_one_wins) + '-'
            + str(player_two_wins) + '-'
            + str(draws))  # text

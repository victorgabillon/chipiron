#! /usr/bin/env python

"""
This module is the execution point of the chess GUI application.

It provides the `MainWindow` class, which creates a surface for the chessboard and handles user interactions.
"""

import math
import os
import queue
import time
import typing

import chess
import chess.svg
import PySide6.QtGui as QtGui
from atomheart.board import BoardFactory, IBoard, create_board_chi
from atomheart.board.utils import FenPlusHistory
from atomheart.move import MoveUci
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QIcon
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)
from valanga import Color

from chipiron.displays.gui_protocol import (
    CmdBackOneMove,
    CmdHumanMoveUci,
    CmdSetStatus,
    GuiCommand,
    GuiUpdate,
    PlayerUiInfo,
    Scope,
    UpdEvaluation,
    UpdGameStatus,
    UpdMatchResults,
    UpdPlayerProgress,
    UpdPlayersInfo,
    UpdStateChess,
)
from chipiron.games.game.game_playing_status import PlayingStatus
from chipiron.utils.dataclass import IsDataclass
from chipiron.utils.logger import chipiron_logger
from chipiron.utils.path_variables import GUI_DIR

if typing.TYPE_CHECKING:
    from atomheart.move.imove import MoveKey


class MainWindow(QWidget):
    """
    Create a surface for the chessboard and handle user interactions.

    This class provides the main window for the chess GUI application. It handles user interactions such as
    clicking on chess pieces, making moves, and controlling the game status.

    Attributes:
        playing_status (PlayingStatus): The current playing status of the game.
        gui_mailbox (queue.Queue[IsDataclass]): The mailbox for receiving messages from the GUI thread.
        main_thread_mailbox (queue.Queue[IsDataclass]): The mailbox for sending messages to the main thread.
    """

    def __init__(
        self,
        gui_mailbox: queue.Queue[GuiUpdate],
        main_thread_mailbox: queue.Queue[IsDataclass],
        board_factory: BoardFactory,
    ) -> None:
        """
        Initialize the chessboard and the main window.

        Args:
            gui_mailbox (queue.Queue[IsDataclass]): The mailbox for receiving messages from the GUI thread.
            main_thread_mailbox (queue.Queue[IsDataclass]): The mailbox for sending messages to the main thread.
        """
        super().__init__()

        self.play_button_clicked_last_time: float | None = None
        self.pause_button_clicked_last_time: float | None = None

        self.board_factory = board_factory
        self.playing_status = PlayingStatus.PLAY

        self.gui_mailbox = gui_mailbox
        self.main_thread_mailbox = main_thread_mailbox

        self.scope: Scope | None = None

        # Set window icon with existence check
        window_icon_path = os.path.join(GUI_DIR, "chipicon.png")
        if os.path.exists(window_icon_path):
            self.setWindowIcon(QIcon(window_icon_path))
        else:
            chipiron_logger.warning("Window icon file not found: %s", window_icon_path)

        self.setWindowTitle("🐙 ♛  Chipiron Chess GUI  ♛ 🐙")
        self.setGeometry(300, 300, 1400, 800)

        self.widgetSvg = QSvgWidget(parent=self)
        self.widgetSvg.setGeometry(10, 10, 600, 600)

        self.closeButton = QPushButton(self)
        self.closeButton.setText("Close")  # text
        self._check_and_set_icon(
            self.closeButton, os.path.join(GUI_DIR, "close.png")
        )  # icon
        self.closeButton.setShortcut("Ctrl+D")  # shortcut key
        self.closeButton.clicked.connect(self.stopppy)
        self.closeButton.setToolTip("Close the widget")  # Tool tip
        self.closeButton.move(800, 20)

        self.pause_button = QPushButton(self)
        self.pause_button.setText("Pause")  # text
        self._check_and_set_icon(
            self.pause_button, os.path.join(GUI_DIR, "pause.png")
        )  # icon
        self.pause_button.clicked.connect(self.pause_button_clicked)
        self.pause_button.setToolTip("pause the game")  # Tool tip
        self.pause_button.move(700, 100)

        self.back_button = QPushButton(self)
        self.back_button.setText("Back")  # text
        self._check_and_set_icon(
            self.back_button, os.path.join(GUI_DIR, "back.png")
        )  # icon
        self.back_button.clicked.connect(self.back_button_clicked)
        self.back_button.setToolTip("back one move")  # Tool tip
        self.back_button.move(900, 100)

        self.player_white_button = QPushButton(self)
        self.player_white_button.setText("Player")  # text
        self._check_and_set_icon(
            self.player_white_button, os.path.join(GUI_DIR, "white_king.png")
        )  # icon
        self.player_white_button.setStyleSheet(
            "QPushButton {background-color: white; color: black;}"
        )
        self.player_white_button.setGeometry(620, 200, 470, 30)

        self.progress_white = QProgressBar(self)
        self.progress_white.setGeometry(620, 230, 470, 30)

        self.player_black_button = QPushButton(self)
        self.player_black_button.setText("Player")  # text
        self._check_and_set_icon(
            self.player_black_button, os.path.join(GUI_DIR, "black_king.png")
        )  # icon
        self.player_black_button.setStyleSheet(
            "QPushButton {background-color: black; color: white;}"
        )
        self.player_black_button.setGeometry(620, 300, 470, 30)

        self.progress_black = QProgressBar(self)
        self.progress_black.setGeometry(620, 330, 470, 30)

        self.tablewidget = QTableWidget(1, 2, self)
        self.tablewidget.setGeometry(1100, 200, 260, 330)

        self.score_button = QPushButton(self)
        self.score_button.setText("⚖ Score 0-0")  # text
        self.score_button.setStyleSheet(
            "QPushButton {background-color: white; color: black;}"
        )
        self.score_button.setGeometry(620, 400, 370, 30)

        self.round_button = QPushButton(self)
        self.round_button.setText("🎲 Round")  # text
        self.round_button.setStyleSheet(
            "QPushButton {background-color: white; color: black;}"
        )
        self.round_button.setGeometry(620, 500, 370, 30)

        self.fen_button = QLabel(self)
        self.fen_button.setText("fen")  # text
        self.fen_button.setStyleSheet(
            "QPushButton { background-color: white; color: black;}"
        )
        self.fen_button.setGeometry(20, 620, 550, 30)
        self.fen_button.setTextInteractionFlags(
            QtGui.Qt.TextInteractionFlag.TextSelectableByMouse
        )

        self.legal_moves_button = QLabel(self)
        self.legal_moves_button.setText("legal moves")  # text
        self.legal_moves_button.setStyleSheet(
            "QPushButton { background-color: white; color: black;}"
        )
        self.legal_moves_button.setGeometry(20, 650, 550, 80)
        self.legal_moves_button.setTextInteractionFlags(
            QtGui.Qt.TextInteractionFlag.TextSelectableByMouse
        )

        self.eval_button = QPushButton(self)
        self.eval_button.setText("🐟 Eval")  # text
        self.eval_button.setStyleSheet(
            "QPushButton {background-color: white; color: black;}"
        )
        self.eval_button.setGeometry(620, 600, 470, 30)

        self.eval_button_chi = QPushButton(self)
        self.eval_button_chi.setText("🐙 Eval")  # text
        self.eval_button_chi.setStyleSheet(
            "QPushButton {background-color: white; color: black;}"
        )
        self.eval_button_chi.setGeometry(620, 650, 470, 30)

        self.eval_button_white = QPushButton(self)
        self.eval_button_white.setText("♕ White Eval")  # text
        self.eval_button_white.setStyleSheet(
            "QPushButton {background-color: white; color: black;}"
        )
        self.eval_button_white.setGeometry(620, 700, 470, 30)

        self.eval_button_black = QPushButton(self)
        self.eval_button_black.setText("♛ Black Eval")  # text
        self.eval_button_black.setStyleSheet(
            "QPushButton {background-color: white; color: black;}"
        )
        self.eval_button_black.setGeometry(620, 750, 470, 30)

        self.board_size = min(self.widgetSvg.width(), self.widgetSvg.height())
        self.coordinates = True
        self.margin = 0.05 * self.board_size if self.coordinates else 0
        self.squareSize = (self.board_size - 2 * self.margin) / 8.0
        self.pieceToMove = [None, None]

        self.checkThreadTimer = QTimer(self)
        self.checkThreadTimer.setInterval(5)  # .5 seconds
        self.checkThreadTimer.timeout.connect(self.process_message)
        self.checkThreadTimer.start()

        # Start with an empty/initial board so click-handling works before first update.
        self.board: IBoard = self.board_factory(
            fen_with_history=FenPlusHistory(current_fen=chess.STARTING_FEN)
        )
        self.draw_board()

    def _should_accept_scope(self, incoming: Scope) -> bool:
        """Routing policy.

        For now (single GUI view):
        - accept first scope
        - accept scope changes within the same session (and same match_id when both are set)
        - ignore updates from other sessions/matches to avoid mixing across runs
        """
        if self.scope is None:
            return True

        if incoming.session_id != self.scope.session_id:
            return False

        if (
            self.scope.match_id is not None
            and incoming.match_id is not None
            and incoming.match_id != self.scope.match_id
        ):
            return False

        return True

    def _check_and_set_icon(self, button: QPushButton, icon_path: str) -> None:
        """
        Check if icon file exists and set it, otherwise log a warning.

        Args:
            button: The button to set the icon for
            icon_path: The path to the icon file
        """
        if os.path.exists(icon_path):
            button.setIcon(QIcon(icon_path))
        else:
            chipiron_logger.warning("Icon file not found: %s", icon_path)
            # Set a default icon or leave empty
            button.setIcon(QIcon())  # Empty icon

    def stopppy(self) -> None:
        """
        Stops the execution of the GUI.

        This method closes the GUI window and sends a kill message to the main thread.

        Returns:
            None
        """
        # should we send a kill message to the main thread?
        self.close()

    def play_button_clicked(self) -> None:
        """
        Handle the event when the play button is clicked.

        This method prints a message indicating that the play button has been clicked,
        and sends a GameStatusMessage with the status set to PlayingStatus.PLAY to the main thread mailbox.
        """
        if self.playing_status != PlayingStatus.PAUSE:
            return
        if self.scope is None:
            return
        if (
            self.play_button_clicked_last_time is None
            or abs(self.play_button_clicked_last_time - time.time()) > 0.01
        ):
            chipiron_logger.info("play_button_clicked")
            cmd = GuiCommand(
                schema_version=1,
                scope=self.scope,
                payload=CmdSetStatus(status=PlayingStatus.PLAY),
            )
            self.main_thread_mailbox.put(cmd)
            self.play_button_clicked_last_time = time.time()

    def back_button_clicked(self) -> None:
        """
        Handle the event when the back button is clicked.

        This method prints a message and puts a `BackMessage` object into the main thread mailbox.
        """
        chipiron_logger.info("back_button_clicked")
        if self.scope is None:
            return
        cmd = GuiCommand(
            schema_version=1,
            scope=self.scope,
            payload=CmdBackOneMove(),
        )
        self.main_thread_mailbox.put(cmd)

    def pause_button_clicked(self) -> None:
        """
        Handles the click event of the pause button.

        Prints 'pause_button_clicked' and sends a GameStatusMessage with the status set to PlayingStatus.PAUSE
        to the main thread mailbox.
        """
        if self.playing_status != PlayingStatus.PLAY:
            return
        if self.scope is None:
            return
        if (
            self.pause_button_clicked_last_time is None
            or abs(self.pause_button_clicked_last_time - time.time()) > 0.01
        ):
            chipiron_logger.info("pause_button_clicked")
            cmd = GuiCommand(
                schema_version=1,
                scope=self.scope,
                payload=CmdSetStatus(status=PlayingStatus.PAUSE),
            )
            self.main_thread_mailbox.put(cmd)
            self.pause_button_clicked_last_time = time.time()

    @typing.no_type_check
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
                if (
                    self.margin < event.x() < self.board_size - self.margin
                    and self.margin < event.y() < self.board_size - self.margin
                ):
                    file = int((event.x() - self.margin) / self.squareSize)
                    rank = 7 - int((event.y() - self.margin) / self.squareSize)
                    square = chess.square(file, rank)
                    piece = self.board.piece_at(square)
                    self.coordinates = f"{chr(file + 97)}{rank + 1}"
                    if self.pieceToMove[0] is not None:
                        try:
                            all_moves_keys: list[MoveKey] = (
                                self.board.legal_moves.get_all()
                            )
                            all_legal_moves_uci: list[MoveUci] = [
                                self.board.legal_moves.generated_moves[move_key].uci()
                                for move_key in all_moves_keys
                            ]
                            move: chess.Move = chess.Move.from_uci(
                                "{}{}".format(self.pieceToMove[1], self.coordinates)
                            )
                            move_promote: chess.Move = chess.Move.from_uci(
                                "{}{}q".format(self.pieceToMove[1], self.coordinates)
                            )
                            if move.uci() in all_legal_moves_uci:
                                self.send_move_to_main_thread(move_uci=move.uci())
                            elif move_promote.uci() in all_legal_moves_uci:
                                self.choice_promote()
                                self.send_move_to_main_thread(
                                    move_uci=self.move_promote_asked.uci()
                                )
                            else:
                                legal_moves_uci: list[MoveUci] = [
                                    self.board.get_uci_from_move_key(move_key)
                                    for move_key in all_moves_keys
                                ]
                                chipiron_logger.info(
                                    "Looks like the move %s is a wrong move.. "
                                    "The legals moves are %s in %s",
                                    move,
                                    legal_moves_uci,
                                    self.board,
                                )
                        except ValueError:
                            chipiron_logger.info("Oops!  Doubleclicked?  Try again...")
                        piece = None
                    self.pieceToMove = [piece, self.coordinates]

    def send_move_to_main_thread(self, move_uci: MoveUci) -> None:
        """
        Sends a move to the main thread for processing.

        Args:
            move (chess.Move): The move to be sent.

        Returns:
            None
        """
        if self.scope is None:
            return
        cmd = GuiCommand(
            schema_version=1,
            scope=self.scope,
            payload=CmdHumanMoveUci(
                move_uci=str(move_uci),
                corresponding_fen=self.board.fen,
                color_to_play=self.board.turn,
            ),
        )
        self.main_thread_mailbox.put(cmd)

    def reset_for_new_game(self, scope: Scope) -> None:
        self.tablewidget.clearContents()
        self.tablewidget.setRowCount(1)
        self.progress_white.reset()
        self.progress_black.reset()
        self.progress_white.setValue(0)
        self.progress_black.setValue(0)
        self.eval_button.setText("🐟 Eval")
        self.eval_button_chi.setText("🐙 Eval")
        self.eval_button_white.setText("♕ White Eval")
        self.eval_button_black.setText("♛ Black Eval")
        self.pieceToMove = [None, None]
        self.board = self.board_factory(
            fen_with_history=FenPlusHistory(current_fen=chess.STARTING_FEN)
        )
        self.draw_board()

    @typing.no_type_check
    def choice_promote(self):
        """
        Displays a dialog box with buttons for promoting a chess piece.

        The dialog box allows the user to choose between promoting the pawn to a queen, rook, bishop, or knight.
        Each button is connected to a corresponding method for handling the promotion.

        Returns:
            None
        """

        self.d = QDialog()
        d = self.d
        d.setWindowTitle("Promote to ?")
        d.setWindowModality(Qt.ApplicationModal)

        d.closeButtonQ = QPushButton(d)
        d.closeButtonQ.setText("Queen")  # text
        d.closeButtonQ.setStyleSheet(
            "QPushButton {background-color: white; color: blue;}"
        )
        d.closeButtonQ.setGeometry(150, 100, 150, 20)
        d.closeButtonQ.clicked.connect(self.promote_queen)

        d.closeButtonR = QPushButton(d)
        d.closeButtonR.setText("Rook")  # text
        d.closeButtonR.setStyleSheet(
            "QPushButton {background-color: white; color: blue;}"
        )
        d.closeButtonR.setGeometry(150, 200, 150, 20)
        d.closeButtonR.clicked.connect(self.promote_rook)

        d.closeButtonB = QPushButton(d)
        d.closeButtonB.setText("Bishop")  # text
        d.closeButtonB.setStyleSheet(
            "QPushButton {background-color: white; color: blue;}"
        )
        d.closeButtonB.setGeometry(150, 300, 150, 20)
        d.closeButtonB.clicked.connect(self.promote_bishop)

        d.closeButtonK = QPushButton(d)
        d.closeButtonK.setText("Knight")  # text
        d.closeButtonK.setStyleSheet(
            "QPushButton {background-color: white; color: blue;}"
        )
        d.closeButtonK.setGeometry(150, 400, 150, 20)
        d.closeButtonK.clicked.connect(self.promote_knight)

        d.exec_()

    @typing.no_type_check
    def promote_queen(self):
        """
        Promotes the selected piece to a queen.

        This method creates a move object to promote the selected piece to a queen by appending 'q' to the UCI notation
        of the piece's destination square. It then closes the dialog window.

        Returns:
            None
        """
        self.move_promote_asked = chess.Move.from_uci(
            "{}{}q".format(self.pieceToMove[1], self.coordinates)
        )
        self.d.close()

    @typing.no_type_check
    def promote_rook(self):
        """
        Promotes a pawn to a rook.

        This method is called when a pawn reaches the opposite end of the board and needs to be promoted to a rook.
        It creates a move object representing the promotion and closes the dialog window.

        Returns:
            None
        """
        self.move_promote_asked = chess.Move.from_uci(
            f"{self.pieceToMove[1]}{self.coordinates}r"
        )
        self.d.close()

    @typing.no_type_check
    def promote_bishop(self):
        """
        Promotes the current piece to a bishop.

        This method creates a move object to promote the current piece to a bishop and closes the dialog window.

        Returns:
            None
        """
        self.move_promote_asked = chess.Move.from_uci(
            f"{self.pieceToMove[1]}{self.coordinates}b"
        )
        self.d.close()

    @typing.no_type_check
    def promote_knight(self):
        """
        Promotes a pawn to a knight.

        This method is called when a pawn is promoted to a knight in the GUI.
        It creates a move object representing the promotion and closes the GUI.

        Returns:
        None
        """
        self.move_promote_asked = chess.Move.from_uci(
            f"{self.pieceToMove[1]}{self.coordinates}n"
        )
        self.d.close()

    def process_message(self) -> None:
        """
        Process a message received by the GUI.

        Draw a chessboard with the starting position and then redraw
        it for every new move.

        This method is responsible for handling different types of messages
        received by the GUI and taking appropriate actions based on the message type.

        Supported message types:
        - BoardMessage: Updates the board and redraws it.
        - EvaluationMessage: Updates the evaluation values.
        - PlayersColorToPlayerMessage: Updates the mapping of player colors to player information.
        - MatchResultsMessage: Updates the match results.
        - GameStatusMessage: Updates the game play status.
        - Other: Raises a ValueError indicating an unknown message type.

        Returns:
        None
        """
        if self.gui_mailbox.empty():
            return

        msg: GuiUpdate = self.gui_mailbox.get()

        if not self._should_accept_scope(msg.scope):
            chipiron_logger.debug(
                "Ignoring GuiUpdate from other scope: %s (current=%s)",
                msg.scope,
                self.scope,
            )
            return

        if self.scope is None:
            self.scope = msg.scope
        elif msg.scope != self.scope:
            # sequential-games policy: accept new scope and reset UI
            self.reset_for_new_game(msg.scope)
            self.scope = msg.scope

        payload = msg.payload

        match payload:
            case UpdStateChess():
                self.board = self.board_factory(
                    fen_with_history=payload.fen_plus_history
                )
                self.draw_board()
                self.display_move_history()

            case UpdPlayerProgress():
                if (
                    payload.player_color == Color.WHITE
                    and payload.progress_percent is not None
                ):
                    self.progress_white.setValue(payload.progress_percent)
                elif (
                    payload.player_color == Color.BLACK
                    and payload.progress_percent is not None
                ):
                    self.progress_black.setValue(payload.progress_percent)

            case UpdEvaluation():
                self.update_evaluation(
                    evaluation_stock=payload.stock,
                    evaluation_chipiron=payload.chipiron,
                    evaluation_white=payload.white,
                    evaluation_black=payload.black,
                )

            case UpdPlayersInfo():
                self.update_players_info(payload.white, payload.black)

            case UpdMatchResults():
                self.update_match_stats(payload)

            case UpdGameStatus():
                self.update_game_play_status(payload.status)

            case _:
                raise AssertionError(f"Unhandled GuiUpdate payload: {payload!r}")

    def display_move_history(self) -> None:
        """
        Display the move history in a table widget.

        This method calculates the number of rounds based on the number of half moves in the move stack.
        It then sets the number of rows in the table widget to the number of rounds.
        The table widget's horizontal header labels are set to 'White' and 'Black'.
        The move history is then iterated over and each move is added to the table widget.

        Returns:
            None
        """

        num_half_move: int = len(self.board.move_history_stack)
        num_rounds: int = int(math.ceil(num_half_move / 2))
        self.tablewidget.setRowCount(num_rounds)
        self.tablewidget.setHorizontalHeaderLabels(["White", "Black"])
        for player in range(2):
            for round_ in range(num_rounds):
                half_move = round_ * 2 + player
                if half_move < num_half_move:
                    item = QTableWidgetItem(
                        str(self.board.move_history_stack[half_move])
                    )
                    self.tablewidget.setItem(round_, player, item)

    def draw_board(self) -> None:
        """
        Draw a chessboard with the starting position and then redraw
        it for every new move.

        Returns:
            None
        """
        board_chi = create_board_chi(
            fen_with_history=FenPlusHistory(
                current_fen=self.board.fen,
                historical_moves=self.board.move_history_stack,
            )
        )

        repr_svg: str = chess.svg.board(
            board=board_chi.chess_board,
            size=390,
            lastmove=board_chi.chess_board.peek()
            if board_chi.chess_board.move_stack
            else None,
            check=board_chi.chess_board.king(board_chi.chess_board.turn)
            if board_chi.chess_board.is_check()
            else None,
        )

        self.boardSvg = repr_svg.encode("UTF-8")
        self.drawBoardSvg = self.widgetSvg.load(self.boardSvg)
        self.round_button.setText(
            "🎲 Round: " + str(self.board.fullmove_number)
        )  # text
        self.fen_button.setText("🔧 <b>fen:</b> " + str(self.board.fen))  # text

        all_moves_keys_chi = self.board.legal_moves.get_all()
        all_moves_uci_chi = [
            self.board.get_uci_from_move_key(move_key=move_key)
            for move_key in all_moves_keys_chi
        ]

        self.legal_moves_button.setWordWrap(True)
        self.legal_moves_button.setMinimumHeight(100)
        lines: list[str] = []

        # Only add sublists if they are non-empty
        for sublist in [
            all_moves_uci_chi[:7],
            all_moves_uci_chi[7:14],
            all_moves_uci_chi[14:21],
            all_moves_uci_chi[21:28],
        ]:
            if sublist:  # skip empty
                lines.append("    " + str(sublist))

        # Join lines with <br>
        moves_html = "\n".join(lines)

        self.legal_moves_button.setText(
            f"📋 <b>legal moves:</b><pre>{moves_html}</pre>"
        )
        return self.drawBoardSvg

    def update_players_info(self, white: PlayerUiInfo, black: PlayerUiInfo) -> None:
        self.player_white_button.setText(" White: " + white.label)
        self.player_black_button.setText(" Black: " + black.label)

        if black.is_human:
            self.progress_black.setTextVisible(True)
            self.progress_black.setValue(100)
            self.progress_black.setFormat("Think Human!")
        else:
            self.progress_black.setValue(0)

        if white.is_human:
            self.progress_white.setTextVisible(True)
            self.progress_white.setValue(100)
            self.progress_white.setFormat("Think Human!")
        else:
            self.progress_white.setValue(0)

    def update_evaluation(
        self,
        evaluation_stock: float | None,
        evaluation_chipiron: float | None,
        evaluation_white: float | None,
        evaluation_black: float | None,
    ) -> None:
        """
        Update the evaluation values displayed on the GUI.

        Args:
            evaluation_stock (float | None): The evaluation value for the stock.
            evaluation_chipiron (float | None): The evaluation value for the chipiron.
            evaluation_white (float | None): The evaluation value for white.
            evaluation_black (float | None): The evaluation value for black.

        Returns:
            None
        """
        self.eval_button.setText("📊 Eval 🐟: " + str(evaluation_stock))
        self.eval_button_chi.setText("🧮 Eval 🐙: " + str(evaluation_chipiron))
        self.eval_button_black.setText("🧠 Eval White: " + str(evaluation_white))
        self.eval_button_white.setText("🧠 Eval Black: " + str(evaluation_black))

    def update_game_play_status(self, play_status: PlayingStatus) -> None:
        """Update the game play status.

        Args:
            play_status (PlayingStatus): The new playing status.
        """
        chipiron_logger.info("update_game_play_status %s", play_status)

        if self.playing_status != play_status:
            self.playing_status = play_status
            if play_status == PlayingStatus.PAUSE:
                self.pause_button.setText("Play")  # text
                self._check_and_set_icon(
                    self.pause_button, os.path.join(GUI_DIR, "play.png")
                )  # icon
                self.pause_button.clicked.connect(self.play_button_clicked)
                self.pause_button.setToolTip("play the game")  # Tool tip
                self.pause_button.move(700, 100)
            elif play_status == PlayingStatus.PLAY:
                self.pause_button.setText("Pause")  # text
                self._check_and_set_icon(
                    self.pause_button, os.path.join(GUI_DIR, "pause.png")
                )  # icon
                self.pause_button.clicked.connect(self.pause_button_clicked)
                self.pause_button.setToolTip("pause the game")  # Tool tip
                self.pause_button.move(700, 100)

    def update_match_stats(self, upd: UpdMatchResults) -> None:
        self.score_button.setText(
            f"⚖ Score: {upd.wins_white}-{upd.wins_black}-{upd.draws}"
        )

        chipiron_logger.info("update match_finished=%s", upd.match_finished)
        if upd.match_finished:
            chipiron_logger.info("finishing the widget")
            self.close()

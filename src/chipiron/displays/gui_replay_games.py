#! /usr/bin/env python

"""Document the module is the execution point of the chess GUI application."""

import typing
from typing import Any

import chess
import chess.svg
from atomheart.board.board_chi import BoardChi
from PySide6.QtCore import Slot
from PySide6.QtGui import QIcon, QKeyEvent
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtWidgets import QPushButton, QWidget


class MainWindow(QWidget):
    """Create a surface for the chessboard."""

    def __init__(self, chess_board: BoardChi) -> None:
        """Initialize the chessboard."""
        super().__init__()

        self.chess_board_recorded = chess_board
        self.chess_board = chess.Board()

        self.setWindowTitle("Chess GUI")
        self.setGeometry(300, 300, 800, 800)

        self.widget_svg = QSvgWidget(parent=self)
        self.widget_svg.setGeometry(10, 10, 600, 600)

        self.close_button = QPushButton(self)
        self.close_button.setText("Close")  # text
        self.close_button.setIcon(QIcon("close.png"))  # icon
        self.close_button.setShortcut("Ctrl+D")  # shortcut key
        self.close_button.setToolTip("Close the widget")  # Tool tip
        self.close_button.move(700, 100)

        self.close_button2 = QPushButton(self)
        self.close_button2.setText("Player")  # text
        self.close_button2.setStyleSheet(
            "QPushButton {background-color: white; color: blue;}"
        )
        self.close_button2.setGeometry(650, 200, 150, 20)

        self.close_button3 = QPushButton(self)
        self.close_button3.setText("Player")  # text
        self.close_button3.setStyleSheet(
            "QPushButton {background-color: black; color: blue;}"
        )
        self.close_button3.setGeometry(650, 300, 150, 20)

        self.close_button4 = QPushButton(self)
        self.close_button4.setText("Score 0-0")  # text
        self.close_button4.setStyleSheet(
            "QPushButton {background-color: black; color: blue;}"
        )
        self.close_button4.setGeometry(650, 400, 150, 20)

        self.board_size = min(self.widget_svg.width(), self.widget_svg.height())
        self.coordinates = True
        self.margin = 0.05 * self.board_size if self.coordinates else 0
        self.square_size = (self.board_size - 2 * self.margin) / 8.0

        self.count = 0
        self.next_move = self.chess_board_recorded.chess_board.move_stack[self.count]

        self.draw_board()

    @typing.override
    @Slot(QWidget)
    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle key press events.

        Args:
            event (QKeyEvent): The key event object.

        """
        print(event.text())
        key = event.text()
        if key == "1" and self.count > 0:
            self.chess_board.pop()
            print(self.chess_board)
            print(self.chess_board.fen)
            self.count -= 1
            self.next_move = self.chess_board_recorded.chess_board.move_stack[
                self.count
            ]
            self.draw_board()
        if key == "2" and self.count < len(
            self.chess_board_recorded.chess_board.move_stack
        ):
            self.chess_board.push(self.next_move)
            print(self.chess_board)
            print(self.chess_board.fen)
            self.count += 1
            if self.count < len(self.chess_board_recorded.chess_board.move_stack):
                self.next_move = self.chess_board_recorded.chess_board.move_stack[
                    self.count
                ]
            self.draw_board()

    def draw_board(self) -> Any:
        """Draw a chessboard with the starting position and then redraw.

        it for every new move.
        """
        repr_svg: str = chess.svg.board(
            board=self.chess_board,
            size=390,
            lastmove=self.chess_board.peek() if self.chess_board.move_stack else None,
            check=self.chess_board.king(self.chess_board.turn)
            if self.chess_board.is_check()
            else None,
        )

        self.board_svg = repr_svg.encode("UTF-8")

        self.draw_board_svg = self.widget_svg.load(self.board_svg)

        return self.draw_board_svg

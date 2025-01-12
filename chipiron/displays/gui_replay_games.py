#! /usr/bin/env python

"""
This module is the execution point of the chess GUI application.
"""

from typing import Any

import chess
from PySide6.QtCore import Slot
from PySide6.QtGui import QIcon, QKeyEvent
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtWidgets import QPushButton, QWidget

from chipiron.environments.chess.board.board_chi import BoardChi


class MainWindow(QWidget):
    """
    Create a surface for the chessboard.
    """

    def __init__(self, chess_board: BoardChi) -> None:
        """
        Initialize the chessboard.
        """
        super().__init__()

        self.chess_board_recorded = chess_board
        self.chess_board = chess.Board()

        self.setWindowTitle("Chess GUI")
        self.setGeometry(300, 300, 800, 800)

        self.widgetSvg = QSvgWidget(parent=self)
        self.widgetSvg.setGeometry(10, 10, 600, 600)

        self.closeButton = QPushButton(self)
        self.closeButton.setText("Close")  # text
        self.closeButton.setIcon(QIcon("close.png"))  # icon
        self.closeButton.setShortcut("Ctrl+D")  # shortcut key
        self.closeButton.setToolTip("Close the widget")  # Tool tip
        self.closeButton.move(700, 100)

        self.closeButton2 = QPushButton(self)
        self.closeButton2.setText("Player")  # text
        # self.closeButton2.move(700, 200)
        self.closeButton2.setStyleSheet(
            "QPushButton {background-color: white; color: blue;}"
        )
        self.closeButton2.setGeometry(650, 200, 150, 20)

        self.closeButton3 = QPushButton(self)
        self.closeButton3.setText("Player")  # text
        self.closeButton3.setStyleSheet(
            "QPushButton {background-color: black; color: blue;}"
        )
        self.closeButton3.setGeometry(650, 300, 150, 20)

        self.closeButton4 = QPushButton(self)
        self.closeButton4.setText("Score 0-0")  # text
        self.closeButton4.setStyleSheet(
            "QPushButton {background-color: black; color: blue;}"
        )
        self.closeButton4.setGeometry(650, 400, 150, 20)

        self.boardSize = min(self.widgetSvg.width(), self.widgetSvg.height())
        self.coordinates = True
        self.margin = 0.05 * self.boardSize if self.coordinates else 0
        self.squareSize = (self.boardSize - 2 * self.margin) / 8.0

        self.count = 0
        self.next_move = self.chess_board_recorded.chess_board.move_stack[self.count]

        self.drawBoard()

    @Slot(QWidget)
    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle key press events.

        Args:
            event (QKeyEvent): The key event object.
        """
        print(event.text())
        key = event.text()
        if key == "1":
            if self.count > 0:
                self.chess_board.pop()
                print(self.chess_board)
                print(self.chess_board.fen)
                self.count -= 1
                self.next_move = self.chess_board_recorded.chess_board.move_stack[
                    self.count
                ]
                self.drawBoard()
        if key == "2":
            if self.count < len(self.chess_board_recorded.chess_board.move_stack):
                self.chess_board.push(self.next_move)
                print(self.chess_board)
                print(self.chess_board.fen)
                self.count += 1
                if self.count < len(self.chess_board_recorded.chess_board.move_stack):
                    self.next_move = self.chess_board_recorded.chess_board.move_stack[
                        self.count
                    ]
                self.drawBoard()

    def drawBoard(self) -> Any:
        """
        Draw a chessboard with the starting position and then redraw
        it for every new move.
        """

        self.boardSvg = self.chess_board._repr_svg_().encode("UTF-8")
        self.drawBoardSvg = self.widgetSvg.load(self.boardSvg)

        return self.drawBoardSvg

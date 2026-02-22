"""Chess-specific SVG adapter for the generic GUI."""

from dataclasses import dataclass
from typing import Any

import chess
import chess.svg
from atomheart.board import BoardFactory, IBoard, create_board_chi
from atomheart.board.utils import FenPlusHistory
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QPushButton

from chipiron.displays.svg_adapter_errors import InvalidSvgAdapterPayloadTypeError
from chipiron.displays.svg_adapter_protocol import (
    ClickResult,
    RenderResult,
    SvgGameAdapter,
    SvgPosition,
)
from chipiron.utils.logger import chipiron_logger


@dataclass
class ChessSvgAdapter(SvgGameAdapter):
    """SVG adapter implementation for chess."""

    board_factory: BoardFactory

    game_name: str = "chess"
    board_side: int = 8

    _selected_from: str | None = None
    _selected_to: str | None = None
    _move_promote_asked: chess.Move | None = None
    _board: IBoard | None = None

    def position_from_update(
        self, *, state_tag: object, adapter_payload: Any
    ) -> SvgPosition:
        """Build chess position from incoming generic adapter payload."""
        if not isinstance(adapter_payload, FenPlusHistory):
            raise InvalidSvgAdapterPayloadTypeError(
                adapter_name=self.__class__.__name__,
                expected_type=FenPlusHistory,
                actual_value=adapter_payload,
            )
        self._board = self.board_factory(fen_with_history=adapter_payload)
        self.reset_interaction()
        return SvgPosition(state_tag=state_tag, payload=adapter_payload)

    def render_svg(self, pos: SvgPosition, size: int) -> RenderResult:
        """Render chess board SVG from fen/history payload."""
        fen_plus_history = pos.payload
        if not isinstance(fen_plus_history, FenPlusHistory):
            raise InvalidSvgAdapterPayloadTypeError(
                adapter_name=self.__class__.__name__,
                expected_type=FenPlusHistory,
                actual_value=fen_plus_history,
            )

        board_chi = create_board_chi(fen_with_history=fen_plus_history)
        repr_svg = chess.svg.board(
            board=board_chi.chess_board,
            size=size,
            lastmove=board_chi.chess_board.peek()
            if board_chi.chess_board.move_stack
            else None,
            check=board_chi.chess_board.king(board_chi.chess_board.turn)
            if board_chi.chess_board.is_check()
            else None,
        )

        all_moves_keys_chi = board_chi.legal_moves.get_all()
        all_moves_uci_chi = [
            board_chi.get_uci_from_move_key(move_key=move_key)
            for move_key in all_moves_keys_chi
        ]
        chunks = [
            str(all_moves_uci_chi[index : index + 7])
            for index in range(0, len(all_moves_uci_chi), 7)
        ]

        return RenderResult(
            svg_bytes=repr_svg.encode("UTF-8"),
            info={
                "round": str(board_chi.fullmove_number),
                "fen": str(board_chi.fen),
                "legal_moves": "\n".join(f"    {chunk}" for chunk in chunks),
            },
        )

    def handle_click(
        self,
        pos: SvgPosition,
        *,
        x: int,
        y: int,
        board_size: int,
        margin: int,
    ) -> ClickResult:
        """Convert click coordinates into chess action names."""
        del pos
        if self._board is None:
            return ClickResult(action_name=None, interaction_continues=False)

        square_size = (board_size - 2 * margin) / 8.0
        file = int((x - margin) / square_size)
        rank = 7 - int((y - margin) / square_size)
        coord = f"{chr(file + 97)}{rank + 1}"

        if self._selected_from is None:
            self._selected_from = coord
            chipiron_logger.debug("selected %s", coord)
            return ClickResult(action_name=None, interaction_continues=True)

        from_sq = self._selected_from
        self._selected_to = coord
        try:
            all_moves_keys = self._board.legal_moves.get_all()
            all_legal_moves_uci: set[str] = {
                str(self._board.get_uci_from_move_key(move_key=move_key))
                for move_key in all_moves_keys
            }

            move = chess.Move.from_uci(f"{from_sq}{coord}")
            move_promote = chess.Move.from_uci(f"{from_sq}{coord}q")

            chipiron_logger.debug("trying %s turn %s", move.uci(), self._board.turn)

            if move.uci() in all_legal_moves_uci:
                self._selected_from = None
                self._selected_to = None
                return ClickResult(action_name=move.uci(), interaction_continues=False)

            if move_promote.uci() in all_legal_moves_uci:
                self._choice_promote()
                if self._move_promote_asked is not None:
                    chosen = self._move_promote_asked.uci()
                    if chosen in all_legal_moves_uci:
                        self._selected_from = None
                        self._selected_to = None
                        return ClickResult(
                            action_name=chosen,
                            interaction_continues=False,
                        )

            chipiron_logger.debug("illegal move; reselecting %s", coord)
            self._selected_from = coord
            return ClickResult(action_name=None, interaction_continues=True)

        except ValueError:
            chipiron_logger.info("Oops! Doubleclicked? Try again...")
            self._selected_from = None
            self._selected_to = None
            return ClickResult(action_name=None, interaction_continues=False)

    def reset_interaction(self) -> None:
        """Reset in-progress user interaction state."""
        self._selected_from = None
        self._selected_to = None
        self._move_promote_asked = None

    def _choice_promote(self) -> None:
        """Display promotion dialog and capture chosen promotion move."""
        d = QDialog()
        d.setWindowTitle("Promote to ?")
        d.setWindowModality(Qt.WindowModality.ApplicationModal)

        close_button_q = QPushButton(d)
        close_button_q.setText("Queen")
        close_button_q.setStyleSheet(
            "QPushButton {background-color: white; color: blue;}"
        )
        close_button_q.setGeometry(150, 100, 150, 20)
        close_button_q.clicked.connect(lambda: self._choose_and_close(d, "q"))  # pylint: disable=no-member

        close_button_r = QPushButton(d)
        close_button_r.setText("Rook")
        close_button_r.setStyleSheet(
            "QPushButton {background-color: white; color: blue;}"
        )
        close_button_r.setGeometry(150, 200, 150, 20)
        close_button_r.clicked.connect(lambda: self._choose_and_close(d, "r"))  # pylint: disable=no-member

        close_button_b = QPushButton(d)
        close_button_b.setText("Bishop")
        close_button_b.setStyleSheet(
            "QPushButton {background-color: white; color: blue;}"
        )
        close_button_b.setGeometry(150, 300, 150, 20)
        close_button_b.clicked.connect(lambda: self._choose_and_close(d, "b"))  # pylint: disable=no-member

        close_button_k = QPushButton(d)
        close_button_k.setText("Knight")
        close_button_k.setStyleSheet(
            "QPushButton {background-color: white; color: blue;}"
        )
        close_button_k.setGeometry(150, 400, 150, 20)
        close_button_k.clicked.connect(lambda: self._choose_and_close(d, "n"))  # pylint: disable=no-member

        d.exec()

    def _choose_and_close(self, dialog: QDialog, promotion: str) -> None:
        if self._selected_from is None or self._selected_to is None:
            return
        self._move_promote_asked = chess.Move.from_uci(
            f"{self._selected_from}{self._selected_to}{promotion}"
        )
        dialog.close()

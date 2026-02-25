"""Checkers SVG adapter for the generic GUI."""

import re
from dataclasses import dataclass, field
from typing import Any, cast

from chipiron.displays.svg_adapter_errors import InvalidSvgAdapterPayloadTypeError
from chipiron.displays.svg_adapter_protocol import (
    ClickResult,
    RenderResult,
    SvgGameAdapter,
    SvgPosition,
)

MOVE_SQUARE_RE = re.compile(r"\d+")


def _sq32_to_rc(square: int) -> tuple[int, int]:
    row = square // 4
    offset = square % 4
    col = 1 + 2 * offset if row % 2 == 0 else 2 * offset
    return row, col


def _click_to_sq32(*, x: int, y: int, board_size: int, margin: int) -> int | None:
    square_size = (board_size - 2 * margin) / 8.0
    file_ = int((x - margin) / square_size)
    row_from_top = int((y - margin) / square_size)

    if file_ < 0 or file_ > 7 or row_from_top < 0 or row_from_top > 7:
        return None
    if (row_from_top + file_) % 2 == 0:
        return None

    base = row_from_top * 4
    offset = (file_ - 1) // 2 if row_from_top % 2 == 0 else file_ // 2
    if offset < 0 or offset > 3:
        return None
    return base + offset


def _extract_path(move_name: str) -> list[int]:
    return [int(chunk) for chunk in MOVE_SQUARE_RE.findall(move_name)]


def _empty_legal_moves() -> list[str]:
    return []


def _empty_pieces() -> list[int]:
    return [0] * 32


@dataclass
class CheckersSvgAdapter(SvgGameAdapter):
    """SVG adapter implementation for checkers."""

    game_name: str = "checkers"
    board_side: int = 8

    _selected_square: int | None = None
    _legal_moves: list[str] = field(default_factory=_empty_legal_moves)
    _position_text: str = ""
    _pieces: list[int] = field(default_factory=_empty_pieces)

    def position_from_update(
        self, *, state_tag: object, adapter_payload: Any
    ) -> SvgPosition:
        """Build checkers position from incoming generic adapter payload."""
        if not isinstance(adapter_payload, dict):
            raise InvalidSvgAdapterPayloadTypeError(
                adapter_name=self.__class__.__name__,
                expected_type=dict,
                actual_value=adapter_payload,
            )

        payload = cast("dict[str, object]", adapter_payload)

        position_text_raw: object = payload.get("position_text", None)
        legal_moves_raw: object = payload.get("legal_moves", None)
        pieces_raw: object = payload.get("pieces", None)

        if not isinstance(position_text_raw, str):
            raise InvalidSvgAdapterPayloadTypeError(
                adapter_name=self.__class__.__name__,
                expected_type=dict,
                actual_value=adapter_payload,
            )

        if not isinstance(legal_moves_raw, list):
            raise InvalidSvgAdapterPayloadTypeError(
                adapter_name=self.__class__.__name__,
                expected_type=dict,
                actual_value=adapter_payload,
            )

        legal_moves_raw_list = cast("list[object]", legal_moves_raw)
        legal_moves: list[str] = []
        for move in legal_moves_raw_list:
            if not isinstance(move, str):
                raise InvalidSvgAdapterPayloadTypeError(
                    adapter_name=self.__class__.__name__,
                    expected_type=dict,
                    actual_value=adapter_payload,
                )
            legal_moves.append(move)

        if not isinstance(pieces_raw, list):
            raise InvalidSvgAdapterPayloadTypeError(
                adapter_name=self.__class__.__name__,
                expected_type=dict,
                actual_value=adapter_payload,
            )

        pieces_raw_list = cast("list[object]", pieces_raw)
        if len(pieces_raw_list) != 32:
            raise InvalidSvgAdapterPayloadTypeError(
                adapter_name=self.__class__.__name__,
                expected_type=dict,
                actual_value=adapter_payload,
            )

        pieces: list[int] = []
        for piece in pieces_raw_list:
            if not isinstance(piece, int):
                raise InvalidSvgAdapterPayloadTypeError(
                    adapter_name=self.__class__.__name__,
                    expected_type=dict,
                    actual_value=adapter_payload,
                )
            pieces.append(piece)

        self._position_text = position_text_raw
        self._legal_moves = legal_moves
        self._pieces = pieces
        self.reset_interaction()
        return SvgPosition(state_tag=state_tag, payload=adapter_payload)

    def render_svg(
        self, pos: SvgPosition, size: int, *, margin: int = 0
    ) -> RenderResult:
        """Render checkerboard SVG and pieces from structured payload."""
        del pos

        m = margin
        square_size = (size - 2 * m) / 8.0
        x0 = m
        y0 = m

        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}">'
        ]

        # draw the board INSET by margin
        for row in range(8):
            for col in range(8):
                fill = "#6b4f33" if (row + col) % 2 else "#f2dfc2"
                parts.append(
                    f'<rect x="{x0 + col * square_size}" y="{y0 + row * square_size}" '
                    f'width="{square_size}" height="{square_size}" fill="{fill}"/>'
                )

        # draw the pieces using the same coordinate system
        for index, piece in enumerate(self._pieces):
            if piece == 0:
                continue
            row, col = _sq32_to_rc(index)

            cx = x0 + (col + 0.5) * square_size
            cy = y0 + (row + 0.5) * square_size

            if piece > 0:
                fill = "#f7f7f7"
                stroke = "#202020"
            else:
                fill = "#202020"
                stroke = "#f7f7f7"

            parts.append(
                f'<circle cx="{cx}" cy="{cy}" r="{square_size * 0.35}" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
            )

            if abs(piece) == 2:
                parts.append(
                    f'<text x="{cx}" y="{cy + square_size * 0.08}" '
                    'text-anchor="middle" font-size="18" fill="#d4af37">K</text>'
                )

        parts.append("</svg>")

        chunks = [
            str(self._legal_moves[i : i + 5])
            for i in range(0, len(self._legal_moves), 5)
        ]

        return RenderResult(
            svg_bytes="".join(parts).encode("utf-8"),
            info={
                "round": "-",
                "fen": self._position_text,
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
        """Convert click coordinates into a checkers action name."""
        del pos

        square = _click_to_sq32(x=x, y=y, board_size=board_size, margin=margin)

        if square is None:
            return ClickResult(action_name=None, interaction_continues=False)

        if self._selected_square is None:
            self._selected_square = square
            return ClickResult(action_name=None, interaction_continues=True)

        matching = [
            move
            for move in self._legal_moves
            if (path := _extract_path(move))
            and path[0] == self._selected_square
            and path[-1] == square
        ]
        self._selected_square = None

        if len(matching) == 1:
            return ClickResult(action_name=matching[0], interaction_continues=False)

        return ClickResult(action_name=None, interaction_continues=False)

    def reset_interaction(self) -> None:
        """Reset click state for checkers interactions."""
        self._selected_square = None

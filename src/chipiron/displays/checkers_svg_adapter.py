"""Checkers SVG adapter for the generic GUI."""

import re
from dataclasses import dataclass, field
from typing import Any

from chipiron.displays.svg_adapter_errors import InvalidSvgAdapterPayloadTypeError
from chipiron.displays.svg_adapter_protocol import (
    ClickResult,
    RenderResult,
    SvgGameAdapter,
    SvgPosition,
)

MOVE_SQUARE_RE = re.compile(r"\d+")


def _sq32_to_rc(square: int) -> tuple[int, int]:
    row = (square - 1) // 4
    offset = (square - 1) % 4
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
    return base + offset + 1


def _extract_path(move_name: str) -> list[int]:
    return [int(chunk) for chunk in MOVE_SQUARE_RE.findall(move_name)]


@dataclass
class CheckersSvgAdapter(SvgGameAdapter):
    """SVG adapter implementation for checkers."""

    game_name: str = "checkers"
    board_side: int = 8

    _selected_square: int | None = None
    _legal_moves: list[str] = field(default_factory=list)
    _position_text: str = ""
    _pieces: list[int] = field(default_factory=lambda: [0] * 32)

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

        position_text = adapter_payload.get("position_text")
        legal_moves = adapter_payload.get("legal_moves")
        pieces = adapter_payload.get("pieces")

        valid_pieces = (
            isinstance(pieces, list)
            and len(pieces) == 32
            and all(isinstance(piece, int) for piece in pieces)
        )

        if (
            not isinstance(position_text, str)
            or not isinstance(legal_moves, list)
            or not all(isinstance(move, str) for move in legal_moves)
            or not valid_pieces
        ):
            raise InvalidSvgAdapterPayloadTypeError(
                adapter_name=self.__class__.__name__,
                expected_type=dict,
                actual_value=adapter_payload,
            )

        self._position_text = position_text
        self._legal_moves = legal_moves
        self._pieces = pieces
        self.reset_interaction()
        return SvgPosition(state_tag=state_tag, payload=adapter_payload)

    def render_svg(self, pos: SvgPosition, size: int) -> RenderResult:
        """Render checkerboard SVG and pieces from structured payload."""
        del pos
        square_size = size / 8

        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}">'
        ]

        for row in range(8):
            for col in range(8):
                fill = "#6b4f33" if (row + col) % 2 else "#f2dfc2"
                parts.append(
                    f'<rect x="{col * square_size}" y="{row * square_size}" '
                    f'width="{square_size}" height="{square_size}" fill="{fill}"/>'
                )

        for index, piece in enumerate(self._pieces):
            if piece == 0:
                continue
            row, col = _sq32_to_rc(index + 1)
            cx = (col + 0.5) * square_size
            cy = (row + 0.5) * square_size
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
            str(self._legal_moves[index : index + 5])
            for index in range(0, len(self._legal_moves), 5)
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

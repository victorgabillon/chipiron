"""Checkers SVG adapter stub for generic GUI extensibility."""

from dataclasses import dataclass
from typing import Any

from chipiron.displays.svg_adapter_protocol import (
    ClickResult,
    RenderResult,
    SvgGameAdapter,
    SvgPosition,
)


@dataclass
class CheckersSvgAdapter(SvgGameAdapter):
    """Minimal checkers adapter placeholder."""

    game_name: str = "checkers"
    board_side: int = 8

    _selected: tuple[int, int] | None = None

    def position_from_update(self, *, state_tag: object, adapter_payload: Any) -> SvgPosition:
        """Return an opaque checkers position."""
        return SvgPosition(state_tag=state_tag, payload=adapter_payload)

    def render_svg(self, pos: SvgPosition, size: int) -> RenderResult:
        """Render a basic checkerboard SVG without pieces."""
        del pos
        square_size = size / 8
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}">'
        ]
        for rank in range(8):
            for file in range(8):
                fill = "#777" if (rank + file) % 2 else "#eee"
                parts.append(
                    f'<rect x="{file * square_size}" y="{rank * square_size}" '
                    f'width="{square_size}" height="{square_size}" fill="{fill}"/>'
                )
        parts.append("</svg>")
        return RenderResult(
            svg_bytes="".join(parts).encode("utf-8"),
            info={"round": "-", "fen": "-", "legal_moves": ""},
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
        """Handle click interactions (stub: no action produced yet)."""
        del pos
        square_size = (board_size - 2 * margin) / 8.0
        file = int((x - margin) / square_size)
        rank = 7 - int((y - margin) / square_size)
        if (rank + file) % 2 == 0:
            return ClickResult(action_name=None, interaction_continues=False)
        return ClickResult(action_name=None, interaction_continues=True)

    def reset_interaction(self) -> None:
        """Reset click state for checkers interactions."""
        self._selected = None

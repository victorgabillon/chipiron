"""Minimal Morpion SVG adapter for the generic GUI."""

from dataclasses import dataclass
from html import escape
from typing import Any

from chipiron.displays.svg_adapter_errors import InvalidSvgAdapterPayloadTypeError
from chipiron.displays.svg_adapter_protocol import (
    ClickResult,
    RenderResult,
    SvgGameAdapter,
    SvgPosition,
)
from chipiron.environments.morpion.morpion_gui_encoder import MorpionDisplayPayload


@dataclass(frozen=True, slots=True)
class _ActionButton:
    """Clickable rectangle associated with one Morpion action."""

    action_name: str
    x: float
    y: float
    width: float
    height: float

    def contains(self, *, x: int, y: int) -> bool:
        """Return whether the given click falls inside the button."""
        return (
            self.x <= x <= self.x + self.width and self.y <= y <= self.y + self.height
        )


def _clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a scalar into a closed interval."""
    return min(max(value, lower), upper)


def _estimate_text_width(text: str, font_size: float, width_factor: float = 0.56) -> float:
    """Estimate SVG text width without requiring font metrics."""
    return len(text) * font_size * width_factor


def _fit_font_size(
    *,
    text: str,
    max_width: float,
    min_size: float,
    max_size: float,
    width_factor: float = 0.56,
) -> float:
    """Return a width-aware font size bounded by readable min/max values."""
    if not text:
        return min_size
    width_limited_size = max_width / max(len(text) * width_factor, 1.0)
    return max(1.0, min(max_size, max(min_size, width_limited_size)))


def _fmt(value: float) -> str:
    """Format SVG coordinates compactly while keeping stable precision."""
    return f"{value:.2f}"


@dataclass
class MorpionSvgAdapter(SvgGameAdapter):
    """SVG adapter implementation for Morpion."""

    game_name: str = "morpion"
    board_side: int = 1

    _payload: MorpionDisplayPayload | None = None
    _buttons: tuple[_ActionButton, ...] = ()

    _TITLE = "Morpion Solitaire"
    _TERMINAL_MESSAGE = "No legal moves remain"

    def position_from_update(
        self, *, state_tag: object, adapter_payload: Any
    ) -> SvgPosition:
        """Build Morpion position from incoming generic adapter payload."""
        if not isinstance(adapter_payload, MorpionDisplayPayload):
            raise InvalidSvgAdapterPayloadTypeError(
                adapter_name=self.__class__.__name__,
                expected_type=MorpionDisplayPayload,
                actual_value=adapter_payload,
            )
        self._payload = adapter_payload
        self._buttons = ()
        return SvgPosition(state_tag=state_tag, payload=adapter_payload)

    def _build_buttons(
        self,
        *,
        payload: MorpionDisplayPayload,
        left: float,
        top: float,
        width: float,
        bottom: float,
    ) -> tuple[tuple[_ActionButton, ...], float]:
        """Return vertical action buttons sized to the available list area."""
        action_count = len(payload.legal_actions)
        if action_count == 0:
            return (), 0.0

        available_height = max(bottom - top, 1.0)
        gap = _clamp(available_height * 0.012, 2.0, 6.0)
        total_gap = gap * max(action_count - 1, 0)
        button_height = max(16.0, (available_height - total_gap) / action_count)

        buttons = tuple(
            _ActionButton(
                action_name=action_name,
                x=left,
                y=top + index * (button_height + gap),
                width=width,
                height=button_height,
            )
            for index, action_name in enumerate(payload.legal_actions)
        )
        widest_label = max(payload.legal_actions, key=len)
        font_size = _fit_font_size(
            text=widest_label,
            max_width=width * 0.92,
            min_size=7.0,
            max_size=min(13.0, button_height * 0.55),
        )
        return buttons, font_size

    def render_svg(
        self,
        pos: SvgPosition,
        size: int,
        *,
        margin: int = 0,
    ) -> RenderResult:
        """Render a simple summary panel with one clickable action row per move."""
        payload = pos.payload
        if not isinstance(payload, MorpionDisplayPayload):
            raise InvalidSvgAdapterPayloadTypeError(
                adapter_name=self.__class__.__name__,
                expected_type=MorpionDisplayPayload,
                actual_value=payload,
            )

        width = float(size)
        height = float(size)
        outer_margin = min(float(margin), min(width, height) / 2.0)
        padding = _clamp(size * 0.03, 12.0, 20.0)

        left = outer_margin + padding
        top = outer_margin + padding
        right = width - outer_margin - padding
        bottom = height - outer_margin - padding
        content_width = max(right - left, 1.0)
        center_x = left + content_width / 2.0

        title_font = _fit_font_size(
            text=self._TITLE,
            max_width=content_width,
            min_size=16.0,
            max_size=28.0,
        )
        meta_font = _fit_font_size(
            text=f"variant={payload.variant} moves={payload.moves} points={payload.point_count}",
            max_width=content_width,
            min_size=10.0,
            max_size=16.0,
            width_factor=0.58,
        )
        info_font = _fit_font_size(
            text=f"legal moves: {len(payload.legal_actions)}",
            max_width=content_width,
            min_size=9.0,
            max_size=14.0,
        )

        cursor = top
        title_y = cursor + title_font * 0.7
        cursor += title_font * 1.5
        variant_y = cursor + meta_font * 0.6
        cursor += meta_font * 1.35
        moves_y = cursor + meta_font * 0.6
        cursor += meta_font * 1.35
        status_text = (
            self._TERMINAL_MESSAGE
            if payload.is_terminal
            else f"legal moves: {len(payload.legal_actions)}"
        )
        status_y = cursor + info_font * 0.6
        cursor += info_font * 1.6

        buttons, button_font = self._build_buttons(
            payload=payload,
            left=left,
            top=cursor,
            width=content_width,
            bottom=bottom,
        )
        self._buttons = buttons

        elements = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" '
                f'height="{size}" viewBox="0 0 {size} {size}">'
            ),
            f'<rect x="0" y="0" width="{size}" height="{size}" fill="#f8fafc"/>',
            (
                f'<rect x="{_fmt(left - 6)}" y="{_fmt(top - 6)}" '
                f'width="{_fmt(content_width + 12)}" height="{_fmt(bottom - top + 12)}" '
                'rx="14" ry="14" fill="#ffffff" stroke="#cbd5e1" stroke-width="2"/>'
            ),
            (
                f'<text x="{_fmt(center_x)}" y="{_fmt(title_y)}" text-anchor="middle" '
                f'font-size="{_fmt(title_font)}" font-family="sans-serif" '
                'fill="#0f172a">Morpion Solitaire</text>'
            ),
            (
                f'<text x="{_fmt(center_x)}" y="{_fmt(variant_y)}" text-anchor="middle" '
                f'font-size="{_fmt(meta_font)}" font-family="monospace" '
                f'fill="#334155">{escape(f"variant = {payload.variant}")}</text>'
            ),
            (
                f'<text x="{_fmt(center_x)}" y="{_fmt(moves_y)}" text-anchor="middle" '
                f'font-size="{_fmt(meta_font)}" font-family="monospace" '
                f'fill="#334155">{escape(f"moves = {payload.moves} | points = {payload.point_count}")}</text>'
            ),
            (
                f'<text x="{_fmt(center_x)}" y="{_fmt(status_y)}" text-anchor="middle" '
                f'font-size="{_fmt(info_font)}" font-family="sans-serif" '
                f'fill="{"#065f46" if payload.is_terminal else "#475569"}">{escape(status_text)}</text>'
            ),
        ]

        for button in buttons:
            elements.extend(
                [
                    (
                        f'<rect x="{_fmt(button.x)}" y="{_fmt(button.y)}" '
                        f'width="{_fmt(button.width)}" height="{_fmt(button.height)}" '
                        'rx="8" ry="8" fill="#e2e8f0" stroke="#94a3b8" stroke-width="1.5"/>'
                    ),
                    (
                        f'<text x="{_fmt(button.x + button.width / 2.0)}" '
                        f'y="{_fmt(button.y + button.height / 2.0)}" text-anchor="middle" '
                        'dominant-baseline="middle" '
                        f'font-size="{_fmt(button_font)}" font-family="monospace" '
                        f'fill="#0f172a">{escape(button.action_name)}</text>'
                    ),
                ]
            )

        elements.append("</svg>")
        info = {
            "fen": (
                f"variant={payload.variant} moves={payload.moves} points={payload.point_count}"
            ),
            "legal_moves": ", ".join(payload.legal_actions),
        }
        return RenderResult(svg_bytes="\n".join(elements).encode("utf-8"), info=info)

    def handle_click(
        self,
        pos: SvgPosition,
        *,
        x: int,
        y: int,
        board_size: int,
        margin: int,
    ) -> ClickResult:
        """Handle clicks by returning the action name for the clicked row."""
        _ = pos
        _ = board_size
        _ = margin
        for button in self._buttons:
            if button.contains(x=x, y=y):
                return ClickResult(
                    action_name=button.action_name,
                    interaction_continues=False,
                )
        return ClickResult(action_name=None, interaction_continues=True)

    def reset_interaction(self) -> None:
        """Reset transient click/selection interaction state."""
        self._buttons = ()

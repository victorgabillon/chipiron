"""Integer reduction SVG adapter for the generic GUI."""

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
from chipiron.displays.svg_text_helpers import fit_font_size, fmt_svg_number
from chipiron.environments.integer_reduction.integer_reduction_gui_encoder import (
    IntegerReductionDisplayPayload,
)


@dataclass(frozen=True, slots=True)
class _ActionButton:
    """Clickable rectangle associated with one integer-reduction action."""

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


@dataclass(frozen=True, slots=True)
class _TextLayout:
    """Precomputed centered text block constrained to the content box."""

    text: str
    x: float
    y: float
    font_size: float
    font_family: str
    fill: str
    width_factor: float
    max_width: float


@dataclass(frozen=True, slots=True)
class _PanelLayout:
    """Resolved integer-reduction layout for a single render pass."""

    title: _TextLayout
    value: _TextLayout
    steps: _TextLayout
    instruction: _TextLayout
    buttons: tuple[_ActionButton, ...]
    button_font_size: float


def _clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a scalar into a closed interval."""
    return min(max(value, lower), upper)


def _estimate_text_width(*, text: str, font_size: float, width_factor: float) -> float:
    """Estimate SVG text width without requiring font metrics."""
    return len(text) * font_size * width_factor


@dataclass
class IntegerReductionSvgAdapter(SvgGameAdapter):
    """SVG adapter implementation for integer reduction."""

    game_name: str = "integer_reduction"
    board_side: int = 1

    _payload: IntegerReductionDisplayPayload | None = None
    _buttons: tuple[_ActionButton, ...] = ()

    _TITLE = "Integer Reduction"
    _ACTION_PROMPT = "Choose a reduction"
    _TERMINAL_MESSAGE = "Reached 1"

    def position_from_update(
        self, *, state_tag: object, adapter_payload: Any
    ) -> SvgPosition:
        """Build integer-reduction position from incoming generic adapter payload."""
        if not isinstance(adapter_payload, IntegerReductionDisplayPayload):
            raise InvalidSvgAdapterPayloadTypeError(
                adapter_name=self.__class__.__name__,
                expected_type=IntegerReductionDisplayPayload,
                actual_value=adapter_payload,
            )
        self._payload = adapter_payload
        self._buttons = ()
        return SvgPosition(state_tag=state_tag, payload=adapter_payload)

    def _build_layout(
        self,
        *,
        size: int,
        margin: int,
        payload: IntegerReductionDisplayPayload,
    ) -> _PanelLayout:
        """Build a single consistent layout from the current viewport bounds.

        The adapter treats the square SVG viewport as the source of truth and derives
        every text/button coordinate from one content box inside that viewport.
        """
        width = float(size)
        height = float(size)
        outer_margin = min(float(margin), min(width, height) / 2.0)
        available_width = max(width - 2.0 * outer_margin, 1.0)
        available_height = max(height - 2.0 * outer_margin, 1.0)
        padding = min(
            max(10.0, min(width, height) * 0.035),
            min(available_width, available_height) / 4.0,
        )

        left = outer_margin + padding
        right = width - outer_margin - padding
        top = outer_margin + padding
        bottom = height - outer_margin - padding
        content_width = max(right - left, 1.0)
        content_height = max(bottom - top, 1.0)
        center_x = left + content_width / 2.0

        title_font = fit_font_size(
            text=self._TITLE,
            max_width=content_width,
            min_size=12.0,
            max_size=min(34.0, content_height * 0.12),
            width_factor=0.56,
        )
        value_text = f"n = {payload.value}"
        value_font = fit_font_size(
            text=value_text,
            max_width=content_width,
            min_size=18.0,
            max_size=min(72.0, content_height * 0.26),
            width_factor=0.62,
        )
        steps_text = f"steps = {payload.steps}"
        steps_font = fit_font_size(
            text=steps_text,
            max_width=content_width,
            min_size=12.0,
            max_size=min(30.0, content_height * 0.11),
            width_factor=0.6,
        )
        instruction_text = (
            self._TERMINAL_MESSAGE if payload.is_terminal else self._ACTION_PROMPT
        )
        instruction_font = fit_font_size(
            text=instruction_text,
            max_width=content_width,
            min_size=11.0,
            max_size=min(22.0, content_height * 0.085),
            width_factor=0.55,
        )

        title_line = title_font * 1.2
        value_line = value_font * 1.1
        steps_line = steps_font * 1.15
        instruction_line = instruction_font * 1.2
        gap_small = _clamp(content_height * 0.03, 8.0, 18.0)
        gap_medium = _clamp(content_height * 0.04, 10.0, 24.0)

        header_height = (
            title_line
            + gap_small
            + value_line
            + gap_small
            + steps_line
            + gap_small
            + instruction_line
        )
        max_header_ratio = 0.56 if payload.is_terminal else 0.62
        max_header_height = content_height * max_header_ratio
        if header_height > max_header_height and header_height > 0.0:
            scale = max_header_height / header_height
            title_font *= scale
            value_font *= scale
            steps_font *= scale
            instruction_font *= scale
            title_line *= scale
            value_line *= scale
            steps_line *= scale
            instruction_line *= scale
            gap_small *= scale
            gap_medium *= scale

        cursor = top
        title_y = cursor + title_line / 2.0
        cursor += title_line + gap_small
        value_y = cursor + value_line / 2.0
        cursor += value_line + gap_small
        steps_y = cursor + steps_line / 2.0
        cursor += steps_line + gap_small
        instruction_y = cursor + instruction_line / 2.0
        cursor += instruction_line + gap_medium

        buttons: tuple[_ActionButton, ...] = ()
        button_font_size = 0.0
        button_count = len(payload.legal_actions)
        if button_count > 0:
            button_width = content_width
            desired_button_height = _clamp(content_height * 0.13, 40.0, 56.0)
            desired_gap = _clamp(content_height * 0.035, 8.0, 18.0)
            available_button_height = max(bottom - cursor, 0.0)
            desired_total = (
                button_count * desired_button_height
                + max(button_count - 1, 0) * desired_gap
            )
            scale = (
                min(1.0, available_button_height / desired_total)
                if desired_total > 0.0
                else 1.0
            )
            button_height = desired_button_height * scale
            button_gap = desired_gap * scale

            buttons = tuple(
                _ActionButton(
                    action_name=action_name,
                    x=left,
                    y=cursor + index * (button_height + button_gap),
                    width=button_width,
                    height=button_height,
                )
                for index, action_name in enumerate(payload.legal_actions)
            )

            widest_button_label = max(payload.legal_actions, key=len)
            button_font_size = fit_font_size(
                text=widest_button_label,
                max_width=button_width * 0.82,
                min_size=12.0,
                max_size=min(28.0, button_height * 0.42),
                width_factor=0.56,
            )

        return _PanelLayout(
            title=_TextLayout(
                text=self._TITLE,
                x=center_x,
                y=title_y,
                font_size=title_font,
                font_family="sans-serif",
                fill="#111827",
                width_factor=0.56,
                max_width=content_width,
            ),
            value=_TextLayout(
                text=value_text,
                x=center_x,
                y=value_y,
                font_size=value_font,
                font_family="monospace",
                fill="#0f172a",
                width_factor=0.62,
                max_width=content_width,
            ),
            steps=_TextLayout(
                text=steps_text,
                x=center_x,
                y=steps_y,
                font_size=steps_font,
                font_family="monospace",
                fill="#334155",
                width_factor=0.6,
                max_width=content_width,
            ),
            instruction=_TextLayout(
                text=instruction_text,
                x=center_x,
                y=instruction_y,
                font_size=instruction_font,
                font_family="sans-serif",
                fill="#065f46" if payload.is_terminal else "#475569",
                width_factor=0.55,
                max_width=content_width,
            ),
            buttons=buttons,
            button_font_size=button_font_size,
        )

    def _render_text(self, text_layout: _TextLayout) -> str:
        """Render a centered text node and compress it when width is tight."""
        attributes = [
            f'x="{fmt_svg_number(text_layout.x)}"',
            f'y="{fmt_svg_number(text_layout.y)}"',
            'text-anchor="middle"',
            'dominant-baseline="middle"',
            f'font-size="{fmt_svg_number(text_layout.font_size)}"',
            f'font-family="{text_layout.font_family}"',
            f'fill="{text_layout.fill}"',
        ]

        if (
            _estimate_text_width(
                text=text_layout.text,
                font_size=text_layout.font_size,
                width_factor=text_layout.width_factor,
            )
            > text_layout.max_width
        ):
            attributes.append(f'textLength="{fmt_svg_number(text_layout.max_width)}"')
            attributes.append('lengthAdjust="spacingAndGlyphs"')

        return f"<text {' '.join(attributes)}>{escape(text_layout.text)}</text>"

    def render_svg(
        self, pos: SvgPosition, size: int, *, margin: int = 0
    ) -> RenderResult:
        """Render an SVG control panel for the integer-reduction game."""
        del pos
        if self._payload is None:
            raise InvalidSvgAdapterPayloadTypeError(
                adapter_name=self.__class__.__name__,
                expected_type=IntegerReductionDisplayPayload,
                actual_value=None,
            )

        payload = self._payload
        width = size
        height = size
        layout = self._build_layout(size=size, margin=margin, payload=payload)
        self._buttons = layout.buttons

        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            f'<rect width="{width}" height="{height}" fill="#f8fafc"/>',
            self._render_text(layout.title),
            self._render_text(layout.value),
            self._render_text(layout.steps),
            self._render_text(layout.instruction),
        ]

        for button in layout.buttons:
            parts.append(
                f'<rect x="{fmt_svg_number(button.x)}" y="{fmt_svg_number(button.y)}" '
                f'width="{fmt_svg_number(button.width)}" height="{fmt_svg_number(button.height)}" '
                'rx="14" fill="#dbeafe" stroke="#2563eb" stroke-width="2"/>'
            )
            parts.append(
                self._render_text(
                    _TextLayout(
                        text=button.action_name,
                        x=button.x + button.width / 2.0,
                        y=button.y + button.height / 2.0,
                        font_size=layout.button_font_size,
                        font_family="sans-serif",
                        fill="#1d4ed8",
                        width_factor=0.56,
                        max_width=button.width * 0.82,
                    )
                )
            )

        parts.append("</svg>")

        return RenderResult(
            svg_bytes="".join(parts).encode("utf-8"),
            info={
                "round": "-",
                "fen": f"value={payload.value} steps={payload.steps}",
                "legal_move_count": str(len(payload.legal_actions)),
                "legal_moves": ", ".join(payload.legal_actions) or "(terminal)",
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
        """Convert click coordinates into an integer-reduction action name."""
        del pos
        del board_size
        del margin
        for button in self._buttons:
            if button.contains(x=x, y=y):
                return ClickResult(
                    action_name=button.action_name,
                    interaction_continues=False,
                )
        return ClickResult(action_name=None, interaction_continues=False)

    def reset_interaction(self) -> None:
        """Reset click state for integer-reduction interactions."""
        self._buttons = ()

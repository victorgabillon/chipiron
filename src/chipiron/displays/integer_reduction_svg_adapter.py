"""Integer reduction SVG adapter for the generic GUI."""

from dataclasses import dataclass
from typing import Any

from chipiron.displays.svg_adapter_errors import InvalidSvgAdapterPayloadTypeError
from chipiron.displays.svg_adapter_protocol import (
    ClickResult,
    RenderResult,
    SvgGameAdapter,
    SvgPosition,
)
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


@dataclass
class IntegerReductionSvgAdapter(SvgGameAdapter):
    """SVG adapter implementation for integer reduction."""

    game_name: str = "integer_reduction"
    board_side: int = 1

    _payload: IntegerReductionDisplayPayload | None = None
    _buttons: tuple[_ActionButton, ...] = ()

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
        body_top = margin + 50
        button_width = max(220.0, width - 2 * (margin + 40))
        button_height = 56.0
        button_gap = 18.0
        start_x = (width - button_width) / 2
        start_y = body_top + 120

        buttons = tuple(
            _ActionButton(
                action_name=action_name,
                x=start_x,
                y=start_y + index * (button_height + button_gap),
                width=button_width,
                height=button_height,
            )
            for index, action_name in enumerate(payload.legal_actions)
        )
        self._buttons = buttons

        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            f'<rect width="{width}" height="{height}" fill="#f8fafc"/>',
            '<text x="50%" y="80" text-anchor="middle" font-size="34" '
            'font-family="sans-serif" fill="#111827">Integer Reduction</text>',
            f'<text x="50%" y="{body_top + 20}" text-anchor="middle" font-size="72" '
            f'font-family="monospace" fill="#0f172a">n = {payload.value}</text>',
        ]

        if not buttons:
            parts.append(
                f'<text x="50%" y="{body_top + 120}" text-anchor="middle" '
                'font-size="28" font-family="sans-serif" fill="#065f46">'
                "Reached 1"
                "</text>"
            )
        else:
            parts.append(
                f'<text x="50%" y="{body_top + 70}" text-anchor="middle" '
                'font-size="22" font-family="sans-serif" fill="#475569">'
                "Choose a reduction"
                "</text>"
            )
            for button in buttons:
                parts.append(
                    f'<rect x="{button.x}" y="{button.y}" width="{button.width}" '
                    f'height="{button.height}" rx="14" fill="#dbeafe" stroke="#2563eb" '
                    'stroke-width="2"/>'
                )
                parts.append(
                    f'<text x="{button.x + button.width / 2}" '
                    f'y="{button.y + button.height / 2 + 8}" text-anchor="middle" '
                    'font-size="28" font-family="sans-serif" fill="#1d4ed8">'
                    f"{button.action_name}</text>"
                )

        parts.append("</svg>")

        return RenderResult(
            svg_bytes="".join(parts).encode("utf-8"),
            info={
                "round": "-",
                "fen": f"value={payload.value}",
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

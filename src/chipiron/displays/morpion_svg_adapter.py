"""Board-based Morpion SVG adapter for the generic GUI."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from html import escape
from typing import Any

from chipiron.displays.svg_adapter_errors import InvalidSvgAdapterPayloadTypeError
from chipiron.displays.svg_adapter_protocol import (
    ClickResult,
    RenderResult,
    SvgGameAdapter,
    SvgPosition,
)
from chipiron.environments.morpion.morpion_gui_encoder import (
    MorpionDisplayPayload,
    MorpionMoveDisplay,
)

type DisplayPoint = tuple[int, int]
type DisplaySegment = tuple[DisplayPoint, DisplayPoint]


def _clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a scalar into a closed interval."""
    return min(max(value, lower), upper)


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


def _collect_geometry_points(payload: MorpionDisplayPayload) -> list[DisplayPoint]:
    """Collect all points that should influence the board transform."""
    geometry_points = list(payload.points)
    geometry_points.extend(point for segment in payload.segments for point in segment)
    for move in payload.legal_moves:
        geometry_points.append(move.new_point)
        geometry_points.extend(move.segment)
    return geometry_points or [(0, 0)]


@dataclass
class MorpionSvgAdapter(SvgGameAdapter):
    """SVG adapter implementation for Morpion."""

    game_name: str = "morpion"
    board_side: int = 1

    _payload: MorpionDisplayPayload | None = None
    _click_targets: list[tuple[str, float, float]] = field(default_factory=list)
    _click_radius: float = 12.0

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
        self._click_targets = []
        return SvgPosition(state_tag=state_tag, payload=adapter_payload)

    def _build_transform(
        self,
        *,
        payload: MorpionDisplayPayload,
        size: int,
        margin: int,
    ) -> tuple[float, float, float, float, float, float]:
        """Return transform inputs for mapping Morpion grid points to SVG coordinates."""
        width = float(size)
        height = float(size)
        outer_margin = min(float(margin), min(width, height) / 2.0)
        padding = _clamp(size * 0.035, 16.0, 28.0)
        header_height = _clamp(size * 0.14, 52.0, 88.0)

        board_left = outer_margin + padding
        board_top = outer_margin + header_height
        board_right = width - outer_margin - padding
        board_bottom = height - outer_margin - padding
        board_width = max(board_right - board_left, 1.0)
        board_height = max(board_bottom - board_top, 1.0)

        geometry_points = _collect_geometry_points(payload)
        xs = [point[0] for point in geometry_points]
        ys = [point[1] for point in geometry_points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        span_x = max(max_x - min_x, 1)
        span_y = max(max_y - min_y, 1)
        grid_padding = 1.5
        scale = min(
            board_width / (float(span_x) + 2.0 * grid_padding),
            board_height / (float(span_y) + 2.0 * grid_padding),
        )

        used_width = (float(span_x) + 2.0 * grid_padding) * scale
        used_height = (float(span_y) + 2.0 * grid_padding) * scale
        offset_x = board_left + (board_width - used_width) / 2.0
        offset_y = board_top + (board_height - used_height) / 2.0
        return float(min_x), float(max_y), float(grid_padding), offset_x, offset_y, scale

    def _to_svg_point(
        self,
        point: DisplayPoint,
        *,
        min_x: float,
        max_y: float,
        grid_padding: float,
        offset_x: float,
        offset_y: float,
        scale: float,
    ) -> tuple[float, float]:
        """Map one Morpion grid point to SVG coordinates."""
        x = offset_x + ((float(point[0]) - min_x) + grid_padding) * scale
        y = offset_y + ((max_y - float(point[1])) + grid_padding) * scale
        return x, y

    def _render_segment(
        self,
        *,
        segment: DisplaySegment,
        stroke: str,
        stroke_width: float,
        opacity: float,
        transform: tuple[float, float, float, float, float, float],
    ) -> str:
        """Render one SVG line from Morpion grid coordinates."""
        start, end = segment
        min_x, max_y, grid_padding, offset_x, offset_y, scale = transform
        x1, y1 = self._to_svg_point(
            start,
            min_x=min_x,
            max_y=max_y,
            grid_padding=grid_padding,
            offset_x=offset_x,
            offset_y=offset_y,
            scale=scale,
        )
        x2, y2 = self._to_svg_point(
            end,
            min_x=min_x,
            max_y=max_y,
            grid_padding=grid_padding,
            offset_x=offset_x,
            offset_y=offset_y,
            scale=scale,
        )
        return (
            f'<line x1="{_fmt(x1)}" y1="{_fmt(y1)}" x2="{_fmt(x2)}" y2="{_fmt(y2)}" '
            f'stroke="{stroke}" stroke-width="{_fmt(stroke_width)}" opacity="{opacity:.2f}"/>'
        )

    def _render_point(
        self,
        *,
        point: DisplayPoint,
        radius: float,
        fill: str,
        opacity: float,
        transform: tuple[float, float, float, float, float, float],
        stroke: str | None = None,
        stroke_width: float = 0.0,
    ) -> str:
        """Render one SVG circle from Morpion grid coordinates."""
        min_x, max_y, grid_padding, offset_x, offset_y, scale = transform
        cx, cy = self._to_svg_point(
            point,
            min_x=min_x,
            max_y=max_y,
            grid_padding=grid_padding,
            offset_x=offset_x,
            offset_y=offset_y,
            scale=scale,
        )
        stroke_attrs = ""
        if stroke is not None and stroke_width > 0.0:
            stroke_attrs = (
                f' stroke="{stroke}" stroke-width="{_fmt(stroke_width)}"'
            )
        return (
            f'<circle cx="{_fmt(cx)}" cy="{_fmt(cy)}" r="{_fmt(radius)}" '
            f'fill="{fill}" opacity="{opacity:.2f}"{stroke_attrs}/>'
        )

    def _register_click_target(
        self,
        *,
        move: MorpionMoveDisplay,
        transform: tuple[float, float, float, float, float, float],
    ) -> None:
        """Store one clickable move target near the preview point."""
        min_x, max_y, grid_padding, offset_x, offset_y, scale = transform
        cx, cy = self._to_svg_point(
            move.new_point,
            min_x=min_x,
            max_y=max_y,
            grid_padding=grid_padding,
            offset_x=offset_x,
            offset_y=offset_y,
            scale=scale,
        )
        self._click_targets.append((move.action_name, cx, cy))

    def render_svg(
        self,
        pos: SvgPosition,
        size: int,
        *,
        margin: int = 0,
    ) -> RenderResult:
        """Render a Morpion board with clickable legal-move previews."""
        payload = pos.payload
        if not isinstance(payload, MorpionDisplayPayload):
            raise InvalidSvgAdapterPayloadTypeError(
                adapter_name=self.__class__.__name__,
                expected_type=MorpionDisplayPayload,
                actual_value=payload,
            )

        width = float(size)
        height = float(size)
        transform = self._build_transform(payload=payload, size=size, margin=margin)
        self._click_targets = []

        title_font = _fit_font_size(
            text=self._TITLE,
            max_width=width * 0.72,
            min_size=15.0,
            max_size=24.0,
        )
        meta_font = _fit_font_size(
            text=f"variant={payload.variant} moves={payload.moves} points={payload.point_count}",
            max_width=width * 0.88,
            min_size=9.0,
            max_size=14.0,
            width_factor=0.58,
        )
        status_font = _fit_font_size(
            text=f"legal previews: {len(payload.legal_moves)}",
            max_width=width * 0.88,
            min_size=9.0,
            max_size=13.0,
        )

        scale = transform[-1]
        point_radius = _clamp(scale * 0.11, 2.5, 4.0)
        preview_radius = _clamp(scale * 0.22, 5.0, 8.0)
        segment_width = _clamp(scale * 0.10, 1.5, 2.6)
        preview_width = _clamp(scale * 0.11, 1.8, 3.0)
        self._click_radius = max(12.0, preview_radius * 1.6)

        status_text = (
            self._TERMINAL_MESSAGE
            if payload.is_terminal
            else f"legal previews: {len(payload.legal_moves)}"
        )
        title_y = _clamp(height * 0.06, 26.0, 38.0)
        meta_y = title_y + title_font * 0.95
        status_y = meta_y + meta_font * 1.1

        elements = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" '
                f'height="{size}" viewBox="0 0 {size} {size}">'
            ),
            f'<rect x="0" y="0" width="{size}" height="{size}" fill="#f8fafc"/>',
            (
                f'<text x="{_fmt(width / 2.0)}" y="{_fmt(title_y)}" text-anchor="middle" '
                f'font-size="{_fmt(title_font)}" font-family="sans-serif" '
                'fill="#0f172a">Morpion Solitaire</text>'
            ),
            (
                f'<text x="{_fmt(width / 2.0)}" y="{_fmt(meta_y)}" text-anchor="middle" '
                f'font-size="{_fmt(meta_font)}" font-family="monospace" '
                f'fill="#334155">{escape(f"variant = {payload.variant} | moves = {payload.moves} | points = {payload.point_count}")}</text>'
            ),
            (
                f'<text x="{_fmt(width / 2.0)}" y="{_fmt(status_y)}" text-anchor="middle" '
                f'font-size="{_fmt(status_font)}" font-family="sans-serif" '
                f'fill="{"#065f46" if payload.is_terminal else "#475569"}">{escape(status_text)}</text>'
            ),
        ]

        for segment in payload.segments:
            elements.append(
                self._render_segment(
                    segment=segment,
                    stroke="#0f172a",
                    stroke_width=segment_width,
                    opacity=1.0,
                    transform=transform,
                )
            )

        for move in payload.legal_moves:
            elements.append(
                self._render_segment(
                    segment=move.segment,
                    stroke="#22c55e",
                    stroke_width=preview_width,
                    opacity=0.60,
                    transform=transform,
                )
            )

        for point in payload.points:
            elements.append(
                self._render_point(
                    point=point,
                    radius=point_radius,
                    fill="#0f172a",
                    opacity=1.0,
                    transform=transform,
                )
            )

        for move in payload.legal_moves:
            self._register_click_target(move=move, transform=transform)
            elements.append(
                self._render_point(
                    point=move.new_point,
                    radius=preview_radius,
                    fill="#22c55e",
                    opacity=0.30,
                    transform=transform,
                    stroke="#16a34a",
                    stroke_width=max(1.0, preview_width * 0.45),
                )
            )

        elements.append("</svg>")
        info = {
            "fen": (
                f"variant={payload.variant} moves={payload.moves} points={payload.point_count}"
            ),
            "legal_moves": (
                ", ".join(move.action_name for move in payload.legal_moves) or "(terminal)"
            ),
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
        """Handle clicks by selecting the preview nearest the click point."""
        _ = pos
        _ = board_size
        _ = margin
        for action_name, cx, cy in self._click_targets:
            if math.hypot(float(x) - cx, float(y) - cy) <= self._click_radius:
                return ClickResult(
                    action_name=action_name,
                    interaction_continues=False,
                )
        return ClickResult(action_name=None, interaction_continues=True)

    def reset_interaction(self) -> None:
        """Reset transient click/selection interaction state."""
        self._click_targets = []

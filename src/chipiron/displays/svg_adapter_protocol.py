"""Protocol and dataclasses for game-specific SVG adapters."""

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class SvgPosition:
    """Opaque position wrapper owned by the adapter."""

    state_tag: object
    payload: Any


@dataclass(frozen=True, slots=True)
class ClickResult:
    """Result of a click handling operation."""

    action_name: str | None
    interaction_continues: bool


@dataclass(frozen=True, slots=True)
class RenderResult:
    """Result of rendering board SVG and optional display metadata."""

    svg_bytes: bytes
    info: dict[str, str]


class SvgGameAdapter(Protocol):
    """Protocol for game-specific SVG rendering and click handling."""

    game_name: str
    board_side: int

    def position_from_update(self, *, state_tag: object, adapter_payload: Any) -> SvgPosition:
        """Build adapter position from generic update payload."""
        ...

    def render_svg(self, pos: SvgPosition, size: int) -> RenderResult:
        """Render an SVG board for the provided position."""
        ...

    def handle_click(
        self,
        pos: SvgPosition,
        *,
        x: int,
        y: int,
        board_size: int,
        margin: int,
    ) -> ClickResult:
        """Handle a click and optionally produce a finalized action name."""
        ...

    def reset_interaction(self) -> None:
        """Reset transient click/selection interaction state."""
        ...

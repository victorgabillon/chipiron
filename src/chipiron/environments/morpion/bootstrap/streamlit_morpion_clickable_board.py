"""Minimal Streamlit component wrapper for clickable Morpion SVG boards."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping


@lru_cache(maxsize=1)
def _morpion_clickable_board_component() -> Any:
    """Return the declared Streamlit component for clickable Morpion boards."""
    import streamlit.components.v1 as components

    frontend_dir = Path(__file__).with_name("streamlit_morpion_clickable_board_frontend")
    return components.declare_component(
        "morpion_clickable_board",
        path=str(frontend_dir),
    )


def render_clickable_morpion_board(
    *,
    svg: str,
    click_targets: Iterable[Any],
    click_radius: float,
    height: int,
    render_size: int,
    key: str,
) -> Mapping[str, Any] | None:
    """Render one clickable Morpion board component and return the latest click event."""
    component = _morpion_clickable_board_component()
    serialized_targets = [
        {
            "action_name": getattr(target, "action_name"),
            "center_x": getattr(target, "center_x"),
            "center_y": getattr(target, "center_y"),
        }
        for target in click_targets
    ]
    value = component(
        svg=svg,
        click_targets=serialized_targets,
        click_radius=click_radius,
        height=height,
        render_size=render_size,
        default=None,
        key=key,
    )
    return value if isinstance(value, Mapping) else None

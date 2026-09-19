"""Dashboard certified/frontier record status rendering."""

from __future__ import annotations

from typing import Any

from chipiron.environments.morpion.bootstrap.dashboard.formatting import (
    format_value as _format_value,
)

__all__ = ["render_record_status_section"]


def render_record_status_section(
    *,
    st: Any,
    certified_status: Any,
    frontier_status: Any,
) -> None:
    """Render a strict certified record summary alongside the frontier best."""
    certified_columns = st.columns(4)
    if certified_status is None or certified_status.current_best_total_points is None:
        certified_columns[0].metric(
            "Certified record total points", "No certified record yet"
        )
        certified_columns[1].metric("Certified record moves", "n/a")
        certified_columns[2].metric("Certified exact", "n/a")
        certified_columns[3].metric("Certified terminal", "n/a")
    else:
        certified_columns[0].metric(
            "Certified record total points",
            _format_value(certified_status.current_best_total_points),
        )
        certified_columns[1].metric(
            "Certified record moves",
            _format_value(certified_status.current_best_moves_since_start),
        )
        certified_columns[2].metric(
            "Certified exact",
            _format_value(certified_status.current_best_is_exact),
        )
        certified_columns[3].metric(
            "Certified terminal",
            _format_value(certified_status.current_best_is_terminal),
        )

    frontier_columns = st.columns(4)
    frontier_columns[0].metric(
        "Frontier best total points",
        _format_value(
            None
            if frontier_status is None
            else frontier_status.current_best_total_points
        ),
    )
    frontier_columns[1].metric(
        "Frontier best moves",
        _format_value(
            None
            if frontier_status is None
            else frontier_status.current_best_moves_since_start
        ),
    )
    frontier_columns[2].metric(
        "Frontier exact",
        _format_value(
            None if frontier_status is None else frontier_status.current_best_is_exact
        ),
    )
    frontier_columns[3].metric(
        "Frontier terminal",
        _format_value(
            None
            if frontier_status is None
            else frontier_status.current_best_is_terminal
        ),
    )
    st.caption(
        "Frontier best source: "
        + _format_value(
            None if frontier_status is None else frontier_status.current_best_source
        )
    )

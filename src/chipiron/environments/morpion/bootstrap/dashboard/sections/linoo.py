"""Dashboard Linoo selection-table rendering."""

from __future__ import annotations

from typing import Any

__all__ = ["linoo_selection_table_rows", "render_linoo_selection_table"]


def render_linoo_selection_table(
    *,
    st: Any,
    latest_linoo_selection_table: Any,
) -> None:
    """Render the latest persisted Linoo depth-selection table."""
    st.markdown("**Latest Linoo depth selection table**")
    st.caption(
        "Linoo selects the depth with minimal opened_count * (depth + 1), "
        "tie-breaking by smaller depth."
    )
    rows = linoo_selection_table_rows(latest_linoo_selection_table)
    if not rows:
        st.caption("No Linoo selection table available yet.")
        return
    st.dataframe(rows, width="stretch", hide_index=True)


def linoo_selection_table_rows(
    latest_linoo_selection_table: Any,
) -> list[dict[str, object]]:
    """Return dashboard rows for the latest Linoo table artifact."""
    rows = getattr(latest_linoo_selection_table, "rows", None)
    if rows is None:
        return []
    return [
        {
            "depth": row.depth,
            "opened_count": row.opened,
            "frontier_count": row.frontier,
            "deterministic_index": row.deterministic_index,
            "weight": row.weight,
            "probability": row.probability,
            "best_node_id": row.best_node,
            "best_direct_value": row.best_value,
            "selected": row.selected,
        }
        for row in rows
    ]

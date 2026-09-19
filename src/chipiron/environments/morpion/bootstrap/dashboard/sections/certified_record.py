"""Dashboard current certified-record board rendering."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from chipiron.environments.morpion.bootstrap.dashboard.formatting import (
    format_value as _format_value,
)

if TYPE_CHECKING:
    from chipiron.environments.morpion.bootstrap.dashboard.history_view import (
        MorpionBootstrapCertifiedRecordBoardView,
    )

__all__ = ["render_current_certified_record_board_section"]


def render_current_certified_record_board_section(
    *,
    st: Any,
    board_view: MorpionBootstrapCertifiedRecordBoardView | None,
) -> None:
    """Render the current strict certified Morpion record board when available."""
    if board_view is None:
        st.caption("No certified record state available yet.")
        return

    summary_columns = st.columns(5)
    summary_columns[0].metric("Total Points", str(board_view.total_points))
    summary_columns[1].metric("Moves Since Start", str(board_view.moves_since_start))
    summary_columns[2].metric("Exact", _format_value(board_view.is_exact))
    summary_columns[3].metric("Terminal", _format_value(board_view.is_terminal))
    summary_columns[4].metric("Source", board_view.source)
    st.components.v1.html(board_view.board_svg, height=760)
    if board_view.board_text is not None:
        st.code(board_view.board_text)

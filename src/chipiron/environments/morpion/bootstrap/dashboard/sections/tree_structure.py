"""Dashboard tree-structure and classification rendering."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from chipiron.environments.morpion.bootstrap.dashboard.formatting import (
    format_value as _format_value,
)
from chipiron.environments.morpion.bootstrap.dashboard.plot import (
    plot_tree_depth_distribution,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.plot import (
    render_plot,
)

if TYPE_CHECKING:
    from chipiron.environments.morpion.bootstrap.dashboard.history_view import (
        TreeDepthDistributionRow,
    )
    from chipiron.environments.morpion.bootstrap.history import (
        MorpionBootstrapTreeStatus,
    )

__all__ = [
    "render_tree_node_classification_summary",
    "render_tree_structure_section",
    "tree_structure_rows",
]


def _format_tree_node_classification_metric(
    count: int,
    *,
    total_nodes: int,
    percentages_available: bool,
) -> str:
    """Render one count and, when safe, its total-tree percentage."""
    if not percentages_available or total_nodes <= 0:
        return str(count)
    return f"{count} ({(count / total_nodes) * 100.0:.1f}%)"


def render_tree_node_classification_summary(
    *,
    st: Any,
    summary: Any,
) -> None:
    """Render compact exact and terminal node proportions for the latest tree."""
    if summary is None:
        st.caption("No latest tree snapshot classification summary available yet.")
        return

    unknown_nodes = getattr(summary, "unknown_classification_nodes", 0)
    total_nodes = getattr(summary, "total_nodes", 0)
    percentages_available = unknown_nodes == 0
    summary_columns = st.columns(5)
    summary_columns[0].metric("Total Nodes", str(total_nodes))
    summary_columns[1].metric(
        "Exact Nodes",
        _format_tree_node_classification_metric(
            getattr(summary, "exact_nodes", 0),
            total_nodes=total_nodes,
            percentages_available=percentages_available,
        ),
    )
    summary_columns[2].metric(
        "Terminal Nodes",
        _format_tree_node_classification_metric(
            getattr(summary, "terminal_nodes", 0),
            total_nodes=total_nodes,
            percentages_available=percentages_available,
        ),
    )
    summary_columns[3].metric(
        "Exact Terminal Nodes",
        _format_tree_node_classification_metric(
            getattr(summary, "exact_terminal_nodes", 0),
            total_nodes=total_nodes,
            percentages_available=percentages_available,
        ),
    )
    summary_columns[4].metric(
        "Non-Exact Non-Terminal",
        _format_tree_node_classification_metric(
            getattr(summary, "non_exact_non_terminal_nodes", 0),
            total_nodes=total_nodes,
            percentages_available=percentages_available,
        ),
    )
    if unknown_nodes > 0:
        st.caption(
            "Some snapshot nodes lack exact/terminal flags; percentages are omitted."
        )


def render_tree_structure_section(
    *,
    st: Any,
    tree_status: MorpionBootstrapTreeStatus | None,
    depth_distribution: tuple[TreeDepthDistributionRow, ...],
) -> None:
    """Render one compact tree-structure summary with per-depth counts."""
    if tree_status is None:
        st.caption("No tree structure has been recorded yet.")
        return

    summary_columns = st.columns(4)
    summary_columns[0].metric("Total Nodes", str(tree_status.num_nodes))
    summary_columns[1].metric(
        "Expanded Nodes",
        _format_value(tree_status.num_expanded_nodes),
    )
    summary_columns[2].metric("Min Depth", _format_value(tree_status.min_depth_present))
    summary_columns[3].metric("Max Depth", _format_value(tree_status.max_depth_present))

    if not depth_distribution:
        st.caption("No tree depth distribution available yet.")
        return
    render_plot(st, lambda: plot_tree_depth_distribution(depth_distribution))
    rows = tree_structure_rows(depth_distribution)
    st.dataframe(rows, width="stretch", hide_index=True)


def tree_structure_rows(
    depth_distribution: tuple[TreeDepthDistributionRow, ...],
) -> list[dict[str, int]]:
    """Return one dashboard-friendly per-depth node-count table."""
    return [
        {
            "depth": row.depth,
            "num_nodes": row.num_nodes,
            "cumulative_nodes": row.cumulative_nodes,
        }
        for row in depth_distribution
    ]

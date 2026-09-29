"""On-demand checkpoint navigation and bounded tree inspection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from chipiron.environments.morpion.bootstrap.dashboard.theme import page_header

if TYPE_CHECKING:
    from chipiron.environments.morpion.bootstrap.dashboard.view_model import (
        BootstrapDashboardSnapshot,
    )


def render(st: Any, snapshot: BootstrapDashboardSnapshot) -> None:
    """Only this active view loads checkpoint-backed inspection data."""
    from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
        MorpionBootstrapPaths,
    )
    from chipiron.environments.morpion.bootstrap.dashboard.data_cache import (
        cached_build_morpion_bootstrap_dashboard_data,
        cached_dashboard_data_freshness_tokens,
    )
    from chipiron.environments.morpion.bootstrap.dashboard.sections.tree_inspector import (
        render_tree_inspector_section,
    )
    from chipiron.environments.morpion.bootstrap.dashboard.sections.tree_structure import (
        render_tree_structure_section,
    )

    page_header(
        st,
        snapshot,
        "Tree inspector",
        "Navigate persisted states without changing the search.",
    )
    paths = MorpionBootstrapPaths.from_work_dir(snapshot.work_dir)
    try:
        data = cached_build_morpion_bootstrap_dashboard_data(
            str(paths.work_dir), cached_dashboard_data_freshness_tokens(paths)
        )
        render_tree_inspector_section(
            st=st,
            paths=paths,
            latest_linoo_selection_table=data.latest_linoo_selection_table,
            tree_node_classification_summary=data.latest_tree_node_classification_summary,
        )
        with st.expander("Whole-tree structure · saved summary"):
            render_tree_structure_section(
                st=st,
                tree_status=data.latest_tree_status,
                depth_distribution=data.latest_tree_depth_distribution,
            )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        st.warning(f"Checkpoint temporarily unavailable: {exc}")

"""On-demand checkpoint navigation and bounded tree inspection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from chipiron.environments.morpion.bootstrap.dashboard.theme import page_header

if TYPE_CHECKING:
    from chipiron.environments.morpion.bootstrap.dashboard.view_model import (
        BootstrapDashboardSnapshot,
    )


def render(st: Any, snapshot: BootstrapDashboardSnapshot) -> None:
    """Keep first paint light; load whole-tree training summaries only on request."""
    from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
        MorpionBootstrapPaths,
    )
    from chipiron.environments.morpion.bootstrap.dashboard.data_cache import (
        cached_build_morpion_bootstrap_tree_structure_data,
        cached_tree_structure_freshness_tokens,
    )
    from chipiron.environments.morpion.bootstrap.dashboard.history_view import (
        load_latest_linoo_selection_table_for_dashboard,
    )
    from chipiron.environments.morpion.bootstrap.dashboard.sections.tree_inspector import (
        render_tree_inspector_fragment,
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
    summary_key = f"morpion_bootstrap_load_tree_summary::{paths.work_dir}"
    load_saved_summary = bool(st.session_state.get(summary_key, False))

    try:
        latest_linoo_selection_table = load_latest_linoo_selection_table_for_dashboard(
            paths.work_dir
        )
        tree_structure_data = None
        if load_saved_summary:
            with st.spinner("Reading saved whole-tree summary…"):
                tree_structure_data = (
                    cached_build_morpion_bootstrap_tree_structure_data(
                        str(paths.work_dir),
                        cached_tree_structure_freshness_tokens(paths),
                    )
                )

        render_tree_inspector_fragment(
            st=st,
            paths=paths,
            latest_linoo_selection_table=latest_linoo_selection_table,
            tree_node_classification_summary=(
                None
                if tree_structure_data is None
                else tree_structure_data.latest_tree_node_classification_summary
            ),
        )
        with st.expander("Whole-tree structure · saved summary"):
            st.checkbox(
                "Load saved whole-tree statistics",
                key=summary_key,
                help=(
                    "This scans the saved training-tree snapshot. Leave it off for "
                    "normal node navigation."
                ),
            )
            if tree_structure_data is None:
                st.caption(
                    "Not loaded. Node inspection above uses the runtime checkpoint "
                    "index and does not need this full-tree scan."
                )
            else:
                if tree_structure_data.latest_tree_snapshot_status_message:
                    st.info(tree_structure_data.latest_tree_snapshot_status_message)
                render_tree_structure_section(
                    st=st,
                    tree_status=tree_structure_data.latest_tree_status,
                    depth_distribution=(
                        tree_structure_data.latest_tree_depth_distribution
                    ),
                )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        st.warning(f"Checkpoint temporarily unavailable: {exc}")

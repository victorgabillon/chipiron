"""Dashboard tree-inspector Streamlit rendering."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from chipiron.environments.morpion.bootstrap.dashboard.formatting import (
    format_bool_icon as _format_bool_icon,
)
from chipiron.environments.morpion.bootstrap.dashboard.formatting import (
    format_value as _format_value,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.linoo import (
    render_linoo_selection_table,
)
from chipiron.environments.morpion.bootstrap.dashboard.sections.tree_structure import (
    render_tree_node_classification_summary,
)
from chipiron.environments.morpion.bootstrap.dashboard.tree_inspector import (
    build_morpion_bootstrap_tree_inspector_snapshot,
)
from chipiron.environments.morpion.bootstrap.streamlit_morpion_clickable_board import (
    render_clickable_morpion_board,
)

if TYPE_CHECKING:
    from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
        MorpionBootstrapPaths,
    )
    from chipiron.environments.morpion.bootstrap.dashboard.tree_inspector import (
        MorpionBootstrapChildSummary,
    )

TREE_INSPECTOR_TIMING_PREFIX = "[tree-inspector-timing]"

__all__ = [
    "render_tree_inspector_fragment",
    "render_tree_inspector_section",
    "selected_child_node_id_for_branch",
    "tree_inspector_child_rows",
]


def _tree_inspector_rerun(st: Any) -> None:
    """Rerun only the tree-inspector fragment when supported."""
    try:
        st.rerun(scope="fragment")
    except TypeError:
        st.rerun()


def render_tree_inspector_section(
    *,
    st: Any,
    paths: MorpionBootstrapPaths,
    latest_linoo_selection_table: Any,
    tree_node_classification_summary: Any,
) -> None:
    """Render the bounded runtime-tree inspector for the latest checkpoint."""
    section_start_time = time.perf_counter()
    state_key = f"morpion_bootstrap_selected_node::{paths.work_dir}"
    selected_node_id = st.session_state.get(state_key)

    render_linoo_selection_table(
        st=st,
        latest_linoo_selection_table=latest_linoo_selection_table,
    )
    render_tree_node_classification_summary(
        st=st,
        summary=tree_node_classification_summary,
    )

    snapshot_start_time = time.perf_counter()
    snapshot = build_morpion_bootstrap_tree_inspector_snapshot(
        paths.work_dir,
        selected_node_id=selected_node_id,
    )
    snapshot_duration = time.perf_counter() - snapshot_start_time
    print(
        f"{TREE_INSPECTOR_TIMING_PREFIX} build_snapshot "
        f"work_dir={paths.work_dir.name} selected_node_id={selected_node_id!r} "
        f"checkpoint={None if snapshot.checkpoint_path is None else snapshot.checkpoint_path.name} "
        f"total_s={snapshot_duration:.6f}",
        flush=True,
    )

    if snapshot.status_message is not None:
        st.info(snapshot.status_message)
    if snapshot.error_message is not None:
        st.warning(snapshot.error_message)
        return
    if snapshot.selected_node_id is None or snapshot.node_summary is None:
        st.caption("No persisted runtime checkpoint available yet.")
        return
    if snapshot.selection_warning is not None:
        st.warning(snapshot.selection_warning)

    st.session_state[state_key] = snapshot.selected_node_id
    _render_tree_inspector_navigation(
        st=st,
        snapshot=snapshot,
        state_key=state_key,
    )

    node_summary = snapshot.node_summary
    summary_columns = st.columns(4)
    summary_columns[0].metric("Selected Node", node_summary.node_id)
    summary_columns[1].metric("Depth", _format_value(node_summary.depth))
    summary_columns[2].metric("Children", str(node_summary.num_children))
    summary_columns[3].metric(
        "Best Branch",
        _format_value(node_summary.best_branch_label),
    )

    details_columns = st.columns(2)
    with details_columns[0]:
        with st.expander("Node Summary"):
            st.json(_tree_inspector_node_summary_dict(snapshot))
        with st.expander("Local Tree"):
            st.json(_tree_inspector_local_tree_dict(snapshot))
    with details_columns[1]:
        if snapshot.state_view is not None:
            st.caption("Selected Morpion State")
            clickable_board_start_time = time.perf_counter()
            board_click_event = render_clickable_morpion_board(
                svg=snapshot.state_view.board_svg,
                click_targets=snapshot.state_view.board_click_targets,
                click_radius=snapshot.state_view.board_click_radius,
                height=760,
                render_size=snapshot.state_view.board_render_size,
                key=f"{state_key}::board::{snapshot.selected_node_id}",
            )
            clickable_board_duration = time.perf_counter() - clickable_board_start_time
            print(
                f"{TREE_INSPECTOR_TIMING_PREFIX} render_clickable_board "
                f"work_dir={paths.work_dir.name} selected_node_id={snapshot.selected_node_id!r} "
                f"total_s={clickable_board_duration:.6f}",
                flush=True,
            )
            if board_click_event is not None:
                click_nonce = board_click_event.get("click_nonce")
                click_nonce_key = f"{state_key}::board_click_nonce"
                if click_nonce != st.session_state.get(click_nonce_key):
                    st.session_state[click_nonce_key] = click_nonce
                    clicked_action_name = board_click_event.get("action_name")
                    if isinstance(clicked_action_name, str):
                        selected_child_node_id = selected_child_node_id_for_branch(
                            snapshot.child_summaries,
                            clicked_action_name,
                        )
                        if selected_child_node_id is not None:
                            st.session_state[state_key] = selected_child_node_id
                            _tree_inspector_rerun(st)
                        else:
                            st.caption(
                                f"Action {_format_value(clicked_action_name)} is not expanded in this checkpoint."
                            )
            with st.expander("ASCII Board"):
                st.code(snapshot.state_view.board_text)

    st.caption("Outgoing actions")
    _render_tree_inspector_outgoing_actions(
        st=st,
        snapshot=snapshot,
        state_key=state_key,
    )
    section_duration = time.perf_counter() - section_start_time
    print(
        f"{TREE_INSPECTOR_TIMING_PREFIX} render_tree_inspector_section "
        f"work_dir={paths.work_dir.name} selected_node_id={snapshot.selected_node_id!r} "
        f"total_s={section_duration:.6f}",
        flush=True,
    )


def render_tree_inspector_fragment(
    *,
    st: Any,
    paths: MorpionBootstrapPaths,
    latest_linoo_selection_table: Any,
    tree_node_classification_summary: Any,
) -> None:
    """Render the tree inspector in a fragment when supported by Streamlit."""
    fragment_renderer = (
        st.fragment(render_tree_inspector_section)
        if hasattr(st, "fragment")
        else render_tree_inspector_section
    )
    fragment_renderer(
        st=st,
        paths=paths,
        latest_linoo_selection_table=latest_linoo_selection_table,
        tree_node_classification_summary=tree_node_classification_summary,
    )


def _render_tree_inspector_navigation(
    *,
    st: Any,
    snapshot: Any,
    state_key: str,
) -> None:
    """Render root/parent/child/direct node navigation controls."""
    local_tree_view = snapshot.local_tree_view
    node_summary = snapshot.node_summary
    if local_tree_view is None or node_summary is None:
        return

    nav_columns = st.columns((1, 1, 2, 3))
    if nav_columns[0].button("Go to root", key=f"{state_key}::root"):
        st.session_state[state_key] = local_tree_view.root_node_id
        _tree_inspector_rerun(st)
    parent_disabled = not node_summary.parent_ids
    if nav_columns[1].button(
        "Go to parent",
        key=f"{state_key}::parent",
        disabled=parent_disabled,
    ):
        st.session_state[state_key] = node_summary.parent_ids[0]
        _tree_inspector_rerun(st)

    child_options = [summary.branch_label for summary in snapshot.child_summaries]
    selected_branch = nav_columns[2].selectbox(
        "Child branch",
        options=child_options if child_options else [""],
        key=f"{state_key}::child_branch",
        disabled=not child_options,
        label_visibility="collapsed",
    )
    if nav_columns[3].button(
        "Go to child",
        key=f"{state_key}::child",
        disabled=not child_options,
    ):
        selected_child_node_id = selected_child_node_id_for_branch(
            snapshot.child_summaries,
            selected_branch,
        )
        if selected_child_node_id is not None:
            st.session_state[state_key] = selected_child_node_id
            _tree_inspector_rerun(st)

    selected_node_input = st.text_input(
        "Node id",
        value=snapshot.selected_node_id,
        key=f"{state_key}::node_input",
        help="Jump directly to a checkpoint node id.",
    )
    if st.button("Go to node id", key=f"{state_key}::node_jump"):
        st.session_state[state_key] = selected_node_input.strip()
        _tree_inspector_rerun(st)


def selected_child_node_id_for_branch(
    child_summaries: tuple[MorpionBootstrapChildSummary, ...],
    branch_label: str,
) -> str | None:
    """Return the expanded child node id for the selected branch row."""
    for child_summary in child_summaries:
        if child_summary.branch_label == branch_label:
            return child_summary.child_node_id
    return None


def _tree_inspector_node_summary_dict(snapshot: Any) -> dict[str, object]:
    """Return one JSON-friendly selected-node summary for the dashboard."""
    node_summary = snapshot.node_summary
    if node_summary is None:
        return {}
    return {
        "node_id": node_summary.node_id,
        "depth": node_summary.depth,
        "parent_ids": list(node_summary.parent_ids),
        "child_ids": list(node_summary.child_ids),
        "visit_count": node_summary.visit_count,
        "is_terminal": node_summary.is_terminal,
        "is_exact": node_summary.is_exact,
        "direct_value_scalar": node_summary.direct_value_scalar,
        "backed_up_value_scalar": node_summary.backed_up_value_scalar,
        "best_child_id": node_summary.best_child_id,
        "best_branch_label": node_summary.best_branch_label,
    }


def _tree_inspector_local_tree_dict(snapshot: Any) -> dict[str, object]:
    """Return one JSON-friendly bounded tree neighborhood summary."""
    local_tree_view = snapshot.local_tree_view
    if local_tree_view is None:
        return {}
    return {
        "root_node_id": local_tree_view.root_node_id,
        "selected_node_id": local_tree_view.selected_node_id,
        "parent_node_ids": list(local_tree_view.parent_node_ids),
        "sibling_node_ids": list(local_tree_view.sibling_node_ids),
        "child_node_ids": list(local_tree_view.child_node_ids),
    }


def tree_inspector_child_rows(snapshot: Any) -> list[dict[str, object]]:
    """Return the child/action rows shown in the inspector table."""
    return [
        {
            "branch": child_summary.branch_label,
            "child_node_id": child_summary.child_node_id,
            "display_value": child_summary.display_value_scalar,
            "backed_up_value": child_summary.backed_up_value_scalar,
            "direct_value": child_summary.direct_value_scalar,
            "visit_count": child_summary.visit_count,
            "is_exact": _format_bool_icon(child_summary.is_exact),
            "is_terminal": _format_bool_icon(child_summary.is_terminal),
        }
        for child_summary in snapshot.child_summaries
    ]


def _render_tree_inspector_outgoing_actions(
    *,
    st: Any,
    snapshot: Any,
    state_key: str,
) -> None:
    """Render one compact outgoing-actions list with direct child navigation."""
    child_rows = tree_inspector_child_rows(snapshot)
    if not child_rows:
        st.caption("No outgoing actions available.")
        return

    row_columns = st.columns((4, 2, 2, 2, 2, 1, 1, 1))
    row_columns[0].caption("Branch")
    row_columns[1].caption("Child Node")
    row_columns[2].caption("Display")
    row_columns[3].caption("Direct")
    row_columns[4].caption("Backed-up")
    row_columns[5].caption("Exact")
    row_columns[6].caption("Terminal")
    row_columns[7].caption("Go")

    for index, child_row in enumerate(child_rows):
        child_node_id = child_row["child_node_id"]
        branch = child_row["branch"]
        row_columns = st.columns((4, 2, 2, 2, 2, 1, 1, 1))
        row_columns[0].write(_format_value(branch))
        row_columns[1].write(_format_value(child_node_id))
        row_columns[2].write(_format_value(child_row["display_value"]))
        row_columns[3].write(_format_value(child_row["direct_value"]))
        row_columns[4].write(_format_value(child_row["backed_up_value"]))
        row_columns[5].write(_format_value(child_row["is_exact"]))
        row_columns[6].write(_format_value(child_row["is_terminal"]))
        if row_columns[7].button(
            "Go",
            key=(
                f"{state_key}::child_row::{index}::"
                f"{_format_value(branch)}::{_format_value(child_node_id)}"
            ),
            disabled=child_node_id is None,
        ):
            st.session_state[state_key] = child_node_id
            _tree_inspector_rerun(st)

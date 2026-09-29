"""Dashboard tree-inspector Streamlit rendering."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from chipiron.environments.morpion.bootstrap.dashboard.formatting import (
    format_bool_icon as _format_bool_icon,
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
    from chipiron.environments.morpion.bootstrap.dashboard.theme import metric_row
    from chipiron.environments.morpion.bootstrap.dashboard.view_model import display

    section_start_time = time.perf_counter()
    state_key = f"morpion_bootstrap_selected_node::{paths.work_dir}"
    selected_node_id = st.session_state.get(state_key)
    with st.spinner("Reading checkpoint for inspection…"):
        snapshot = build_morpion_bootstrap_tree_inspector_snapshot(
            paths.work_dir, selected_node_id=selected_node_id
        )
    if snapshot.status_message:
        st.info(snapshot.status_message)
    if snapshot.error_message:
        st.warning(snapshot.error_message)
        return
    if snapshot.selected_node_id is None or snapshot.node_summary is None:
        if not snapshot.status_message:
            st.caption("No persisted runtime checkpoint available yet.")
        return
    if snapshot.selection_warning:
        st.warning(snapshot.selection_warning)
    st.session_state[state_key] = snapshot.selected_node_id
    _render_tree_inspector_navigation(st=st, snapshot=snapshot, state_key=state_key)
    node = snapshot.node_summary
    metric_row(
        st,
        [
            ("Node ID", node.node_id),
            ("Depth", node.depth),
            ("Visits", node.visit_count),
            ("Children", node.num_children),
        ],
    )
    left, right = st.columns((2, 1))
    with left, st.container(border=True):
        if snapshot.state_view is not None:
            state = snapshot.state_view
            board_click_event = render_clickable_morpion_board(
                svg=state.board_svg,
                click_targets=state.board_click_targets,
                click_radius=state.board_click_radius,
                height=520,
                render_size=state.board_render_size,
                key=f"{state_key}::board::{snapshot.selected_node_id}",
            )
            if board_click_event is not None:
                nonce = board_click_event.get("click_nonce")
                nonce_key = f"{state_key}::board_click_nonce"
                if nonce != st.session_state.get(nonce_key):
                    st.session_state[nonce_key] = nonce
                    action = board_click_event.get("action_name")
                    child = (
                        selected_child_node_id_for_branch(
                            snapshot.child_summaries, action
                        )
                        if isinstance(action, str)
                        else None
                    )
                    if child is not None:
                        st.session_state[state_key] = child
                        _tree_inspector_rerun(st)
                    else:
                        st.caption("This action is not expanded in the checkpoint.")
    with right:
        st.metric("Stored direct value", display(node.direct_value_scalar))
        st.metric("Stored backed-up value", display(node.backed_up_value_scalar))
        st.write("Exact:", display(node.is_exact))
        st.write("Terminal:", display(node.is_terminal))
        st.write("Best branch:", display(node.best_branch_label))
        st.caption(
            "Stored evaluation version: " + display(node.direct_evaluation_version)
        )
        st.caption(
            "Stored values are checkpoint observations, not fresh predictions from the current active evaluator."
        )
    st.markdown("**Outgoing actions**")
    _render_tree_inspector_outgoing_actions(
        st=st, snapshot=snapshot, state_key=state_key
    )
    with st.expander("Local tree neighborhood"):
        import graphviz

        graph = graphviz.Digraph(
            graph_attr={"rankdir": "TB", "bgcolor": "transparent"},
            node_attr={
                "shape": "box",
                "style": "rounded",
                "fontname": "Arial",
                "color": "#9fb4a8",
            },
        )
        graph.node(
            node.node_id,
            f"Selected · {node.node_id}",
            style="rounded,filled",
            fillcolor="#e2eee8",
        )
        for parent in node.parent_ids[:8]:
            graph.edge(parent, node.node_id)
        for child in node.child_ids[:16]:
            graph.edge(
                node.node_id, child, label="best" if child == node.best_child_id else ""
            )
        st.graphviz_chart(graph, width="stretch")
        st.caption(
            "Bounded to 8 parents and 16 children; full identifiers remain in advanced details."
        )
    with st.expander("Advanced node details"):
        st.write("Node Summary")
        st.json(_tree_inspector_node_summary_dict(snapshot))
        st.write("Local Tree")
        st.json(_tree_inspector_local_tree_dict(snapshot))
        if snapshot.state_view is not None:
            st.code(snapshot.state_view.board_text)
        st.caption(
            f"Checkpoint: {snapshot.checkpoint_path} · Reader: {time.perf_counter() - section_start_time:.2f}s"
        )
    with st.expander("Search / Linoo and tree classification"):
        render_linoo_selection_table(
            st=st, latest_linoo_selection_table=latest_linoo_selection_table
        )
        render_tree_node_classification_summary(
            st=st, summary=tree_node_classification_summary
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

    nav_columns = st.columns((1.3, 1.6, 3, 1.3))
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

    jump_input, jump_button = st.columns((4, 1.4))
    selected_node_input = jump_input.text_input(
        "Node id",
        value=snapshot.selected_node_id,
        key=f"{state_key}::node_input",
        help="Jump directly to a checkpoint node id.",
        label_visibility="collapsed",
    )
    if jump_button.button("Go to node id", key=f"{state_key}::node_jump"):
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
        "direct_evaluation_version": node_summary.direct_evaluation_version,
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
    rows = tree_inspector_child_rows(snapshot)
    if not rows:
        st.caption("No outgoing actions available.")
        return
    displayed = [
        {
            "Action": (
                "Best · "
                if row["branch"] == snapshot.node_summary.best_branch_label
                else ""
            )
            + str(row["branch"]),
            "Child": row["child_node_id"],
            "Visits": row["visit_count"],
            "Direct": row["direct_value"],
            "Backed-up": row["backed_up_value"],
            "Exact": row["is_exact"],
            "Terminal": row["is_terminal"],
        }
        for row in rows
    ]
    selected = st.dataframe(
        displayed,
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key=f"{state_key}::actions::{snapshot.selected_node_id}",
    )
    if selected.selection.rows:
        child = rows[selected.selection.rows[0]]["child_node_id"]
        if child is not None:
            st.session_state[state_key] = child
            _tree_inspector_rerun(st)

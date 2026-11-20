"""Explicit checkpoint navigation and saved-tree statistics."""

from typing import TYPE_CHECKING, Any

from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
    MorpionBootstrapPaths,
)
from chipiron.environments.morpion.bootstrap.dashboard.theme import (
    metric_row,
    page_header,
)
from chipiron.environments.morpion.bootstrap.dashboard.view_model import display

if TYPE_CHECKING:
    from chipiron.environments.morpion.bootstrap.dashboard.view_model import (
        BootstrapDashboardSnapshot,
    )


def _render_checkpoint(st: Any, paths: MorpionBootstrapPaths) -> None:
    """Recheck consent inside the fragment, including node-navigation reruns."""
    from chipiron.environments.morpion.bootstrap.dashboard.sections.tree_inspector import (
        render_tree_inspector_section,
    )
    from chipiron.environments.morpion.bootstrap.dashboard.tree_inspector import (
        checkpoint_inspection_identity,
        resolve_latest_runtime_checkpoint,
    )

    request_key = f"morpion_bootstrap_inspection::{paths.work_dir}"
    selection_key = f"morpion_bootstrap_selected_node::{paths.work_dir}"
    try:
        identity = checkpoint_inspection_identity(paths)
        if identity is None or st.session_state.get(request_key) != identity:
            if request_key in st.session_state:
                st.info(
                    "Checkpoint changed. Load the inspector again to inspect this generation."
                )
            st.session_state.pop(request_key, None)
            for key in list(st.session_state):
                if key == selection_key or key.startswith(selection_key + "::"):
                    del st.session_state[key]
        reference = resolve_latest_runtime_checkpoint(paths)
        if reference.status_message:
            st.info(reference.status_message)
        if identity is not None:
            st.caption(f"Runtime checkpoint: {identity[0]}")
        st.caption(
            "Full inspection is optional. The first inspection may read the entire "
            "checkpoint and build its node index in memory."
        )
        if st.button(
            "Load checkpoint inspector",
            disabled=identity is None,
            key=f"{request_key}::load",
        ):
            st.session_state[request_key] = identity
        if identity is None or st.session_state.get(request_key) != identity:
            return
        render_tree_inspector_section(
            st=st,
            paths=paths,
            latest_linoo_selection_table=None,
            tree_node_classification_summary=None,
            expected_checkpoint=identity,
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        st.session_state.pop(request_key, None)
        st.session_state.pop(selection_key, None)
        st.warning(f"Checkpoint temporarily unavailable: {exc}")


def _render_statistics(st: Any, paths: MorpionBootstrapPaths) -> None:
    """Keep only requested summaries in this session; never reload on invalidation."""
    from chipiron.environments.morpion.bootstrap.dashboard.history_view import (
        load_saved_tree_statistics,
        saved_tree_statistics_identity,
    )
    from chipiron.environments.morpion.bootstrap.dashboard.sections.linoo import (
        render_linoo_selection_table,
    )
    from chipiron.environments.morpion.bootstrap.dashboard.sections.tree_structure import (
        render_tree_node_classification_summary,
        render_tree_structure_section,
    )
    from chipiron.environments.morpion.bootstrap.dashboard.tree_inspector import (
        checkpoint_inspection_identity,
    )

    key = f"morpion_bootstrap_tree_statistics::{paths.work_dir}"
    try:
        identity = (
            checkpoint_inspection_identity(paths),
            saved_tree_statistics_identity(paths.work_dir),
        )
        saved = st.session_state.get(key)
        if saved is not None and saved[0] != identity:
            st.session_state.pop(key, None)
            saved = None
            st.info("Saved tree inputs changed. Request the statistics again.")
        st.caption(
            "Saved whole-tree statistics and Linoo details are loaded separately. "
            "Classification and depth summaries may require reading the full training-tree export."
        )
        if st.button("Load saved whole-tree statistics", key=f"{key}::load"):
            st.session_state.pop(key, None)
            with st.spinner("Reading saved tree statistics…"):
                data = load_saved_tree_statistics(paths.work_dir)
            if identity != (
                checkpoint_inspection_identity(paths),
                saved_tree_statistics_identity(paths.work_dir),
            ):
                message = (
                    "Saved tree inputs changed while loading; request statistics again."
                )
                st.warning(message)
                return
            saved = (identity, data)
            st.session_state[key] = saved
        if saved is None:
            return
        data = saved[1]
        if data.status_message:
            st.info(data.status_message)
        if data.error_message:
            st.warning(data.error_message)
        st.caption(f"Saved training-tree export: {display(data.snapshot_path)}")
        with st.expander("Whole-tree structure · saved summary"):
            render_tree_structure_section(
                st=st,
                tree_status=data.tree_status,
                depth_distribution=data.depth_distribution,
            )
            render_tree_node_classification_summary(st=st, summary=data.classification)
            render_linoo_selection_table(st=st, latest_linoo_selection_table=data.linoo)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        st.session_state.pop(key, None)
        st.warning(f"Saved tree statistics unavailable: {exc}")


def render(st: Any, snapshot: BootstrapDashboardSnapshot) -> None:
    """First paint uses small artifacts; both heavy views require explicit consent."""
    page_header(
        st,
        snapshot,
        "Tree inspector",
        "Navigate persisted states without changing the search.",
    )
    metric_row(
        st,
        [
            ("Generation", display(snapshot.generation)),
            ("Reported tree nodes", display(snapshot.tree_nodes)),
        ],
    )
    paths = MorpionBootstrapPaths.from_work_dir(snapshot.work_dir)
    renderer = (
        st.fragment(_render_checkpoint)
        if hasattr(st, "fragment")
        else _render_checkpoint
    )
    renderer(st=st, paths=paths)
    _render_statistics(st, paths)

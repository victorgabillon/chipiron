"""Explicit heavy-load consent, freshness and read-only Tree page regressions."""

import os
from contextlib import nullcontext
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
    MorpionBootstrapPaths,
)
from chipiron.environments.morpion.bootstrap.dashboard import (
    history_view,
    tree_inspector,
)
from chipiron.environments.morpion.bootstrap.dashboard.pages import tree
from chipiron.environments.morpion.bootstrap.dashboard.sections import (
    tree_inspector as section,
)
from tests.environments.test_morpion_bootstrap_operator_ui import _app
from tests.environments.test_morpion_bootstrap_tree_inspector import (
    _create_runtime_checkpoint,
)

if TYPE_CHECKING:
    from pathlib import Path


def forbidden(*args: Any, **kwargs: Any) -> Any:
    """Fail on an unwanted expensive dependency."""
    pytest.fail("Unrequested full artifact load")


class Controls:
    """Small controls for fragment-only reruns, without restarting the page."""

    def __init__(self) -> None:
        """Start with no explicit requests or saved observations."""
        self.session_state: dict[str, Any] = {}
        self.click: str | None = None
        self.messages: list[str] = []

    def button(self, label: str, **kwargs: Any) -> bool:
        """Consume one enabled click."""
        clicked = self.click == label and not kwargs.get("disabled", False)
        if clicked:
            self.click = None
        return clicked

    def caption(self, value: str) -> None:
        """Capture visible messages."""
        self.messages.append(value)

    markdown = caption
    info = caption
    warning = caption

    def spinner(self, label: str) -> Any:
        """Provide a presentation-only context."""
        return nullcontext()

    expander = spinner


def test_tree_first_paint_never_loads_heavy_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even an existing corrupt checkpoint is not decoded merely to open Tree."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.runtime_checkpoint_dir.mkdir(parents=True)
    paths.runtime_checkpoint_path_for_generation(1).write_text("invalid checkpoint")
    paths.tree_snapshot_dir.mkdir(parents=True)
    (paths.tree_snapshot_dir / "generation_000001.json").write_text("invalid export")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    for name in (
        "build_morpion_bootstrap_dashboard_data",
        "load_training_tree_snapshot",
        "load_morpion_sharded_training_tree_snapshot",
        "load_saved_tree_statistics",
        "summarize_tree_node_classification",
        "latest_tree_depth_distribution",
    ):
        monkeypatch.setattr(history_view, name, forbidden)
    monkeypatch.setattr(tree_inspector, "read_inspection_checkpoint", forbidden)
    monkeypatch.setattr(tree_inspector, "_index_checkpoint_payload", forbidden)
    app = _app(tmp_path).run()
    app.radio(key="bootstrap_operator_page").set_value("Tree").run()
    assert not app.exception, [e.message for e in app.exception]
    assert {"Load checkpoint inspector", "Load saved whole-tree statistics"} <= {
        b.label for b in app.button
    }
    app.run()
    assert not app.exception
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("change", ["generation", "mtime", "missing", "workspace"])
def test_fragment_change_revokes_consent_and_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    """The fragment guard runs even when the surrounding page does not rerun."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.runtime_checkpoint_dir.mkdir(parents=True)
    checkpoint = paths.runtime_checkpoint_path_for_generation(1)
    checkpoint.write_text("not decoded in this control test")
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        section, "render_tree_inspector_section", lambda **kw: calls.append(kw)
    )
    st = Controls()
    tree._render_checkpoint(st, paths)
    assert not calls
    st.click = "Load checkpoint inspector"
    tree._render_checkpoint(st, paths)
    assert len(calls) == 1 and calls[0]["expected_checkpoint"] is not None
    selection = f"morpion_bootstrap_selected_node::{paths.work_dir}"
    st.session_state[selection] = "42"
    st.session_state[selection + "::child_branch"] = "old child"
    tree._render_checkpoint(st, paths)
    assert len(calls) == 2 and st.session_state[selection] == "42"
    if change == "generation":
        paths.runtime_checkpoint_path_for_generation(2).write_text("new generation")
    elif change == "mtime":
        old = checkpoint.stat().st_mtime_ns
        os.utime(checkpoint, ns=(old + 1_000_000, old + 1_000_000))
    elif change == "missing":
        checkpoint.unlink()
    else:
        paths = MorpionBootstrapPaths.from_work_dir(tmp_path / "other")
    tree._render_checkpoint(st, paths)
    assert len(calls) == 2
    if change != "workspace":
        assert not any(k.startswith(selection) for k in st.session_state)
    if change not in ("missing", "workspace"):
        st.click = "Load checkpoint inspector"
        tree._render_checkpoint(st, paths)
        assert (
            len(calls) == 3
            and calls[-1]["expected_checkpoint"] != calls[0]["expected_checkpoint"]
        )


def test_statistics_explicit_cached_and_invalidated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An expander/rerun never loads statistics, and stats never load the inspector."""
    from chipiron.environments.morpion.bootstrap.dashboard.sections import (
        linoo,
        tree_structure,
    )

    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(section, "render_tree_inspector_section", forbidden)
    monkeypatch.setattr(
        history_view,
        "load_saved_tree_statistics",
        lambda work: (
            calls.append(str(work))
            or SimpleNamespace(
                snapshot_path=None,
                status_message=None,
                tree_status=None,
                depth_distribution=(),
                classification=None,
                linoo=None,
                error_message=None,
            )
        ),
    )
    monkeypatch.setattr(
        tree_structure, "render_tree_structure_section", lambda **kw: None
    )
    monkeypatch.setattr(
        tree_structure, "render_tree_node_classification_summary", lambda **kw: None
    )
    monkeypatch.setattr(linoo, "render_linoo_selection_table", lambda **kw: None)
    st = Controls()
    tree._render_statistics(st, paths)
    assert not calls
    st.click = "Load saved whole-tree statistics"
    tree._render_statistics(st, paths)
    tree._render_statistics(st, paths)
    assert len(calls) == 1
    paths.latest_linoo_selection_table_path.parent.mkdir(parents=True, exist_ok=True)
    paths.latest_linoo_selection_table_path.write_text("{}")
    tree._render_statistics(st, paths)
    assert len(calls) == 1
    assert (
        f"morpion_bootstrap_tree_statistics::{paths.work_dir}" not in st.session_state
    )
    st.click = "Load saved whole-tree statistics"
    tree._render_statistics(st, paths)
    assert len(calls) == 2


def test_guard_blocks_change_between_consent_and_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-resolving the pointer cannot silently load another generation."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    _create_runtime_checkpoint(tmp_path)
    identity = tree_inspector.checkpoint_inspection_identity(paths)
    paths.runtime_checkpoint_path_for_generation(2).write_text("never read")
    monkeypatch.setattr(tree_inspector, "read_inspection_checkpoint", forbidden)
    with pytest.raises(ValueError, match="Checkpoint changed"):
        tree_inspector.build_morpion_bootstrap_tree_inspector_snapshot(
            tmp_path, expected_checkpoint=identity
        )


def test_explicit_navigation_reuses_real_index_and_preserves_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guarded root/child/parent inspection preserves boards, values and artifacts."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    checkpoint = _create_runtime_checkpoint(tmp_path)
    before = checkpoint.read_bytes()
    identity = tree_inspector.checkpoint_inspection_identity(paths)
    root = tree_inspector.build_morpion_bootstrap_tree_inspector_snapshot(
        tmp_path, expected_checkpoint=identity
    )
    child = next(c for c in root.child_summaries if c.child_node_id is not None)
    monkeypatch.setattr(tree_inspector, "read_inspection_checkpoint", forbidden)
    selected = tree_inspector.build_morpion_bootstrap_tree_inspector_snapshot(
        tmp_path, selected_node_id=child.child_node_id, expected_checkpoint=identity
    )
    assert selected.node_summary is not None and selected.state_view is not None
    assert root.selected_node_id in selected.node_summary.parent_ids
    assert "<svg" in selected.state_view.board_svg
    assert selected.node_summary.direct_value_scalar == child.direct_value_scalar
    assert selected.node_summary.backed_up_value_scalar == child.backed_up_value_scalar
    assert selected.node_summary.is_terminal == child.is_terminal
    assert selected.node_summary.is_exact == child.is_exact
    parent = tree_inspector.build_morpion_bootstrap_tree_inspector_snapshot(
        tmp_path, selected_node_id=root.selected_node_id, expected_checkpoint=identity
    )
    assert parent == root
    assert checkpoint.read_bytes() == before


def test_corrupt_inspection_reports_error_and_does_not_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit decoding errors revoke consent; a later rerun stays cheap."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.runtime_checkpoint_dir.mkdir(parents=True)
    paths.runtime_checkpoint_path_for_generation(1).write_text("not JSON")
    st = Controls()
    st.click = "Load checkpoint inspector"
    tree._render_checkpoint(st, paths)
    assert f"morpion_bootstrap_inspection::{paths.work_dir}" not in st.session_state
    assert any("checkpoint" in m.lower() for m in st.messages)
    monkeypatch.setattr(tree_inspector, "read_inspection_checkpoint", forbidden)
    tree._render_checkpoint(st, paths)


def test_narrow_statistics_reuses_summaries_without_history_or_disk_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A requested export is decoded once; no unrelated aggregation is reachable."""
    from anemone.training_export import (
        TrainingTreeSnapshot,
        save_training_tree_snapshot,
    )

    from tests.environments.morpion_training_snapshot_helpers import (
        make_training_node_snapshot,
    )

    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    export = paths.tree_snapshot_dir / "generation_000001.json"
    export.parent.mkdir(parents=True)
    snapshot = TrainingTreeSnapshot(
        root_node_id="0", nodes=(make_training_node_snapshot(node_id="0", depth=0),)
    )
    save_training_tree_snapshot(snapshot, export)
    for name in (
        "load_bootstrap_history",
        "load_morpion_bootstrap_run_view",
        "build_disk_usage_summary",
        "build_morpion_bootstrap_dashboard_data",
    ):
        monkeypatch.setattr(history_view, name, forbidden)
    result = history_view.load_saved_tree_statistics(tmp_path)
    assert result.classification == history_view.summarize_tree_node_classification(
        snapshot
    )
    assert result.tree_status is not None and result.tree_status.num_nodes == 1
    assert result.depth_distribution[0].num_nodes == 1
    assert result.snapshot_path == export


def test_change_during_load_discards_statistics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Do not seal a session summary if an input changes while the reader runs."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    st = Controls()
    st.click = "Load saved whole-tree statistics"

    def changing(work: Path) -> Any:
        """Simulate publication while the explicit summary is loading."""
        paths.latest_linoo_selection_table_path.parent.mkdir(
            parents=True, exist_ok=True
        )
        paths.latest_linoo_selection_table_path.write_text("{}")
        return None

    monkeypatch.setattr(history_view, "load_saved_tree_statistics", changing)
    tree._render_statistics(st, paths)
    assert (
        f"morpion_bootstrap_tree_statistics::{paths.work_dir}" not in st.session_state
    )
    assert any("changed while loading" in m for m in st.messages)
    monkeypatch.setattr(history_view, "load_saved_tree_statistics", forbidden)
    tree._render_statistics(st, paths)


def test_missing_and_corrupt_statistics_remain_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No invented summary is returned for a corrupt export."""
    missing = history_view.load_saved_tree_statistics(tmp_path)
    assert missing.snapshot_path is None and missing.classification is None
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.tree_snapshot_dir.mkdir(parents=True)
    (paths.tree_snapshot_dir / "generation_000001.json").write_text("invalid JSON")
    st = Controls()
    st.click = "Load saved whole-tree statistics"
    tree._render_statistics(st, paths)
    assert any("statistics unavailable" in m for m in st.messages)
    saved = st.session_state[f"morpion_bootstrap_tree_statistics::{paths.work_dir}"]
    assert saved[1].classification is None
    assert saved[1].error_message and "statistics unavailable" in saved[1].error_message
    monkeypatch.setattr(history_view, "load_saved_tree_statistics", forbidden)
    tree._render_statistics(st, paths)
    assert (
        st.session_state[f"morpion_bootstrap_tree_statistics::{paths.work_dir}"]
        is saved
    )


def test_stale_run_state_reference_is_visible(tmp_path: Path) -> None:
    """A missing dedicated pointer is explained before any fallback inspection."""
    from chipiron.environments.morpion.bootstrap.run_state import (
        MorpionBootstrapRunState,
        save_bootstrap_run_state,
    )

    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    save_bootstrap_run_state(
        MorpionBootstrapRunState(
            generation=1,
            cycle_index=0,
            latest_tree_snapshot_path=None,
            latest_rows_path=None,
            latest_model_bundle_paths=None,
            active_evaluator_name=None,
            tree_size_at_last_save=1,
            last_save_unix_s=None,
            latest_runtime_checkpoint_path="missing.json",
        ),
        paths.run_state_path,
    )
    st = Controls()
    tree._render_checkpoint(st, paths)
    assert any("missing runtime checkpoint" in m for m in st.messages)
    assert f"morpion_bootstrap_inspection::{paths.work_dir}" not in st.session_state


def test_native_tree_inspection_action_preserves_navigation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real Streamlit button reaches the existing board and navigation controls."""
    _create_runtime_checkpoint(tmp_path)
    monkeypatch.setattr(history_view, "load_saved_tree_statistics", forbidden)
    monkeypatch.setattr(section, "render_clickable_morpion_board", lambda **kw: None)
    app = _app(tmp_path).run()
    app.radio(key="bootstrap_operator_page").set_value("Tree").run()
    next(b for b in app.button if b.label == "Load checkpoint inspector").click().run()
    assert not app.exception, [e.message for e in app.exception]
    assert {"Go to root", "Go to parent"} <= {b.label for b in app.button}
    assert any(m.label == "Stored direct value" for m in app.metric)
    monkeypatch.setattr(tree_inspector, "read_inspection_checkpoint", forbidden)
    next(b for b in app.button if b.label == "Go to root").click().run()
    assert not app.exception, [e.message for e in app.exception]


def test_unavailable_export_preserves_linoo_without_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unavailable classification must not hide independent saved selection evidence."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.tree_snapshot_dir.mkdir(parents=True)
    (paths.tree_snapshot_dir / "generation_000001.json").write_text("unsupported")
    calls: list[object] = []
    linoo = object()
    monkeypatch.setattr(
        history_view,
        "_load_resolved_training_tree_snapshot",
        lambda ref: calls.append(ref),
    )
    monkeypatch.setattr(
        history_view,
        "load_latest_linoo_selection_table_for_dashboard",
        lambda work: linoo,
    )
    data = history_view.load_saved_tree_statistics(tmp_path)
    assert len(calls) == 1
    assert data.linoo is linoo and data.classification is None
    assert data.error_message and "unsupported" in data.error_message


@pytest.mark.parametrize("mode", ["fragment", "full", "legacy", "unexpected"])
def test_navigation_rerun_scope_fallback_is_narrow(mode: str) -> None:
    """Keep fragment reruns when valid and propagate unrelated execution errors."""
    from streamlit.errors import StreamlitInvalidLayoutContextError

    calls: list[dict[str, str]] = []

    def rerun(**kwargs: str) -> None:
        """Model Streamlit's scope checks without starting another script run."""
        calls.append(kwargs)
        if not kwargs or mode == "fragment":
            return
        message = "scope unavailable during this run"
        if mode == "full":
            raise StreamlitInvalidLayoutContextError(message)
        if mode == "legacy":
            raise TypeError(message)
        raise RuntimeError(message)

    st = SimpleNamespace(rerun=rerun)
    if mode == "unexpected":
        with pytest.raises(RuntimeError, match="scope unavailable"):
            section._tree_inspector_rerun(st)
    else:
        section._tree_inspector_rerun(st)
    expected = [{"scope": "fragment"}]
    if mode in ("full", "legacy"):
        expected.append({})
    assert calls == expected

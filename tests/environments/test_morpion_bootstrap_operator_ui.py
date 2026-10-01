"""High-value native Streamlit integration checks for navigation and operator safety."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

    from _pytest.monkeypatch import MonkeyPatch
    from streamlit.testing.v1 import AppTest


def _app(work_dir: Path) -> AppTest:
    """Run the real page composition with an isolated artifact directory."""
    from streamlit.testing.v1 import AppTest

    return AppTest.from_string(
        "from pathlib import Path\n"
        "from chipiron.environments.morpion.bootstrap.dashboard.app import run_dashboard_app\n"
        f"run_dashboard_app(Path({str(work_dir)!r}))\n",
        default_timeout=30,
    )


def test_empty_pages_navigation_and_pipeline_controls(tmp_path: Path) -> None:
    """All five views handle empty artifacts and pipeline mode hides unsafe Start."""
    from chipiron.environments.morpion.bootstrap.bootstrap_args import (
        MorpionBootstrapArgs,
    )
    from chipiron.environments.morpion.bootstrap.config import (
        bootstrap_config_from_args,
        save_bootstrap_config,
    )

    config = bootstrap_config_from_args(
        MorpionBootstrapArgs(work_dir=tmp_path, pipeline_mode="artifact_pipeline")
    )
    save_bootstrap_config(config, tmp_path / "bootstrap_config.json")
    app = _app(tmp_path).run()
    assert not app.exception
    assert app.radio(key="bootstrap_operator_page").value == "Overview"
    assert app.selectbox(key="operator_refresh_Overview").value == "15s"
    assert not app.json
    for page in ("Record", "Tree", "Evaluator", "Operations"):
        app.radio(key="bootstrap_operator_page").set_value(page).run()
        assert not app.exception, [e.message for e in app.exception]
        assert app.selectbox(key=f"operator_refresh_{page}").value == "Off"
    assert not {"Start", "Stop", "Restart"}.intersection(
        button.label for button in app.button
    )
    assert not (tmp_path / "control.json").exists()
    app.radio(key="bootstrap_operator_page").set_value("Overview").run()
    assert not app.exception and not app.json


def test_overview_never_deserializes_checkpoint(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """A periodic overview refresh must not enter expensive tree or record readers."""
    from chipiron.environments.morpion.bootstrap.dashboard import (
        history_view,
        tree_inspector,
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("Overview attempted an expensive artifact load")

    monkeypatch.setattr(
        history_view, "build_morpion_bootstrap_dashboard_data", forbidden
    )
    monkeypatch.setattr(
        history_view, "build_current_certified_record_board_view", forbidden
    )
    monkeypatch.setattr(
        tree_inspector, "build_morpion_bootstrap_tree_inspector_snapshot", forbidden
    )
    (tmp_path / "bootstrap_config.json").write_text("{}")
    app = _app(tmp_path).run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception


def test_tree_first_paint_skips_saved_whole_tree_scan(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Opening Tree should not deserialize the training tree until explicitly requested."""
    from chipiron.environments.morpion.bootstrap.dashboard import data_cache

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("Tree first paint attempted a saved whole-tree scan")

    monkeypatch.setattr(
        data_cache,
        "cached_build_morpion_bootstrap_tree_structure_data",
        forbidden,
    )
    (tmp_path / "bootstrap_config.json").write_text("{}")
    app = _app(tmp_path).run()
    app.radio(key="bootstrap_operator_page").set_value("Tree").run()

    assert not app.exception, [e.message for e in app.exception]
    assert any(
        checkbox.label == "Load saved whole-tree statistics"
        for checkbox in app.checkbox
    )


def test_tree_checkpoint_index_build_is_explicit(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """A new large-checkpoint path should show a build button before deserialization."""
    from chipiron.environments.morpion.bootstrap import (
        AnemoneMorpionSearchRunner,
        MorpionBootstrapPaths,
    )
    from chipiron.environments.morpion.bootstrap.control import (
        MorpionBootstrapEffectiveRuntimeConfig,
    )

    cache_dir = tmp_path / "dashboard-cache"
    monkeypatch.setenv("CHIPIRON_DASHBOARD_CACHE_DIR", str(cache_dir))
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    runner = AnemoneMorpionSearchRunner()
    runner.load_or_create(
        None,
        None,
        MorpionBootstrapEffectiveRuntimeConfig(tree_branch_limit=2),
    )
    runner.grow(1)
    runner.save_checkpoint(paths.runtime_checkpoint_path_for_generation(1))
    (tmp_path / "bootstrap_config.json").write_text("{}")

    app = _app(tmp_path).run()
    app.radio(key="bootstrap_operator_page").set_value("Tree").run()

    assert not app.exception, [e.message for e in app.exception]
    assert any(button.label == "Build tree inspector index" for button in app.button)
    assert not list(cache_dir.rglob("*.sqlite3"))


def test_pending_edit_preview_precedes_apply(tmp_path: Path) -> None:
    """Operator edits preview a persistent request; no mutation occurs until apply."""
    from chipiron.environments.morpion.bootstrap.bootstrap_args import (
        MorpionBootstrapArgs,
    )
    from chipiron.environments.morpion.bootstrap.config import (
        bootstrap_config_from_args,
        save_bootstrap_config,
    )

    save_bootstrap_config(
        bootstrap_config_from_args(
            MorpionBootstrapArgs(work_dir=tmp_path, pipeline_mode="artifact_pipeline")
        ),
        tmp_path / "bootstrap_config.json",
    )
    app = _app(tmp_path).run()
    app.radio(key="bootstrap_operator_page").set_value("Operations").run()
    assert not app.exception
    next(
        w for w in app.checkbox if w.label == "Persist explicit override for max rows"
    ).check().run()
    next(w for w in app.number_input if w.label == "Max rows").set_value(42).run()
    assert not (tmp_path / "control.json").exists()
    assert any("max_rows" in str(frame.value) for frame in app.dataframe)
    next(w for w in app.button if w.label == "Apply changes").click().run()
    assert not app.exception
    assert json.loads((tmp_path / "control.json").read_text())["max_rows"] == 42

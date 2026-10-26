"""Repository test configuration must refuse real default persistence destinations."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from anemone.training_export import TrainingTreeSnapshot

from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
    MorpionBootstrapPaths,
)
from chipiron.environments.morpion.bootstrap.record_status import (
    persist_certified_leaderboard_candidates,
)
from chipiron.utils.path_runtime import assert_test_output_path, get_output_root
from chipiron.utils.path_variables import get_mlflow_db_path, get_mlflow_uri

if TYPE_CHECKING:
    from pytest import MonkeyPatch

pytestmark = pytest.mark.fast


def test_protected_workspace_refuses_all_derived_destinations() -> None:
    """Checkpoint, science, rows, models, status and controls share the guarded root."""
    with pytest.raises(ValueError, match="Protected persistence"):
        MorpionBootstrapPaths.from_work_dir(Path.home() / "morpion_runs" / "test")


def test_protected_leaderboard_is_refused_before_read_or_write() -> None:
    """Even explicitly selecting the historical leaderboard is unsafe in tests."""
    snapshot = TrainingTreeSnapshot(root_node_id="unused", nodes=())
    with pytest.raises(ValueError, match="Protected persistence"):
        persist_certified_leaderboard_candidates(
            snapshot=snapshot,
            generation=1,
            run_work_dir="unused",
            cycle_index=1,
            timestamp_utc="2026-10-26T00:00:00Z",
            leaderboard_path=Path.home() / "morpion_runs" / "morpion_leaderboard.jsonl",
        )


def test_unsafe_output_and_database_overrides_are_refused(
    monkeypatch: MonkeyPatch,
) -> None:
    """Inherited application and SQLite overrides cannot select research storage."""
    protected = Path.home() / "morpion_runs"
    monkeypatch.setenv("CHIPIRON_OUTPUT_DIR", str(protected))
    monkeypatch.setenv("ML_FLOW_DB_PATH", str(protected / "mlruns.db"))
    with pytest.raises(ValueError, match="Protected persistence"):
        get_output_root()
    with pytest.raises(ValueError, match="Protected persistence"):
        get_mlflow_db_path()
    with pytest.raises(ValueError, match="Protected persistence"):
        assert_test_output_path(protected / "reader.sqlite3")
    monkeypatch.setenv("ML_FLOW_URI_PATH", f"sqlite:///{protected}/mlruns.db")
    with pytest.raises(ValueError, match="Protected persistence"):
        get_mlflow_uri()


def test_private_defaults_and_explicit_production_workspace(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Tests own default outputs; production retains explicit workspace semantics."""
    root = Path(os.environ["CHIPIRON_TEST_OUTPUT_ROOT"])
    assert get_output_root() == root
    assert get_mlflow_db_path().is_relative_to(root)
    assert Path(os.environ["MORPION_LEADERBOARD_PATH"]).is_relative_to(root)
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    assert paths.model_dir.is_relative_to(tmp_path)
    monkeypatch.delenv("CHIPIRON_TEST_OUTPUT_ROOT")
    protected = Path.home() / "morpion_runs" / "explicit-production"
    assert (
        MorpionBootstrapPaths.from_work_dir(protected).work_dir == protected.resolve()
    )


def test_reader_cache_refuses_protected_override(tmp_path: Path) -> None:
    """The derived SQLite writer cannot be redirected into a historical workspace."""
    from chipiron.environments.morpion.bootstrap.sharded_training_reader import (
        ShardedTrainingGenerationView,
    )
    from tests.environments.test_morpion_sharded_training_reader import write_generation

    source = write_generation(tmp_path / "source", ["a", "b"])
    with pytest.raises(ValueError, match="Protected persistence"):
        ShardedTrainingGenerationView(
            source, cache_dir=Path.home() / "morpion_runs" / "unsafe-cache"
        )

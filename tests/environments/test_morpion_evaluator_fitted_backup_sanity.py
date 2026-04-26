"""Tests for fixed-tree Morpion fitted-backup sanity checks."""
# ruff: noqa: E402

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, cast

import torch

if TYPE_CHECKING:
    from pytest import MonkeyPatch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHIPIRON_PACKAGE_ROOT = _REPO_ROOT / "src" / "chipiron"
_ATOMHEART_PACKAGE_ROOT = _REPO_ROOT.parent / "atomheart" / "src" / "atomheart"
_ANEMONE_PACKAGE_ROOT = _REPO_ROOT.parent / "anemone" / "src" / "anemone"
_MORPION_EVALUATORS_PACKAGE_ROOT = (
    _REPO_ROOT
    / "src"
    / "chipiron"
    / "environments"
    / "morpion"
    / "players"
    / "evaluators"
)

if "chipiron" not in sys.modules:
    _chipiron_stub = ModuleType("chipiron")
    _chipiron_stub.__path__ = [str(_CHIPIRON_PACKAGE_ROOT)]
    sys.modules["chipiron"] = _chipiron_stub

if "chipiron.environments.morpion.players.evaluators" not in sys.modules:
    _evaluators_stub = ModuleType("chipiron.environments.morpion.players.evaluators")
    _evaluators_stub.__path__ = [str(_MORPION_EVALUATORS_PACKAGE_ROOT)]
    sys.modules["chipiron.environments.morpion.players.evaluators"] = _evaluators_stub

if "atomheart" not in sys.modules:
    _atomheart_stub = ModuleType("atomheart")
    _atomheart_stub.__path__ = [str(_ATOMHEART_PACKAGE_ROOT)]
    sys.modules["atomheart"] = _atomheart_stub

if "anemone" not in sys.modules:
    _anemone_stub = ModuleType("anemone")
    _anemone_stub.__path__ = [str(_ANEMONE_PACKAGE_ROOT)]
    sys.modules["anemone"] = _anemone_stub

from anemone.training_export import (
    TrainingNodeSnapshot,
    TrainingTreeSnapshot,
    save_training_tree_snapshot,
)

import chipiron.environments.morpion.bootstrap.evaluator_fitted_backup_sanity as fitted_module
from chipiron.environments.morpion.bootstrap.bootstrap_loop import (
    MorpionBootstrapPaths,
    MorpionEvaluatorSpec,
)
from chipiron.environments.morpion.bootstrap.evaluator_diagnostics import (
    MorpionEvaluatorTrainingDiagnostics,
)
from chipiron.environments.morpion.bootstrap.evaluator_fitted_backup_sanity import (
    MorpionFittedBackupSanityArgs,
    fitted_backup_node_values,
    run_fitted_backup_sanity,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.model import (
    MorpionRegressor,
    MorpionRegressorArgs,
)


def _node(
    node_id: str,
    *,
    parent_ids: tuple[str, ...] = (),
    child_ids: tuple[str, ...] = (),
    depth: int = 0,
    direct_value_scalar: float | None = 0.0,
    backed_up_value_scalar: float | None = None,
    is_exact: bool = False,
    is_terminal: bool = False,
    state_ref_payload: dict[str, object] | None = None,
) -> TrainingNodeSnapshot:
    """Build one synthetic training node."""
    return TrainingNodeSnapshot(
        node_id=node_id,
        parent_ids=parent_ids,
        child_ids=child_ids,
        depth=depth,
        state_ref_payload=state_ref_payload,
        direct_value_scalar=direct_value_scalar,
        backed_up_value_scalar=backed_up_value_scalar,
        is_terminal=is_terminal,
        is_exact=is_exact,
        over_event_label=None,
        visit_count=depth + 1,
        metadata={"source": "fitted-backup-test"},
    )


def _backup_snapshot() -> TrainingTreeSnapshot:
    """Build a tiny frozen tree for fitted-backup tests."""
    root = _node("root", child_ids=("left", "right"), depth=0, direct_value_scalar=0.1)
    left = _node(
        "left",
        parent_ids=("root",),
        child_ids=("left_leaf",),
        depth=1,
        direct_value_scalar=0.2,
    )
    right = _node(
        "right",
        parent_ids=("root",),
        depth=1,
        direct_value_scalar=3.0,
        is_exact=True,
        backed_up_value_scalar=5.0,
    )
    left_leaf = _node(
        "left_leaf",
        parent_ids=("left",),
        depth=2,
        direct_value_scalar=7.0,
    )
    return TrainingTreeSnapshot(
        root_node_id="root",
        nodes=(root, left, right, left_leaf),
        metadata={"format_kind": "training_tree_snapshot", "format_version": 1},
    )


def _spec() -> MorpionEvaluatorSpec:
    """Return a minimal evaluator spec for tests."""
    return MorpionEvaluatorSpec(
        name="linear_5",
        model_type="linear",
        hidden_sizes=None,
        num_epochs=1,
        batch_size=1,
        learning_rate=1e-3,
        feature_subset_name="handcrafted_5_core",
    )


def _model() -> MorpionRegressor:
    """Return an unused model object for direct-value fallback tests."""
    model = MorpionRegressor(MorpionRegressorArgs(model_kind="linear"))
    model.eval()
    return model


def test_exact_terminal_values_stay_fixed_across_backup_iterations() -> None:
    """Exact/terminal nodes should keep their ground-truth values."""
    values = fitted_backup_node_values(
        snapshot=_backup_snapshot(),
        model=_model(),
        spec=_spec(),
    )

    assert values["right"].backed_up_target == 5.0
    assert values["right"].direct_value_before_backup == 5.0
    assert values["right"].target_source == "ground_truth_exact_or_terminal"


def test_non_terminal_parent_receives_max_child_backed_up_value() -> None:
    """Parent backups should be max over child backed-up targets."""
    values = fitted_backup_node_values(
        snapshot=_backup_snapshot(),
        model=_model(),
        spec=_spec(),
    )

    assert values["left"].backed_up_target == 7.0
    assert values["left"].target_source == "child_backup"
    assert values["root"].backed_up_target == 7.0
    assert values["root"].target_source == "child_backup"


def test_leaf_non_exact_node_uses_direct_or_evaluator_prediction() -> None:
    """Non-exact leaves should use their pre-backup fitted value."""
    values = fitted_backup_node_values(
        snapshot=_backup_snapshot(),
        model=_model(),
        spec=_spec(),
    )

    assert values["left_leaf"].backed_up_target == 7.0
    assert values["left_leaf"].target_source == "frontier_prediction"


def test_target_change_metrics_are_computed_between_iterations() -> None:
    """Previous targets should produce absolute target-change fields."""
    values = fitted_backup_node_values(
        snapshot=_backup_snapshot(),
        model=_model(),
        spec=_spec(),
        previous_targets={"root": 4.0, "left": 7.0, "right": 3.0, "left_leaf": 1.0},
    )

    assert values["root"].abs_target_change == 3.0
    assert values["left"].abs_target_change == 0.0
    assert values["right"].abs_target_change == 2.0
    assert values["left_leaf"].abs_target_change == 6.0


def test_two_tiny_iterations_write_summary_and_artifacts(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """A monkeypatched two-iteration run should write the expected artifacts."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    snapshot = TrainingTreeSnapshot(
        root_node_id="root",
        nodes=(
            _node(
                "root",
                child_ids=("leaf",),
                state_ref_payload={"fake": "payload"},
            ),
            _node(
                "leaf",
                parent_ids=("root",),
                depth=1,
                direct_value_scalar=2.0,
                state_ref_payload={"fake": "payload"},
            ),
        ),
        metadata={"format_kind": "training_tree_snapshot", "format_version": 1},
    )
    original_nodes = snapshot.nodes
    save_training_tree_snapshot(snapshot, paths.tree_snapshot_path_for_generation(1))
    model = _model()

    monkeypatch.setattr(fitted_module, "_initial_model", lambda **_kwargs: model)
    feature_cache_builds: list[int] = []

    def _fake_feature_cache(snapshot: TrainingTreeSnapshot, **_kwargs: object) -> object:
        feature_cache_builds.append(len(snapshot.nodes))
        return fitted_module.SnapshotFeatureCache(
            node_ids=(),
            input_tensor=torch.empty((0, 5)),
        )

    monkeypatch.setattr(
        fitted_module,
        "build_snapshot_feature_cache",
        _fake_feature_cache,
    )
    monkeypatch.setattr(fitted_module, "_predict_rows", lambda *_args, **_kwargs: [0.0, 0.0])

    def _fake_train(_args: object) -> tuple[MorpionRegressor, dict[str, float]]:
        return model, {"final_loss": 0.25, "num_epochs": 1.0, "num_samples": 2.0}

    monkeypatch.setattr(fitted_module, "train_morpion_regressor", _fake_train)

    def _fake_diagnostics(**kwargs: object) -> MorpionEvaluatorTrainingDiagnostics:
        return MorpionEvaluatorTrainingDiagnostics(
            generation=int(kwargs["generation"]),
            evaluator_name=str(kwargs["evaluator_name"]),
            dataset_size=2,
            created_at=str(kwargs["created_at"]),
            representative_examples=[],
            worst_examples=[],
            mae_before=None,
            mae_after=0.5,
            max_abs_error_before=None,
            max_abs_error_after=1.0,
        )

    monkeypatch.setattr(
        fitted_module,
        "build_evaluator_training_diagnostics",
        _fake_diagnostics,
    )
    monkeypatch.setattr(
        fitted_module,
        "build_backup_target_diagnostics",
        lambda **_kwargs: {"dataset_size": 2},
    )

    summary = run_fitted_backup_sanity(
        MorpionFittedBackupSanityArgs(
            work_dir=tmp_path,
            generation=1,
            evaluator_name="linear_5",
            num_iterations=2,
            num_epochs=1,
            batch_size=2,
            run_name="test_run",
            max_backup_nodes=1,
        )
    )

    run_dir = tmp_path / "evaluator_fitted_backup_sanity" / "test_run"
    assert (run_dir / "summary.json").is_file()
    assert (run_dir / "iteration_000" / "rows.json").is_file()
    assert (run_dir / "iteration_001" / "rows.json").is_file()
    assert (run_dir / "iteration_000" / "target_diagnostics.json").is_file()
    assert (
        run_dir / "iteration_001" / "diagnostics" / "linear_5.json"
    ).is_file()

    summary_data = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["num_iterations"] == 2
    assert summary["backup_nodes"] == 1
    assert summary["max_backup_nodes"] == 1
    assert feature_cache_builds == [1]
    assert len(summary_data["iterations"]) == 2
    assert summary_data["iterations"][1]["mean_abs_target_change"] == 0.0
    assert snapshot.nodes is original_nodes
    assert cast("TrainingNodeSnapshot", snapshot.nodes[0]).backed_up_value_scalar is None

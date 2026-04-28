"""Tests for fixed-tree Morpion fitted-backup sanity checks."""
# ruff: noqa: E402

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, cast

import torch

if TYPE_CHECKING:
    from pytest import MonkeyPatch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHIPIRON_PACKAGE_ROOT = _REPO_ROOT / "src" / "chipiron"
_ENVIRONMENTS_PACKAGE_ROOT = _CHIPIRON_PACKAGE_ROOT / "environments"
_MORPION_PACKAGE_ROOT = _ENVIRONMENTS_PACKAGE_ROOT / "morpion"
_BOOTSTRAP_PACKAGE_ROOT = _MORPION_PACKAGE_ROOT / "bootstrap"
_MORPION_PLAYERS_PACKAGE_ROOT = _MORPION_PACKAGE_ROOT / "players"
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
_MORPION_NN_PACKAGE_ROOT = _MORPION_EVALUATORS_PACKAGE_ROOT / "neural_networks"

if "chipiron" not in sys.modules:
    _chipiron_stub = ModuleType("chipiron")
    _chipiron_stub.__path__ = [str(_CHIPIRON_PACKAGE_ROOT)]
    sys.modules["chipiron"] = _chipiron_stub

if "chipiron.environments" not in sys.modules:
    _environments_stub = ModuleType("chipiron.environments")
    _environments_stub.__path__ = [str(_ENVIRONMENTS_PACKAGE_ROOT)]
    sys.modules["chipiron.environments"] = _environments_stub

if "chipiron.environments.morpion" not in sys.modules:
    _morpion_stub = ModuleType("chipiron.environments.morpion")
    _morpion_stub.__path__ = [str(_MORPION_PACKAGE_ROOT)]
    sys.modules["chipiron.environments.morpion"] = _morpion_stub

if "chipiron.environments.morpion.bootstrap" not in sys.modules:
    _bootstrap_stub = ModuleType("chipiron.environments.morpion.bootstrap")
    _bootstrap_stub.__path__ = [str(_BOOTSTRAP_PACKAGE_ROOT)]
    sys.modules["chipiron.environments.morpion.bootstrap"] = _bootstrap_stub

if "chipiron.environments.morpion.bootstrap.bootstrap_loop" not in sys.modules:
    _bootstrap_loop_stub = ModuleType(
        "chipiron.environments.morpion.bootstrap.bootstrap_loop"
    )

    @dataclass(frozen=True, slots=True)
    class _MorpionEvaluatorSpec:
        name: str
        model_type: str
        hidden_sizes: tuple[int, ...] | None
        num_epochs: int
        batch_size: int
        learning_rate: float
        feature_subset_name: str
        feature_names: tuple[str, ...] | None = None

        @property
        def feature_subset(self) -> object:
            class _FeatureSubset:
                dimension = 5

            return _FeatureSubset()

    @dataclass(frozen=True, slots=True)
    class _MorpionBootstrapPaths:
        work_dir: Path

        @classmethod
        def from_work_dir(cls, work_dir: Path | str) -> _MorpionBootstrapPaths:
            return cls(Path(work_dir))

        @property
        def tree_snapshot_dir(self) -> Path:
            return self.work_dir / "tree_exports"

        @property
        def model_dir(self) -> Path:
            return self.work_dir / "models"

        def ensure_directories(self) -> None:
            self.tree_snapshot_dir.mkdir(parents=True, exist_ok=True)
            self.model_dir.mkdir(parents=True, exist_ok=True)

        def tree_snapshot_path_for_generation(self, generation: int) -> Path:
            return self.tree_snapshot_dir / f"generation_{generation:06d}.json"

        def model_bundle_path_for_generation(
            self,
            generation: int,
            evaluator_name: str,
        ) -> Path:
            return self.model_dir / f"generation_{generation:06d}" / evaluator_name

    _bootstrap_loop_stub.MorpionBootstrapPaths = _MorpionBootstrapPaths
    _bootstrap_loop_stub.MorpionEvaluatorSpec = _MorpionEvaluatorSpec
    sys.modules["chipiron.environments.morpion.bootstrap.bootstrap_loop"] = (
        _bootstrap_loop_stub
    )

if "chipiron.environments.morpion.bootstrap.evaluator_diagnostics" not in sys.modules:
    _diagnostics_stub = ModuleType(
        "chipiron.environments.morpion.bootstrap.evaluator_diagnostics"
    )

    @dataclass(frozen=True, slots=True)
    class _MorpionEvaluatorTrainingDiagnostics:
        generation: int
        evaluator_name: str
        dataset_size: int
        created_at: str
        representative_examples: list[object]
        worst_examples: list[object]
        mae_before: float | None
        mae_after: float | None
        max_abs_error_before: float | None
        max_abs_error_after: float | None

    def _build_evaluator_training_diagnostics(
        **_kwargs: object,
    ) -> _MorpionEvaluatorTrainingDiagnostics:
        return _MorpionEvaluatorTrainingDiagnostics(
            generation=0,
            evaluator_name="stub",
            dataset_size=0,
            created_at="",
            representative_examples=[],
            worst_examples=[],
            mae_before=None,
            mae_after=None,
            max_abs_error_before=None,
            max_abs_error_after=None,
        )

    def _save_evaluator_training_diagnostics(
        diagnostics: _MorpionEvaluatorTrainingDiagnostics,
        path: Path,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(diagnostics)) + "\n", encoding="utf-8")

    _diagnostics_stub.MorpionEvaluatorTrainingDiagnostics = (
        _MorpionEvaluatorTrainingDiagnostics
    )
    _diagnostics_stub.build_evaluator_training_diagnostics = (
        _build_evaluator_training_diagnostics
    )
    _diagnostics_stub.save_evaluator_training_diagnostics = (
        _save_evaluator_training_diagnostics
    )
    sys.modules["chipiron.environments.morpion.bootstrap.evaluator_diagnostics"] = (
        _diagnostics_stub
    )

if "chipiron.environments.morpion.bootstrap.evaluator_family" not in sys.modules:
    _family_stub = ModuleType(
        "chipiron.environments.morpion.bootstrap.evaluator_family"
    )

    @dataclass(frozen=True, slots=True)
    class _MorpionEvaluatorsConfig:
        evaluators: dict[str, _MorpionEvaluatorSpec]

    def _morpion_evaluators_config_from_preset(
        _preset: str,
    ) -> _MorpionEvaluatorsConfig:
        spec = _MorpionEvaluatorSpec(
            name="linear_5",
            model_type="linear",
            hidden_sizes=None,
            num_epochs=1,
            batch_size=1,
            learning_rate=1e-3,
            feature_subset_name="handcrafted_5_core",
        )
        return _MorpionEvaluatorsConfig(evaluators={"linear_5": spec})

    _family_stub.CANONICAL_MORPION_EVALUATOR_FAMILY_PRESET = "canonical"
    _family_stub.morpion_evaluators_config_from_preset = (
        _morpion_evaluators_config_from_preset
    )
    sys.modules["chipiron.environments.morpion.bootstrap.evaluator_family"] = (
        _family_stub
    )

if "chipiron.environments.morpion.bootstrap.evaluator_sanity_check" not in sys.modules:
    _sanity_stub = ModuleType(
        "chipiron.environments.morpion.bootstrap.evaluator_sanity_check"
    )

    class _EmptyMorpionSanityDatasetError(ValueError):
        pass

    def _build_backup_target_diagnostics(**_kwargs: object) -> dict[str, object]:
        return {}

    def _terminal_path_nodes(snapshot: object) -> tuple[object, ...]:
        return tuple(getattr(snapshot, "nodes", ()))

    def _top_terminal_path_nodes(
        snapshot: object,
        *,
        max_terminal_nodes: int,
    ) -> tuple[object, ...]:
        terminal_nodes = [
            node
            for node in getattr(snapshot, "nodes", ())
            if getattr(node, "is_terminal", False) or getattr(node, "is_exact", False)
        ]
        terminal_nodes.sort(key=lambda node: getattr(node, "depth", 0), reverse=True)
        return tuple(terminal_nodes[:max_terminal_nodes])

    _sanity_stub.EmptyMorpionSanityDatasetError = _EmptyMorpionSanityDatasetError
    _sanity_stub.MorpionSanityDatasetMode = str
    _sanity_stub.build_backup_target_diagnostics = _build_backup_target_diagnostics
    _sanity_stub.terminal_path_nodes = _terminal_path_nodes
    _sanity_stub.top_terminal_path_nodes = _top_terminal_path_nodes
    sys.modules["chipiron.environments.morpion.bootstrap.evaluator_sanity_check"] = (
        _sanity_stub
    )

if "chipiron.environments.morpion.learning" not in sys.modules:
    _learning_stub = ModuleType("chipiron.environments.morpion.learning")

    @dataclass(frozen=True, slots=True)
    class _MorpionSupervisedRow:
        node_id: str
        state_ref_payload: dict[str, object]
        target_value: float
        is_terminal: bool
        is_exact: bool
        depth: int
        visit_count: int | None
        direct_value: float | None
        over_event_label: str | None
        metadata: dict[str, object]

    @dataclass(frozen=True, slots=True)
    class _MorpionSupervisedRows:
        rows: tuple[_MorpionSupervisedRow, ...]
        metadata: dict[str, object]

    def _decode_morpion_state_ref_payload(
        payload: dict[str, object],
    ) -> dict[str, object]:
        return payload

    def _save_morpion_supervised_rows(
        rows: _MorpionSupervisedRows,
        path: Path,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "rows": [asdict(row) for row in rows.rows],
            "metadata": rows.metadata,
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    _learning_stub.MorpionSupervisedRow = _MorpionSupervisedRow
    _learning_stub.MorpionSupervisedRows = _MorpionSupervisedRows
    _learning_stub.decode_morpion_state_ref_payload = _decode_morpion_state_ref_payload
    _learning_stub.save_morpion_supervised_rows = _save_morpion_supervised_rows
    sys.modules["chipiron.environments.morpion.learning"] = _learning_stub

if "chipiron.environments.morpion.types" not in sys.modules:
    _types_stub = ModuleType("chipiron.environments.morpion.types")

    class _MorpionDynamics:
        def wrap_atomheart_state(self, state: object) -> object:
            return state

    _types_stub.MorpionDynamics = _MorpionDynamics
    sys.modules["chipiron.environments.morpion.types"] = _types_stub

if "chipiron.environments.morpion.players" not in sys.modules:
    _players_stub = ModuleType("chipiron.environments.morpion.players")
    _players_stub.__path__ = [str(_MORPION_PLAYERS_PACKAGE_ROOT)]
    sys.modules["chipiron.environments.morpion.players"] = _players_stub

if "chipiron.environments.morpion.players.evaluators" not in sys.modules:
    _evaluators_stub = ModuleType("chipiron.environments.morpion.players.evaluators")
    _evaluators_stub.__path__ = [str(_MORPION_EVALUATORS_PACKAGE_ROOT)]
    sys.modules["chipiron.environments.morpion.players.evaluators"] = _evaluators_stub

if (
    "chipiron.environments.morpion.players.evaluators.neural_networks"
    not in sys.modules
):
    _nn_stub = ModuleType(
        "chipiron.environments.morpion.players.evaluators.neural_networks"
    )
    _nn_stub.__path__ = [str(_MORPION_NN_PACKAGE_ROOT)]
    sys.modules["chipiron.environments.morpion.players.evaluators.neural_networks"] = (
        _nn_stub
    )

if (
    "chipiron.environments.morpion.players.evaluators.neural_networks.model"
    not in sys.modules
):
    _model_stub = ModuleType(
        "chipiron.environments.morpion.players.evaluators.neural_networks.model"
    )

    @dataclass(frozen=True, slots=True)
    class _MorpionRegressorArgs:
        model_kind: str = "linear"
        feature_subset_name: str | None = None
        feature_names: tuple[str, ...] | None = None
        hidden_sizes: tuple[int, ...] | None = None

    class _MorpionRegressor(torch.nn.Module):
        def __init__(self, _args: _MorpionRegressorArgs) -> None:
            super().__init__()

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return torch.zeros((features.shape[0],), dtype=torch.float32)

    def _build_morpion_regressor(
        args: _MorpionRegressorArgs,
    ) -> _MorpionRegressor:
        return _MorpionRegressor(args)

    _model_stub.MorpionRegressor = _MorpionRegressor
    _model_stub.MorpionRegressorArgs = _MorpionRegressorArgs
    _model_stub.build_morpion_regressor = _build_morpion_regressor
    sys.modules[
        "chipiron.environments.morpion.players.evaluators.neural_networks.model"
    ] = _model_stub

if (
    "chipiron.environments.morpion.players.evaluators.neural_networks.bundle"
    not in sys.modules
):
    _bundle_stub = ModuleType(
        "chipiron.environments.morpion.players.evaluators.neural_networks.bundle"
    )

    def _load_morpion_regressor_for_inference(_path: Path) -> object:
        return _MorpionRegressor(_MorpionRegressorArgs())

    _bundle_stub.load_morpion_regressor_for_inference = (
        _load_morpion_regressor_for_inference
    )
    sys.modules[
        "chipiron.environments.morpion.players.evaluators.neural_networks.bundle"
    ] = _bundle_stub

if (
    "chipiron.environments.morpion.players.evaluators.neural_networks.state_to_tensor"
    not in sys.modules
):
    _state_to_tensor_stub = ModuleType(
        "chipiron.environments.morpion.players.evaluators.neural_networks.state_to_tensor"
    )

    class _MorpionFeatureTensorConverter:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def state_to_tensor(self, _state: object) -> torch.Tensor:
            return torch.zeros((5,), dtype=torch.float32)

    _state_to_tensor_stub.MorpionFeatureTensorConverter = _MorpionFeatureTensorConverter
    sys.modules[
        "chipiron.environments.morpion.players.evaluators.neural_networks.state_to_tensor"
    ] = _state_to_tensor_stub

if (
    "chipiron.environments.morpion.players.evaluators.neural_networks.train"
    not in sys.modules
):
    _train_stub = ModuleType(
        "chipiron.environments.morpion.players.evaluators.neural_networks.train"
    )

    @dataclass(frozen=True, slots=True)
    class _MorpionTrainingArgs:
        dataset_file: Path
        output_dir: Path
        batch_size: int
        num_epochs: int
        learning_rate: float
        shuffle: bool
        model_kind: str
        feature_subset_name: str | None = None
        feature_names: tuple[str, ...] | None = None
        hidden_sizes: tuple[int, ...] | None = None

    def _train_morpion_regressor(
        _args: _MorpionTrainingArgs,
    ) -> tuple[object, dict[str, float]]:
        return _MorpionRegressor(_MorpionRegressorArgs()), {"final_loss": 0.0}

    _train_stub.MorpionTrainingArgs = _MorpionTrainingArgs
    _train_stub.train_morpion_regressor = _train_morpion_regressor
    sys.modules[
        "chipiron.environments.morpion.players.evaluators.neural_networks.train"
    ] = _train_stub

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
from chipiron.environments.morpion.bootstrap.pv_family_targets import (
    family_adjusted_targets,
    principal_variation_families_from_selected_child,
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


def _terminal_beyond_prefix_snapshot() -> TrainingTreeSnapshot:
    """Build a tree where exact/terminal anchors sit beyond a small prefix."""
    root = _node("root", child_ids=("a", "b"), depth=0)
    node_a = _node("a", parent_ids=("root",), child_ids=("a_leaf",), depth=1)
    node_b = _node("b", parent_ids=("root",), child_ids=("b_leaf",), depth=1)
    node_a_leaf = _node(
        "a_leaf",
        parent_ids=("a",),
        depth=2,
        is_terminal=True,
        backed_up_value_scalar=9.0,
    )
    node_b_leaf = _node(
        "b_leaf",
        parent_ids=("b",),
        depth=3,
        is_exact=True,
        backed_up_value_scalar=12.0,
    )
    return TrainingTreeSnapshot(
        root_node_id="root",
        nodes=(root, node_a, node_b, node_a_leaf, node_b_leaf),
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
    assert values["left"].selected_child_id == "left_leaf"
    assert values["root"].backed_up_target == 7.0
    assert values["root"].target_source == "child_backup"
    assert values["root"].selected_child_id == "left"


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


def test_principal_variation_families_from_selected_child() -> None:
    """PV families should group nodes by final selected-child representative."""
    families = principal_variation_families_from_selected_child(
        {
            "0": "2",
            "1": None,
            "2": "3",
            "3": None,
        }
    )

    assert families["3"] == ("0", "2", "3")
    assert families["1"] == ("1",)


def test_pv_mean_prediction_family_target() -> None:
    """PV mean prediction should use prediction values along each family."""
    family_targets = family_adjusted_targets(
        raw_targets={"0": 10.0, "1": 1.0, "2": 10.0, "3": 10.0},
        prediction_values={"0": 0.0, "1": 1.0, "2": 0.0, "3": 10.0},
        exact_or_terminal_node_ids=set(),
        selected_child_by_node={"0": "2", "1": None, "2": "3", "3": None},
        family_target_policy="pv_mean_prediction",
    )

    assert family_targets.effective_targets["0"] == 10.0 / 3.0
    assert family_targets.effective_targets["2"] == 10.0 / 3.0
    assert family_targets.effective_targets["3"] == 10.0 / 3.0


def test_family_smoothing_preserves_exact_terminal_target() -> None:
    """Exact/terminal nodes should remain hard ground-truth anchors."""
    family_targets = family_adjusted_targets(
        raw_targets={"0": 10.0, "1": 1.0, "2": 10.0, "3": 10.0},
        prediction_values={"0": 0.0, "1": 99.0, "2": 0.0, "3": 10.0},
        exact_or_terminal_node_ids={"1"},
        selected_child_by_node={"0": "2", "1": None, "2": "3", "3": None},
        family_target_policy="pv_mean_prediction",
    )

    assert family_targets.effective_targets["1"] == 1.0


def test_pv_blend_mean_prediction_mixes_raw_and_family_prediction() -> None:
    """PV blend should mix raw backup target with family prediction mean."""
    family_targets = family_adjusted_targets(
        raw_targets={"0": 10.0, "1": 1.0, "2": 10.0, "3": 10.0},
        prediction_values={"0": 0.0, "1": 1.0, "2": 0.0, "3": 10.0},
        exact_or_terminal_node_ids=set(),
        selected_child_by_node={"0": "2", "1": None, "2": "3", "3": None},
        family_target_policy="pv_blend_mean_prediction",
        family_prediction_blend=0.25,
    )

    expected = 0.75 * 10.0 + 0.25 * (10.0 / 3.0)
    assert family_targets.effective_targets["0"] == expected
    assert family_targets.effective_targets["2"] == expected
    assert family_targets.effective_targets["3"] == expected


def test_pv_exact_then_mean_prediction_exact_family_dominates() -> None:
    """Exact-family policies should set the whole PV family to exact target."""
    family_targets = family_adjusted_targets(
        raw_targets={"0": 72.0, "2": 72.0, "3": 72.0},
        prediction_values={"0": 5.0, "2": 10.0, "3": 20.0},
        exact_or_terminal_node_ids={"3"},
        selected_child_by_node={"0": "2", "2": "3", "3": None},
        family_target_policy="pv_exact_then_mean_prediction",
    )

    assert family_targets.effective_targets["0"] == 72.0
    assert family_targets.effective_targets["2"] == 72.0
    assert family_targets.effective_targets["3"] == 72.0
    assert family_targets.family_has_exact_by_node["0"]
    assert family_targets.family_exact_target_by_node["0"] == 72.0
    assert family_targets.family_target_rule_by_node["0"] == "pv_exact_family"
    assert family_targets.family_num_exact_by_node["0"] == 1


def test_pv_exact_then_mean_prediction_non_exact_family_uses_mean() -> None:
    """Exact-then mean should fall back to prediction mean without anchors."""
    family_targets = family_adjusted_targets(
        raw_targets={"0": 72.0, "2": 72.0, "3": 72.0},
        prediction_values={"0": 5.0, "2": 10.0, "3": 20.0},
        exact_or_terminal_node_ids=set(),
        selected_child_by_node={"0": "2", "2": "3", "3": None},
        family_target_policy="pv_exact_then_mean_prediction",
    )

    expected = 35.0 / 3.0
    assert family_targets.effective_targets["0"] == expected
    assert family_targets.effective_targets["2"] == expected
    assert family_targets.effective_targets["3"] == expected


def test_pv_exact_then_min_prediction_non_exact_family_uses_min() -> None:
    """Exact-then min should fall back to prediction min without anchors."""
    family_targets = family_adjusted_targets(
        raw_targets={"0": 72.0, "2": 72.0, "3": 72.0},
        prediction_values={"0": 5.0, "2": 10.0, "3": 20.0},
        exact_or_terminal_node_ids=set(),
        selected_child_by_node={"0": "2", "2": "3", "3": None},
        family_target_policy="pv_exact_then_min_prediction",
    )

    assert family_targets.effective_targets["0"] == 5.0
    assert family_targets.effective_targets["2"] == 5.0
    assert family_targets.effective_targets["3"] == 5.0


def test_pv_exact_then_blend_mean_prediction_exact_family_not_blended() -> None:
    """Exact-family policies should not blend exact families with predictions."""
    family_targets = family_adjusted_targets(
        raw_targets={"0": 72.0, "2": 72.0, "3": 72.0},
        prediction_values={"0": 5.0, "2": 10.0, "3": 20.0},
        exact_or_terminal_node_ids={"3"},
        selected_child_by_node={"0": "2", "2": "3", "3": None},
        family_target_policy="pv_exact_then_blend_mean_prediction",
        family_prediction_blend=0.25,
    )

    assert family_targets.effective_targets["0"] == 72.0
    assert family_targets.effective_targets["2"] == 72.0
    assert family_targets.effective_targets["3"] == 72.0


def test_pv_exact_family_multiple_exact_values_uses_max() -> None:
    """Multiple exact values in one family should use max for this max backup diagnostic."""
    family_targets = family_adjusted_targets(
        raw_targets={"0": 72.0, "2": 72.0, "3": 70.0, "4": 72.0},
        prediction_values={"0": 5.0, "2": 10.0, "3": 20.0, "4": 30.0},
        exact_or_terminal_node_ids={"3", "4"},
        selected_child_by_node={"0": "2", "2": "3", "3": "4", "4": None},
        family_target_policy="pv_exact_then_mean_prediction",
    )

    assert family_targets.effective_targets["0"] == 72.0
    assert family_targets.effective_targets["2"] == 72.0
    assert family_targets.effective_targets["3"] == 72.0
    assert family_targets.effective_targets["4"] == 72.0
    assert family_targets.family_num_exact_by_node["0"] == 2
    assert family_targets.family_target_rule_by_node["0"] == "pv_exact_family_multi_max"


def test_exact_terminal_plus_prefix_preserves_anchors_beyond_prefix() -> None:
    """Exact/terminal selection should keep anchors and ancestors beyond prefix."""
    snapshot = _terminal_beyond_prefix_snapshot()
    kept_node_ids = fitted_module._select_backup_node_ids(
        snapshot=snapshot,
        max_backup_nodes=2,
        backup_selection="exact_terminal_plus_prefix",
        top_terminal_path_count=1,
    )

    assert kept_node_ids == {"root", "a", "b", "a_leaf", "b_leaf"}


def test_top_terminal_paths_selection_keeps_paths_beyond_prefix() -> None:
    """Top-terminal path mode should keep full terminal path even over cap."""
    snapshot = _terminal_beyond_prefix_snapshot()
    kept_node_ids = fitted_module._select_backup_node_ids(
        snapshot=snapshot,
        max_backup_nodes=1,
        backup_selection="top_terminal_paths",
        top_terminal_path_count=1,
    )

    assert kept_node_ids == {"root", "b", "b_leaf"}


def test_filtered_snapshot_links_only_reference_kept_nodes() -> None:
    """Filtered snapshots should remove parent/child links to dropped nodes."""
    snapshot = _terminal_beyond_prefix_snapshot()
    filtered = fitted_module._filtered_snapshot(
        snapshot,
        kept_node_ids={"root", "b", "b_leaf"},
    )

    assert tuple(node.node_id for node in filtered.nodes) == ("root", "b", "b_leaf")
    by_id = {node.node_id: node for node in filtered.nodes}
    assert by_id["root"].child_ids == ("b",)
    assert by_id["b"].parent_ids == ("root",)
    assert by_id["b"].child_ids == ("b_leaf",)
    assert by_id["b_leaf"].parent_ids == ("b",)


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

    def _fake_feature_cache(
        snapshot: TrainingTreeSnapshot, **_kwargs: object
    ) -> object:
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
    monkeypatch.setattr(
        fitted_module, "_predict_rows", lambda *_args, **_kwargs: [0.0, 0.0]
    )

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
            backup_selection="prefix",
        )
    )

    run_dir = tmp_path / "evaluator_fitted_backup_sanity" / "test_run"
    assert (run_dir / "summary.json").is_file()
    assert (run_dir / "iteration_000" / "rows.json").is_file()
    assert (run_dir / "iteration_001" / "rows.json").is_file()
    assert (run_dir / "iteration_000" / "target_diagnostics.json").is_file()
    assert (run_dir / "iteration_001" / "diagnostics" / "linear_5.json").is_file()

    summary_data = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["num_iterations"] == 2
    assert summary["backup_nodes"] == 1
    assert summary["max_backup_nodes"] == 1
    assert summary["backup_selection"] == "prefix"
    assert summary["exact_or_terminal_backup_nodes"] == 0
    assert feature_cache_builds == [1]
    assert len(summary_data["iterations"]) == 2
    assert summary_data["iterations"][1]["mean_abs_target_change"] == 0.0
    assert summary_data["iterations"][1]["raw_target_change_mean"] == 0.0
    assert summary_data["iterations"][1]["effective_target_change_mean"] == 0.0
    assert summary_data["iterations"][1]["effective_minus_raw_mean_abs"] == 0.0
    assert summary_data["family_target_policy"] == "none"
    rows_data = json.loads(
        (run_dir / "iteration_000" / "rows.json").read_text(encoding="utf-8")
    )
    row_metadata = rows_data["rows"][0]["metadata"]
    assert row_metadata["raw_target"] == row_metadata["effective_target"]
    assert row_metadata["family_representative_node_id"] == "root"
    assert row_metadata["family_size"] == 1
    assert snapshot.nodes is original_nodes
    assert (
        cast("TrainingNodeSnapshot", snapshot.nodes[0]).backed_up_value_scalar is None
    )

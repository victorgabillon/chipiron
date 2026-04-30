"""Tests for persisted Morpion bootstrap config behavior."""
# ruff: noqa: E402

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

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
from atomheart.games.morpion import MorpionDynamics as AtomMorpionDynamics
from atomheart.games.morpion import initial_state as morpion_initial_state
from atomheart.games.morpion.checkpoints import MorpionStateCheckpointCodec

from chipiron.environments.morpion.bootstrap import (
    BOOTSTRAP_CONFIG_HASH_METADATA_KEY,
    CANONICAL_MORPION_EVALUATOR_FAMILY_PRESET,
    DEFAULT_MORPION_EVALUATOR_UPDATE_POLICY,
    DEFAULT_MORPION_PIPELINE_MODE,
    DEFAULT_MORPION_TREE_BRANCH_LIMIT,
    GROWTH_RUNTIME_MUTABLE_BOOTSTRAP_CONFIG_FIELDS,
    MORPION_BOOTSTRAP_GAME,
    MORPION_BOOTSTRAP_INITIAL_PATTERN,
    MORPION_BOOTSTRAP_INITIAL_POINT_COUNT,
    MORPION_BOOTSTRAP_VARIANT,
    IncompatibleStageBootstrapConfigError,
    MorpionBootstrapArgs,
    MorpionBootstrapConfig,
    MorpionBootstrapDatasetConfig,
    MorpionBootstrapExperimentIdentityConfig,
    MorpionBootstrapPaths,
    MorpionBootstrapRuntimeConfig,
    MorpionEvaluatorsConfig,
    MorpionEvaluatorSpec,
    UnsafeMorpionBootstrapConfigChangeError,
    RUNTIME_RELAUNCH_MUTABLE_BOOTSTRAP_CONFIG_FIELDS,
    STAGE_IRRELEVANT_BOOTSTRAP_CONFIG_FIELDS,
    bootstrap_config_from_args,
    bootstrap_config_sha256,
    bootstrap_fields_owned_by_stage,
    canonical_morpion_evaluator_family_config,
    dataset_stage_owned_bootstrap_fields,
    growth_stage_owned_bootstrap_fields,
    load_bootstrap_config,
    load_bootstrap_history,
    load_bootstrap_run_state,
    reevaluation_stage_owned_bootstrap_fields,
    run_morpion_bootstrap_loop,
    save_bootstrap_config,
    save_bootstrap_run_state,
    training_stage_owned_bootstrap_fields,
    validate_bootstrap_config_change,
    validate_stage_bootstrap_config_compatibility,
)
from chipiron.environments.morpion.bootstrap.run_state import MorpionBootstrapRunState
from chipiron.environments.morpion.players.evaluators.neural_networks import (
    MORPION_CANONICAL_FEATURE_NAMES,
)


def _feature_subset(width: int) -> tuple[str, tuple[str, ...]]:
    """Return one deterministic explicit subset selection for config tests."""
    return (
        f"handcrafted_{width}_custom",
        MORPION_CANONICAL_FEATURE_NAMES[:width],
    )


def _make_morpion_payload() -> dict[str, object]:
    """Build one real Morpion checkpoint payload from a one-step state."""
    dynamics = AtomMorpionDynamics()
    start_state = morpion_initial_state()
    first_action = dynamics.all_legal_actions(start_state)[0]
    next_state = dynamics.step(start_state, first_action).next_state
    codec = MorpionStateCheckpointCodec()
    return cast("dict[str, object]", codec.dump_state_ref(next_state))


def _make_training_snapshot(
    *, target_value: float, root_node_id: str
) -> TrainingTreeSnapshot:
    """Build one minimal valid training snapshot for config-path tests."""
    node = TrainingNodeSnapshot(
        node_id=root_node_id,
        parent_ids=(),
        child_ids=(),
        depth=2,
        state_ref_payload=_make_morpion_payload(),
        direct_value_scalar=target_value / 2.0,
        backed_up_value_scalar=target_value,
        is_terminal=True,
        is_exact=True,
        over_event_label=None,
        visit_count=7,
        metadata={"source": "bootstrap-config-test"},
    )
    return TrainingTreeSnapshot(
        root_node_id=root_node_id,
        nodes=(node,),
        metadata={"format_kind": "training_tree_snapshot", "format_version": 1},
    )


class FakeMorpionSearchRunner:
    """Tiny deterministic runner satisfying the Morpion bootstrap protocol."""

    def __init__(
        self, *, tree_sizes: tuple[int, ...], target_values: tuple[float, ...]
    ) -> None:
        """Initialize the fake runner with per-cycle tree sizes and targets."""
        self._tree_sizes = tree_sizes
        self._target_values = target_values
        self._cycle_index = -1
        self.load_calls: list[tuple[str | None, str | None]] = []

    def load_or_create(
        self,
        tree_snapshot_path: str | Path | None,
        model_bundle_path: str | Path | None,
        effective_runtime_config: object | None = None,
        *,
        reevaluate_tree: bool = False,
    ) -> None:
        """Record the latest tree/model inputs used to initialize the runner."""
        _ = effective_runtime_config, reevaluate_tree
        self.load_calls.append(
            (
                None if tree_snapshot_path is None else str(tree_snapshot_path),
                None if model_bundle_path is None else str(model_bundle_path),
            )
        )

    def grow(self, max_growth_steps: int) -> None:
        """Advance the fake runner to the next predefined tree size."""
        del max_growth_steps
        if self._cycle_index + 1 < len(self._tree_sizes):
            self._cycle_index += 1

    def export_training_tree_snapshot(self, output_path: str | Path) -> None:
        """Write one real training snapshot to ``output_path``."""
        index = max(self._cycle_index, 0)
        snapshot = _make_training_snapshot(
            target_value=self._target_values[index],
            root_node_id=f"node-{index}",
        )
        save_training_tree_snapshot(snapshot, output_path)

    def current_tree_size(self) -> int:
        """Return the current predefined tree size."""
        index = max(self._cycle_index, 0)
        return self._tree_sizes[index]


def _multi_evaluator_config() -> MorpionEvaluatorsConfig:
    """Return one representative two-evaluator bootstrap config."""
    return MorpionEvaluatorsConfig(
        evaluators={
            "linear": MorpionEvaluatorSpec(
                name="linear",
                model_type="linear",
                hidden_sizes=None,
                num_epochs=1,
                batch_size=1,
                learning_rate=1e-3,
            ),
            "mlp": MorpionEvaluatorSpec(
                name="mlp",
                model_type="mlp",
                hidden_sizes=(8, 4),
                num_epochs=1,
                batch_size=1,
                learning_rate=1e-3,
            ),
        }
    )


def _make_args(work_dir: Path) -> MorpionBootstrapArgs:
    """Build one representative bootstrap args object for config tests."""
    return MorpionBootstrapArgs(
        work_dir=work_dir,
        max_growth_steps_per_cycle=5,
        save_after_tree_growth_factor=1.5,
        save_after_seconds=10.0,
        require_exact_or_terminal=True,
        min_depth=2,
        min_visit_count=3,
        max_rows=17,
        use_backed_up_value=False,
        tree_branch_limit=96,
        batch_size=1,
        num_epochs=1,
        shuffle=False,
        evaluators_config=_multi_evaluator_config(),
    )


def _make_config() -> MorpionBootstrapConfig:
    """Build one representative bootstrap config for pure config tests."""
    return MorpionBootstrapConfig(
        experiment=MorpionBootstrapExperimentIdentityConfig(
            game=MORPION_BOOTSTRAP_GAME,
            variant=MORPION_BOOTSTRAP_VARIANT,
            initial_pattern=MORPION_BOOTSTRAP_INITIAL_PATTERN,
            initial_point_count=MORPION_BOOTSTRAP_INITIAL_POINT_COUNT,
        ),
        runtime=MorpionBootstrapRuntimeConfig(
            save_after_tree_growth_factor=2.0,
            save_after_seconds=60.0,
            max_growth_steps_per_cycle=8,
            tree_branch_limit=192,
        ),
        dataset=MorpionBootstrapDatasetConfig(
            require_exact_or_terminal=False,
            min_depth=2,
            min_visit_count=5,
            max_rows=99,
            use_backed_up_value=True,
            family_target_policy="none",
            family_prediction_blend=0.25,
        ),
        evaluators=_multi_evaluator_config(),
        evaluator_update_policy="future_only",
        pipeline_mode="single_process",
        metadata={"owner": "test"},
    )


def test_growth_worker_allows_runtime_relaunch_batch_size_drift(tmp_path: Path) -> None:
    """Growth workers may vary runtime batching without changing run semantics."""
    args = _make_args(tmp_path)
    persisted = bootstrap_config_from_args(args)
    requested = bootstrap_config_from_args(
        replace(args, max_growth_steps_per_cycle=args.max_growth_steps_per_cycle + 1)
    )

    validate_stage_bootstrap_config_compatibility(
        stage="growth",
        persisted_config=persisted,
        requested_config=requested,
    )


def test_growth_worker_allows_runtime_tree_branch_limit_drift(tmp_path: Path) -> None:
    """Growth workers may vary the tree branch limit between relaunches."""
    args = _make_args(tmp_path)
    persisted = bootstrap_config_from_args(replace(args, tree_branch_limit=1_000_000))
    requested = bootstrap_config_from_args(replace(args, tree_branch_limit=128))

    validate_stage_bootstrap_config_compatibility(
        stage="growth",
        persisted_config=persisted,
        requested_config=requested,
    )


def test_growth_worker_allows_runtime_save_after_seconds_drift(
    tmp_path: Path,
) -> None:
    """Growth workers may vary save cadence seconds between relaunches."""
    args = _make_args(tmp_path)
    persisted = bootstrap_config_from_args(replace(args, save_after_seconds=3600.0))
    requested = bootstrap_config_from_args(replace(args, save_after_seconds=10.0))

    validate_stage_bootstrap_config_compatibility(
        stage="growth",
        persisted_config=persisted,
        requested_config=requested,
    )


def test_growth_worker_allows_runtime_save_after_tree_growth_factor_drift(
    tmp_path: Path,
) -> None:
    """Growth workers may vary save cadence growth factor between relaunches."""
    args = _make_args(tmp_path)
    persisted = bootstrap_config_from_args(
        replace(args, save_after_tree_growth_factor=2.0)
    )
    requested = bootstrap_config_from_args(
        replace(args, save_after_tree_growth_factor=1.2)
    )

    validate_stage_bootstrap_config_compatibility(
        stage="growth",
        persisted_config=persisted,
        requested_config=requested,
    )


def test_loop_stage_allows_runtime_relaunch_batch_size_drift(tmp_path: Path) -> None:
    """Loop stage should allow the same runtime batching override."""
    args = _make_args(tmp_path)
    persisted = bootstrap_config_from_args(args)
    requested = bootstrap_config_from_args(
        replace(args, max_growth_steps_per_cycle=args.max_growth_steps_per_cycle + 1)
    )

    validate_stage_bootstrap_config_compatibility(
        stage="loop",
        persisted_config=persisted,
        requested_config=requested,
    )


def test_dataset_worker_rejects_tree_branch_limit_drift(tmp_path: Path) -> None:
    """Dataset workers should ignore growth-only runtime policy drift."""
    args = _make_args(tmp_path)
    persisted = bootstrap_config_from_args(replace(args, tree_branch_limit=1_000_000))
    requested = bootstrap_config_from_args(
        replace(args, tree_branch_limit=DEFAULT_MORPION_TREE_BRANCH_LIMIT)
    )

    validate_stage_bootstrap_config_compatibility(
        stage="dataset_worker",
        persisted_config=persisted,
        requested_config=requested,
    )


@pytest.mark.parametrize("stage", ["dataset_worker", "training_worker", "reevaluation"])
def test_non_growth_workers_ignore_growth_save_cadence_drift(
    tmp_path: Path, stage: str
) -> None:
    """Non-growth workers should ignore growth-only save cadence runtime drift."""
    args = _make_args(tmp_path)
    persisted = bootstrap_config_from_args(
        replace(
            args,
            save_after_seconds=3600.0,
            save_after_tree_growth_factor=2.0,
        )
    )
    requested = bootstrap_config_from_args(
        replace(
            args,
            save_after_seconds=10.0,
            save_after_tree_growth_factor=1.2,
        )
    )

    validate_stage_bootstrap_config_compatibility(
        stage=cast("object", stage),
        persisted_config=persisted,
        requested_config=requested,
    )


def test_training_worker_ignores_growth_runtime_drift(tmp_path: Path) -> None:
    """Training workers should ignore growth-only runtime fields."""
    args = _make_args(tmp_path)
    persisted = bootstrap_config_from_args(replace(args, tree_branch_limit=1_000_000))
    requested = bootstrap_config_from_args(
        replace(args, tree_branch_limit=DEFAULT_MORPION_TREE_BRANCH_LIMIT)
    )

    validate_stage_bootstrap_config_compatibility(
        stage="training_worker",
        persisted_config=persisted,
        requested_config=requested,
    )


def test_reevaluation_ignores_growth_runtime_drift(tmp_path: Path) -> None:
    """Reevaluation workers should ignore growth-only runtime fields."""
    args = _make_args(tmp_path)
    persisted = bootstrap_config_from_args(replace(args, tree_branch_limit=1_000_000))
    requested = bootstrap_config_from_args(
        replace(args, tree_branch_limit=DEFAULT_MORPION_TREE_BRANCH_LIMIT)
    )

    validate_stage_bootstrap_config_compatibility(
        stage="reevaluation",
        persisted_config=persisted,
        requested_config=requested,
    )

def test_bootstrap_config_from_args_contains_expected_fields(tmp_path: Path) -> None:
    """Args should normalize into one canonical persisted config object."""
    config = bootstrap_config_from_args(_make_args(tmp_path))

    assert config.experiment.game == MORPION_BOOTSTRAP_GAME
    assert config.experiment.variant == MORPION_BOOTSTRAP_VARIANT
    assert config.experiment.initial_pattern == MORPION_BOOTSTRAP_INITIAL_PATTERN
    assert (
        config.experiment.initial_point_count == MORPION_BOOTSTRAP_INITIAL_POINT_COUNT
    )
    assert config.dataset.max_rows == 17
    assert config.dataset.use_backed_up_value is False
    assert config.runtime.tree_branch_limit == 96
    assert set(config.evaluators.evaluators) == {"linear", "mlp"}
    assert config.evaluator_update_policy == DEFAULT_MORPION_EVALUATOR_UPDATE_POLICY
    assert config.pipeline_mode == DEFAULT_MORPION_PIPELINE_MODE


def test_bootstrap_config_from_dict_defaults_missing_phase1_fields() -> None:
    """Older persisted configs should load with phase-1 defaults."""
    config = _make_config()
    payload = {
        "experiment": {
            "game": config.experiment.game,
            "variant": config.experiment.variant,
            "initial_pattern": config.experiment.initial_pattern,
            "initial_point_count": config.experiment.initial_point_count,
        },
        "runtime": {
            "save_after_tree_growth_factor": config.runtime.save_after_tree_growth_factor,
            "save_after_seconds": config.runtime.save_after_seconds,
            "max_growth_steps_per_cycle": config.runtime.max_growth_steps_per_cycle,
            "tree_branch_limit": config.runtime.tree_branch_limit,
        },
        "dataset": {
            "require_exact_or_terminal": config.dataset.require_exact_or_terminal,
            "min_depth": config.dataset.min_depth,
            "min_visit_count": config.dataset.min_visit_count,
            "max_rows": config.dataset.max_rows,
            "use_backed_up_value": config.dataset.use_backed_up_value,
            "family_target_policy": config.dataset.family_target_policy,
            "family_prediction_blend": config.dataset.family_prediction_blend,
        },
        "evaluators": {
            "evaluators": {
                name: {
                    "name": spec.name,
                    "model_type": spec.model_type,
                    "hidden_sizes": None if spec.hidden_sizes is None else list(spec.hidden_sizes),
                    "num_epochs": spec.num_epochs,
                    "batch_size": spec.batch_size,
                    "learning_rate": spec.learning_rate,
                    "feature_subset_name": spec.feature_subset_name,
                    "feature_names": list(spec.feature_names),
                }
                for name, spec in config.evaluators.evaluators.items()
            }
        },
        "metadata": dict(config.metadata),
    }

    loaded = cast(
        "MorpionBootstrapConfig",
        __import__(
            "chipiron.environments.morpion.bootstrap.config",
            fromlist=["bootstrap_config_from_dict"],
        ).bootstrap_config_from_dict(payload),
    )

    assert loaded.evaluator_update_policy == "future_only"
    assert loaded.pipeline_mode == "single_process"


def test_first_run_writes_bootstrap_config(tmp_path: Path) -> None:
    """The loop entry should create the canonical config file on first run."""
    args = _make_args(tmp_path)
    runner = FakeMorpionSearchRunner(tree_sizes=(5,), target_values=(1.0,))

    run_morpion_bootstrap_loop(args, runner, max_cycles=1)

    config_path = MorpionBootstrapPaths.from_work_dir(tmp_path).bootstrap_config_path
    assert config_path.is_file()
    assert load_bootstrap_config(config_path) == bootstrap_config_from_args(args)


def test_safe_bootstrap_config_change_is_allowed() -> None:
    """Operational changes should be allowed for one persistent run."""
    previous = _make_config()
    current = MorpionBootstrapConfig(
        experiment=previous.experiment,
        runtime=MorpionBootstrapRuntimeConfig(
            save_after_tree_growth_factor=3.0,
            save_after_seconds=5.0,
            max_growth_steps_per_cycle=12,
            tree_branch_limit=256,
        ),
        dataset=MorpionBootstrapDatasetConfig(
            require_exact_or_terminal=True,
            min_depth=4,
            min_visit_count=1,
            max_rows=20,
            use_backed_up_value=False,
        ),
        evaluators=MorpionEvaluatorsConfig(
            evaluators={
                "linear": MorpionEvaluatorSpec(
                    name="linear",
                    model_type="mlp",
                    hidden_sizes=(16, 8),
                    num_epochs=3,
                    batch_size=2,
                    learning_rate=2e-3,
                )
            }
        ),
        metadata={"owner": "updated"},
    )

    validate_bootstrap_config_change(previous, current)


def test_dataset_worker_rejects_owned_bootstrap_field_drift(tmp_path: Path) -> None:
    """Dataset workers must match persisted dataset extraction knobs."""
    args = _make_args(tmp_path)
    persisted = bootstrap_config_from_args(args)
    requested = bootstrap_config_from_args(replace(args, min_visit_count=99))

    with pytest.raises(IncompatibleStageBootstrapConfigError, match="min_visit_count"):
        validate_stage_bootstrap_config_compatibility(
            stage="dataset_worker",
            persisted_config=persisted,
            requested_config=requested,
        )


def test_training_worker_rejects_owned_bootstrap_field_drift(tmp_path: Path) -> None:
    """Training workers must match persisted evaluator/training knobs."""
    args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        pipeline_mode="artifact_pipeline",
        num_epochs=1,
        batch_size=2,
    )
    persisted = bootstrap_config_from_args(args)
    requested = bootstrap_config_from_args(replace(args, num_epochs=3, batch_size=4))

    with pytest.raises(IncompatibleStageBootstrapConfigError, match="evaluators"):
        validate_stage_bootstrap_config_compatibility(
            stage="training_worker",
            persisted_config=persisted,
            requested_config=requested,
        )


def test_growth_worker_rejects_dataset_config_drift(tmp_path: Path) -> None:
    """Growth workers must not silently change dataset-owned fields."""
    args = _make_args(tmp_path)
    persisted = bootstrap_config_from_args(args)
    requested = bootstrap_config_from_args(replace(args, min_depth=12))

    with pytest.raises(IncompatibleStageBootstrapConfigError, match="min_depth"):
        validate_stage_bootstrap_config_compatibility(
            stage="growth",
            persisted_config=persisted,
            requested_config=requested,
        )


def test_reevaluation_worker_rejects_unrelated_config_drift(tmp_path: Path) -> None:
    """Reevaluation workers should not own bootstrap config drift."""
    args = _make_args(tmp_path)
    persisted = bootstrap_config_from_args(args)
    requested = bootstrap_config_from_args(replace(args, max_rows=1234))

    with pytest.raises(IncompatibleStageBootstrapConfigError, match="max_rows"):
        validate_stage_bootstrap_config_compatibility(
            stage="reevaluation",
            persisted_config=persisted,
            requested_config=requested,
        )


def test_package_root_reexports_stage_config_ownership_helpers() -> None:
    """The bootstrap package root should re-export public ownership helpers."""
    import chipiron.environments.morpion.bootstrap as bootstrap_package
    import chipiron.environments.morpion.bootstrap.config as config_module

    assert (
        bootstrap_package.IncompatibleStageBootstrapConfigError
        is config_module.IncompatibleStageBootstrapConfigError
    )
    assert (
        bootstrap_package.dataset_stage_owned_bootstrap_fields
        is config_module.dataset_stage_owned_bootstrap_fields
    )
    assert (
        bootstrap_package.training_stage_owned_bootstrap_fields
        is config_module.training_stage_owned_bootstrap_fields
    )
    assert (
        bootstrap_package.growth_stage_owned_bootstrap_fields
        is config_module.growth_stage_owned_bootstrap_fields
    )
    assert (
        bootstrap_package.reevaluation_stage_owned_bootstrap_fields
        is config_module.reevaluation_stage_owned_bootstrap_fields
    )
    assert (
        bootstrap_package.GROWTH_RUNTIME_MUTABLE_BOOTSTRAP_CONFIG_FIELDS
        is config_module.GROWTH_RUNTIME_MUTABLE_BOOTSTRAP_CONFIG_FIELDS
    )
    assert (
        bootstrap_package.bootstrap_fields_owned_by_stage
        is config_module.bootstrap_fields_owned_by_stage
    )
    assert (
        bootstrap_package.STAGE_IRRELEVANT_BOOTSTRAP_CONFIG_FIELDS
        is config_module.STAGE_IRRELEVANT_BOOTSTRAP_CONFIG_FIELDS
    )
    assert (
        bootstrap_package.RUNTIME_RELAUNCH_MUTABLE_BOOTSTRAP_CONFIG_FIELDS
        is config_module.RUNTIME_RELAUNCH_MUTABLE_BOOTSTRAP_CONFIG_FIELDS
    )
    assert (
        bootstrap_package.validate_stage_bootstrap_config_compatibility
        is config_module.validate_stage_bootstrap_config_compatibility
    )


def test_stage_owned_field_helpers_are_stable() -> None:
    """Stage ownership helpers should expose deterministic bootstrap field names."""
    assert "min_visit_count" in dataset_stage_owned_bootstrap_fields()
    assert "num_epochs" in training_stage_owned_bootstrap_fields()
    assert "max_growth_steps_per_cycle" in growth_stage_owned_bootstrap_fields()
    assert "tree_branch_limit" in growth_stage_owned_bootstrap_fields()
    assert "tree_branch_limit" not in dataset_stage_owned_bootstrap_fields()
    assert "tree_branch_limit" not in training_stage_owned_bootstrap_fields()
    assert reevaluation_stage_owned_bootstrap_fields() == ()
    assert bootstrap_fields_owned_by_stage("dataset_worker") == (
        dataset_stage_owned_bootstrap_fields()
    )
    assert GROWTH_RUNTIME_MUTABLE_BOOTSTRAP_CONFIG_FIELDS == {
        "max_growth_steps_per_cycle",
        "tree_branch_limit",
        "save_after_seconds",
        "save_after_tree_growth_factor",
    }
    assert RUNTIME_RELAUNCH_MUTABLE_BOOTSTRAP_CONFIG_FIELDS == {
        "max_growth_steps_per_cycle",
        "tree_branch_limit",
        "save_after_seconds",
        "save_after_tree_growth_factor",
    }
    assert STAGE_IRRELEVANT_BOOTSTRAP_CONFIG_FIELDS["dataset_worker"] == {
        "max_growth_steps_per_cycle",
        "tree_branch_limit",
        "save_after_seconds",
        "save_after_tree_growth_factor",
    }


def test_unsafe_variant_change_is_rejected() -> None:
    """Changing experiment variant must fail loudly for an existing run."""
    previous = _make_config()
    current = MorpionBootstrapConfig(
        experiment=MorpionBootstrapExperimentIdentityConfig(
            game=previous.experiment.game,
            variant="5D",
            initial_pattern=previous.experiment.initial_pattern,
            initial_point_count=previous.experiment.initial_point_count,
        ),
        runtime=previous.runtime,
        dataset=previous.dataset,
        evaluators=previous.evaluators,
    )

    with pytest.raises(UnsafeMorpionBootstrapConfigChangeError):
        validate_bootstrap_config_change(previous, current)


def test_unsafe_initial_pattern_change_is_rejected() -> None:
    """Changing the starting pattern must fail loudly for an existing run."""
    previous = _make_config()
    current = MorpionBootstrapConfig(
        experiment=MorpionBootstrapExperimentIdentityConfig(
            game=previous.experiment.game,
            variant=previous.experiment.variant,
            initial_pattern="diamond",
            initial_point_count=previous.experiment.initial_point_count,
        ),
        runtime=previous.runtime,
        dataset=previous.dataset,
        evaluators=previous.evaluators,
    )

    with pytest.raises(UnsafeMorpionBootstrapConfigChangeError):
        validate_bootstrap_config_change(previous, current)


def test_unsafe_initial_point_count_change_is_rejected() -> None:
    """Changing the starting point count must fail loudly for an existing run."""
    previous = _make_config()
    current = MorpionBootstrapConfig(
        experiment=MorpionBootstrapExperimentIdentityConfig(
            game=previous.experiment.game,
            variant=previous.experiment.variant,
            initial_pattern=previous.experiment.initial_pattern,
            initial_point_count=40,
        ),
        runtime=previous.runtime,
        dataset=previous.dataset,
        evaluators=previous.evaluators,
    )

    with pytest.raises(UnsafeMorpionBootstrapConfigChangeError):
        validate_bootstrap_config_change(previous, current)


def test_bootstrap_config_hash_is_stable_and_changes_with_content() -> None:
    """The config hash should be stable for identical config and change on diffs."""
    config = _make_config()
    same_config = _make_config()
    changed_config = MorpionBootstrapConfig(
        experiment=config.experiment,
        runtime=MorpionBootstrapRuntimeConfig(
            save_after_tree_growth_factor=9.0,
            save_after_seconds=config.runtime.save_after_seconds,
            max_growth_steps_per_cycle=config.runtime.max_growth_steps_per_cycle,
            tree_branch_limit=config.runtime.tree_branch_limit,
        ),
        dataset=MorpionBootstrapDatasetConfig(
            require_exact_or_terminal=config.dataset.require_exact_or_terminal,
            min_depth=config.dataset.min_depth,
            min_visit_count=config.dataset.min_visit_count,
            max_rows=config.dataset.max_rows,
            use_backed_up_value=not config.dataset.use_backed_up_value,
        ),
        evaluators=config.evaluators,
        metadata=config.metadata,
    )

    assert bootstrap_config_sha256(config) == bootstrap_config_sha256(same_config)
    assert bootstrap_config_sha256(config) != bootstrap_config_sha256(changed_config)


def test_bootstrap_config_roundtrip_preserves_evaluator_feature_subset(
    tmp_path: Path,
) -> None:
    """Persisted evaluator subset selection should survive config roundtrips."""
    subset_name, feature_names = _feature_subset(10)
    config = MorpionBootstrapConfig(
        experiment=_make_config().experiment,
        runtime=_make_config().runtime,
        dataset=_make_config().dataset,
        evaluators=MorpionEvaluatorsConfig(
            evaluators={
                "linear": MorpionEvaluatorSpec(
                    name="linear",
                    model_type="linear",
                    hidden_sizes=None,
                    num_epochs=1,
                    batch_size=1,
                    learning_rate=1e-3,
                    feature_subset_name=subset_name,
                    feature_names=feature_names,
                )
            }
        ),
    )
    config_path = tmp_path / "bootstrap_config.json"

    save_bootstrap_config(config, config_path)
    loaded = load_bootstrap_config(config_path)

    assert loaded == config
    assert loaded.evaluators.evaluators["linear"].feature_subset_name == subset_name
    assert loaded.evaluators.evaluators["linear"].feature_names == feature_names


def test_bootstrap_config_hash_changes_when_evaluator_subset_changes() -> None:
    """Subset-only evaluator differences should affect the bootstrap config hash."""
    subset_name_10, feature_names_10 = _feature_subset(10)
    subset_name_20, feature_names_20 = _feature_subset(20)
    base = _make_config()
    config_10 = MorpionBootstrapConfig(
        experiment=base.experiment,
        runtime=base.runtime,
        dataset=base.dataset,
        evaluators=MorpionEvaluatorsConfig(
            evaluators={
                "linear": MorpionEvaluatorSpec(
                    name="linear",
                    model_type="linear",
                    hidden_sizes=None,
                    num_epochs=1,
                    batch_size=1,
                    learning_rate=1e-3,
                    feature_subset_name=subset_name_10,
                    feature_names=feature_names_10,
                )
            }
        ),
    )
    config_20 = MorpionBootstrapConfig(
        experiment=base.experiment,
        runtime=base.runtime,
        dataset=base.dataset,
        evaluators=MorpionEvaluatorsConfig(
            evaluators={
                "linear": MorpionEvaluatorSpec(
                    name="linear",
                    model_type="linear",
                    hidden_sizes=None,
                    num_epochs=1,
                    batch_size=1,
                    learning_rate=1e-3,
                    feature_subset_name=subset_name_20,
                    feature_names=feature_names_20,
                )
            }
        ),
    )

    assert config_10.evaluators != config_20.evaluators
    assert bootstrap_config_sha256(config_10) != bootstrap_config_sha256(config_20)


def test_bootstrap_config_from_args_resolves_canonical_family_preset() -> None:
    """Bootstrap config persistence should capture the resolved canonical family specs."""
    args = MorpionBootstrapArgs(
        work_dir=Path("/tmp/morpion-family"),
        evaluator_family_preset=CANONICAL_MORPION_EVALUATOR_FAMILY_PRESET,
    )

    config = bootstrap_config_from_args(args)

    assert config.evaluators == canonical_morpion_evaluator_family_config()


def test_bootstrap_config_hash_differs_between_single_evaluator_and_family() -> None:
    """The canonical family should produce a different config hash than legacy defaults."""
    single = bootstrap_config_from_args(
        MorpionBootstrapArgs(work_dir=Path("/tmp/single"))
    )
    family = bootstrap_config_from_args(
        MorpionBootstrapArgs(
            work_dir=Path("/tmp/family"),
            evaluator_family_preset=CANONICAL_MORPION_EVALUATOR_FAMILY_PRESET,
        )
    )

    assert bootstrap_config_sha256(single) != bootstrap_config_sha256(family)


def test_loop_stores_bootstrap_config_hash_in_metadata(tmp_path: Path) -> None:
    """Accepted config hash should be persisted in run state and cycle history."""
    args = _make_args(tmp_path)
    runner = FakeMorpionSearchRunner(tree_sizes=(5,), target_values=(1.0,))
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)

    final_state = run_morpion_bootstrap_loop(args, runner, max_cycles=1)
    history = load_bootstrap_history(paths.history_jsonl_path)
    persisted_run_state = load_bootstrap_run_state(paths.run_state_path)
    expected_hash = bootstrap_config_sha256(bootstrap_config_from_args(args))

    assert final_state.metadata[BOOTSTRAP_CONFIG_HASH_METADATA_KEY] == expected_hash
    assert (
        persisted_run_state.metadata[BOOTSTRAP_CONFIG_HASH_METADATA_KEY]
        == expected_hash
    )
    assert history[-1].metadata[BOOTSTRAP_CONFIG_HASH_METADATA_KEY] == expected_hash


def test_legacy_run_without_config_file_migrates_cleanly(tmp_path: Path) -> None:
    """Older runs without a config file should write one and continue normally."""
    args = _make_args(tmp_path)
    runner = FakeMorpionSearchRunner(tree_sizes=(10,), target_values=(1.0,))
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    legacy_state = MorpionBootstrapRunState(
        generation=1,
        cycle_index=0,
        latest_tree_snapshot_path="tree_exports/generation_000001.json",
        latest_rows_path="rows/generation_000001.json",
        latest_model_bundle_paths={"default": "models/generation_000001/default"},
        active_evaluator_name="default",
        tree_size_at_last_save=10,
        last_save_unix_s=1.0,
        latest_record_status=None,
        metadata={},
    )
    save_bootstrap_run_state(legacy_state, paths.run_state_path)

    final_state = run_morpion_bootstrap_loop(args, runner, max_cycles=1)

    assert paths.bootstrap_config_path.is_file()
    assert load_bootstrap_config(
        paths.bootstrap_config_path
    ) == bootstrap_config_from_args(args)
    assert final_state.metadata[
        BOOTSTRAP_CONFIG_HASH_METADATA_KEY
    ] == bootstrap_config_sha256(bootstrap_config_from_args(args))


def test_legacy_config_without_tree_branch_limit_uses_default(tmp_path: Path) -> None:
    """Older persisted configs should pick up the default branch limit cleanly."""
    config_path = tmp_path / "bootstrap_config.json"
    config_path.write_text(
        """
{
    "dataset": {
        "max_rows": null,
        "min_depth": null,
        "min_visit_count": null,
        "require_exact_or_terminal": false,
        "use_backed_up_value": true
    },
    "evaluators": {
        "evaluators": {
            "default": {
                "batch_size": 1,
                "hidden_sizes": null,
                "learning_rate": 0.001,
                "model_type": "linear",
                "name": "default",
                "num_epochs": 1
            }
        }
    },
    "experiment": {
        "game": "morpion",
        "initial_pattern": "greek_cross",
        "initial_point_count": 36,
        "variant": "5T"
    },
    "metadata": {},
    "runtime": {
        "max_growth_steps_per_cycle": 8,
        "save_after_seconds": 60.0,
        "save_after_tree_growth_factor": 2.0
    }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    assert load_bootstrap_config(config_path).runtime.tree_branch_limit == (
        DEFAULT_MORPION_TREE_BRANCH_LIMIT
    )

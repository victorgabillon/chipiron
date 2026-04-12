"""Tests for persisted Morpion bootstrap config behavior."""
# ruff: noqa: E402

from __future__ import annotations

import sys
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
    DEFAULT_MORPION_TREE_BRANCH_LIMIT,
    MORPION_BOOTSTRAP_GAME,
    MORPION_BOOTSTRAP_INITIAL_PATTERN,
    MORPION_BOOTSTRAP_INITIAL_POINT_COUNT,
    MORPION_BOOTSTRAP_VARIANT,
    MorpionBootstrapArgs,
    MorpionBootstrapConfig,
    MorpionBootstrapDatasetConfig,
    MorpionBootstrapExperimentIdentityConfig,
    MorpionBootstrapPaths,
    MorpionBootstrapRuntimeConfig,
    MorpionEvaluatorsConfig,
    MorpionEvaluatorSpec,
    UnsafeMorpionBootstrapConfigChangeError,
    bootstrap_config_from_args,
    bootstrap_config_sha256,
    load_bootstrap_config,
    load_bootstrap_history,
    load_bootstrap_run_state,
    run_morpion_bootstrap_loop,
    save_bootstrap_config,
    save_bootstrap_run_state,
    validate_bootstrap_config_change,
)
from chipiron.environments.morpion.bootstrap.run_state import MorpionBootstrapRunState


def _make_morpion_payload() -> dict[str, object]:
    """Build one real Morpion checkpoint payload from a one-step state."""
    dynamics = AtomMorpionDynamics()
    start_state = morpion_initial_state()
    first_action = dynamics.all_legal_actions(start_state)[0]
    next_state = dynamics.step(start_state, first_action).next_state
    codec = MorpionStateCheckpointCodec()
    return cast("dict[str, object]", codec.dump_state_ref(next_state))


def _make_training_snapshot(*, target_value: float, root_node_id: str) -> TrainingTreeSnapshot:
    """Build one minimal valid training snapshot for config-path tests."""
    node = TrainingNodeSnapshot(
        node_id=root_node_id,
        parent_ids=(),
        child_ids=(),
        depth=2,
        state_ref_payload=_make_morpion_payload(),
        direct_value_scalar=target_value / 2.0,
        backed_up_value_scalar=target_value,
        is_terminal=False,
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

    def __init__(self, *, tree_sizes: tuple[int, ...], target_values: tuple[float, ...]) -> None:
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
    ) -> None:
        """Record the latest tree/model inputs used to initialize the runner."""
        _ = effective_runtime_config
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
        ),
        evaluators=_multi_evaluator_config(),
        metadata={"owner": "test"},
    )


def test_bootstrap_config_roundtrip(tmp_path: Path) -> None:
    """One persisted bootstrap config should round-trip through JSON unchanged."""
    config = _make_config()
    config_path = tmp_path / "bootstrap_config.json"

    save_bootstrap_config(config, config_path)

    assert load_bootstrap_config(config_path) == config


def test_bootstrap_config_from_args_contains_expected_fields(tmp_path: Path) -> None:
    """Args should normalize into one canonical persisted config object."""
    config = bootstrap_config_from_args(_make_args(tmp_path))

    assert config.experiment.game == MORPION_BOOTSTRAP_GAME
    assert config.experiment.variant == MORPION_BOOTSTRAP_VARIANT
    assert config.experiment.initial_pattern == MORPION_BOOTSTRAP_INITIAL_PATTERN
    assert config.experiment.initial_point_count == MORPION_BOOTSTRAP_INITIAL_POINT_COUNT
    assert config.dataset.max_rows == 17
    assert config.dataset.use_backed_up_value is False
    assert config.runtime.tree_branch_limit == 96
    assert set(config.evaluators.evaluators) == {"linear", "mlp"}


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
    assert persisted_run_state.metadata[BOOTSTRAP_CONFIG_HASH_METADATA_KEY] == expected_hash
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
    assert load_bootstrap_config(paths.bootstrap_config_path) == bootstrap_config_from_args(args)
    assert final_state.metadata[BOOTSTRAP_CONFIG_HASH_METADATA_KEY] == bootstrap_config_sha256(
        bootstrap_config_from_args(args)
    )


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
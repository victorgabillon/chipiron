"""Tests for Morpion reevaluation patch consumption helpers."""
# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

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

import chipiron.environments.morpion.bootstrap.reevaluation_patch_consumer as reevaluation_patch_consumer_module
from chipiron.environments.morpion.bootstrap import (
    MorpionBootstrapPaths,
    MorpionReevaluationPatch,
    MorpionReevaluationPatchConsumptionResult,
    MorpionReevaluationPatchRow,
    apply_pending_reevaluation_patch_to_runner,
    load_reevaluation_patch,
    save_reevaluation_patch,
)


def _unexpected_patch_apply_call_error() -> AssertionError:
    """Build the assertion used when a missing-patch call reaches the hook."""
    return AssertionError("apply_reevaluation_patch should not be called")


class _RunnerThatShouldNotBeCalled:
    """Runner test double that fails if its patch hook is invoked."""

    def apply_reevaluation_patch(self, patch: MorpionReevaluationPatch) -> int:
        """Fail loudly if the consumer calls into this hook unexpectedly."""
        del patch
        raise _unexpected_patch_apply_call_error()


class _RecordingRunner:
    """Runner test double that records the consumed reevaluation patch."""

    def __init__(self, return_value: object) -> None:
        """Initialize the runner with a configurable hook return value."""
        self.return_value = return_value
        self.received_patches: list[MorpionReevaluationPatch] = []

    def apply_reevaluation_patch(self, patch: MorpionReevaluationPatch) -> object:
        """Record the patch and return the configured hook result."""
        self.received_patches.append(patch)
        return self.return_value


class _FailingRunner:
    """Runner test double whose patch hook raises a configurable failure."""

    def __init__(self, error: Exception) -> None:
        """Initialize the runner with the desired failure."""
        self.error = error

    def apply_reevaluation_patch(self, patch: MorpionReevaluationPatch) -> int:
        """Raise the configured error when patch application is attempted."""
        del patch
        raise self.error


def _make_patch(*, patch_id: str = "patch-1") -> MorpionReevaluationPatch:
    """Build one small reevaluation patch for consumer tests."""
    return MorpionReevaluationPatch(
        patch_id=patch_id,
        created_at_utc="2026-04-28T12:00:00Z",
        evaluator_generation=2,
        evaluator_name="default",
        model_bundle_path="models/generation_000002/default",
        rows=(
            MorpionReevaluationPatchRow(
                node_id="node-a",
                direct_value=1.0,
                metadata={"source": "consumer-test"},
            ),
            MorpionReevaluationPatchRow(
                node_id="node-b",
                direct_value=2.0,
                metadata={"source": "consumer-test"},
            ),
        ),
        tree_generation=7,
        start_cursor="node-a",
        end_cursor="node-b",
        metadata={"source": "consumer-test"},
    )


def test_missing_patch_returns_missing_result(tmp_path: Path) -> None:
    """Consumer should return a stable missing result when no patch exists."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)

    result = apply_pending_reevaluation_patch_to_runner(
        paths=paths,
        runner=_RunnerThatShouldNotBeCalled(),
    )

    assert result == MorpionReevaluationPatchConsumptionResult(
        patch_found=False,
        patch_applied=False,
        patch_id=None,
        num_rows=0,
        reason="missing_patch",
    )


def test_patch_is_applied_and_deleted(tmp_path: Path) -> None:
    """Consumer should apply a pending patch and delete it after success."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    patch = _make_patch(patch_id="patch-applied")
    save_reevaluation_patch(patch, paths.pipeline_reevaluation_patch_path)
    runner = _RecordingRunner(return_value=2)

    result = apply_pending_reevaluation_patch_to_runner(paths=paths, runner=runner)

    assert result == MorpionReevaluationPatchConsumptionResult(
        patch_found=True,
        patch_applied=True,
        patch_id="patch-applied",
        num_rows=2,
        reason=None,
    )
    assert runner.received_patches == [patch]
    assert not paths.pipeline_reevaluation_patch_path.exists()


def test_patch_not_deleted_when_runner_fails(tmp_path: Path) -> None:
    """Consumer should keep the patch file when runner application fails."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    patch = _make_patch(patch_id="patch-fail")
    save_reevaluation_patch(patch, paths.pipeline_reevaluation_patch_path)

    with pytest.raises(RuntimeError, match="boom"):
        apply_pending_reevaluation_patch_to_runner(
            paths=paths,
            runner=_FailingRunner(RuntimeError("boom")),
        )

    assert load_reevaluation_patch(paths.pipeline_reevaluation_patch_path) == patch


def test_missing_runner_hook_raises_without_deleting_patch(tmp_path: Path) -> None:
    """Consumer should reject runners that do not expose the patch hook."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    patch = _make_patch(patch_id="patch-missing-hook")
    save_reevaluation_patch(patch, paths.pipeline_reevaluation_patch_path)

    with pytest.raises(NotImplementedError, match="apply_reevaluation_patch"):
        apply_pending_reevaluation_patch_to_runner(paths=paths, runner=object())

    assert load_reevaluation_patch(paths.pipeline_reevaluation_patch_path) == patch


def test_invalid_runner_return_type_raises_without_deleting_patch(tmp_path: Path) -> None:
    """Consumer should reject invalid hook return types without deleting the patch."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    patch = _make_patch(patch_id="patch-invalid-return")
    save_reevaluation_patch(patch, paths.pipeline_reevaluation_patch_path)

    with pytest.raises(TypeError, match="must return int or None"):
        apply_pending_reevaluation_patch_to_runner(
            paths=paths,
            runner=_RecordingRunner(return_value="bad"),
        )

    assert load_reevaluation_patch(paths.pipeline_reevaluation_patch_path) == patch


def test_none_runner_return_counts_all_rows(tmp_path: Path) -> None:
    """Consumer should treat a None hook return as full patch application."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    patch = _make_patch(patch_id="patch-none-return")
    save_reevaluation_patch(patch, paths.pipeline_reevaluation_patch_path)

    result = apply_pending_reevaluation_patch_to_runner(
        paths=paths,
        runner=_RecordingRunner(return_value=None),
    )

    assert result.patch_found
    assert result.patch_applied
    assert result.patch_id == patch.patch_id
    assert result.num_rows == len(patch.rows)
    assert not paths.pipeline_reevaluation_patch_path.exists()


def test_package_root_reexports_patch_consumer_api() -> None:
    """Package root should re-export the reevaluation patch consumer public API."""
    assert (
        MorpionReevaluationPatchConsumptionResult
        is reevaluation_patch_consumer_module.MorpionReevaluationPatchConsumptionResult
    )
    assert (
        apply_pending_reevaluation_patch_to_runner
        is reevaluation_patch_consumer_module.apply_pending_reevaluation_patch_to_runner
    )

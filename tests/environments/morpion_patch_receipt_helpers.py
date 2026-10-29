"""Tiny cold-checkpoint fixture shared by receipt tests and crash subprocesses."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from chipiron.environments.morpion.bootstrap import (
    MorpionBootstrapArgs,
    MorpionBootstrapPaths,
    MorpionPipelineActiveModel,
    MorpionReevaluationCursor,
    MorpionReevaluationPatch,
    MorpionReevaluationPatchRow,
    run_pipeline_growth_stage,
    save_bootstrap_run_state,
    save_pipeline_active_model,
    save_reevaluation_cursor,
)
from chipiron.environments.morpion.bootstrap.reevaluation_publication import (
    publish_reevaluation_patch_if_absent,
)
from chipiron.environments.morpion.bootstrap.run_state import (
    initialize_bootstrap_run_state,
)

if TYPE_CHECKING:
    from chipiron.environments.morpion.bootstrap.control import (
        MorpionBootstrapEffectiveRuntimeConfig,
    )

BASELINE = {"root": 2.0, "leaf": 4.0}
PATCHED = {"root": 6.0, "leaf": 8.0}
MODEL = "models/generation_000001/default"


class ColdScalarRunner:
    """Synthetic two-node runtime: exact alpha-0.5 values expose repeated blending."""

    def __init__(self) -> None:
        """Start without a runtime so every test must cold restore."""
        self.values: dict[str, float] | None = None
        self.alpha = 0.5
        self.apply_calls = 0
        self.load_paths: list[Path] = []
        self.grow_delta = 0

    def discard_uncommitted_runtime(self) -> None:
        """Require the next load to obtain all values from the persisted head."""
        self.values = None

    def load_or_create(
        self,
        tree_snapshot_path: str | Path | None,
        model_bundle_path: str | Path | None,
        effective_runtime_config: MorpionBootstrapEffectiveRuntimeConfig | None = None,
        *,
        reevaluate_tree: bool = False,
    ) -> None:
        """Never create a fresh runtime or supply fallback scientific values."""
        assert tree_snapshot_path is not None and not reevaluate_tree
        del model_bundle_path
        path = Path(tree_snapshot_path)
        self.load_paths.append(path)
        if path.is_dir():
            path /= "nodes.json"
        self.values = json.loads(path.read_bytes())
        assert effective_runtime_config is not None
        self.alpha = effective_runtime_config.reevaluation_blend_alpha

    def apply_reevaluation_patch(self, patch: MorpionReevaluationPatch) -> int:
        """Blend present nodes in supplied row order, preserving missing-node behavior."""
        assert self.values is not None
        self.apply_calls += 1
        count = 0
        for row in patch.rows:
            if row.node_id in self.values:
                self.values[row.node_id] = (1 - self.alpha) * self.values[
                    row.node_id
                ] + self.alpha * row.direct_value
                count += 1
        return count

    def grow(self, max_growth_steps: int) -> None:
        """Optionally add one synthetic node; never perform search."""
        del max_growth_steps
        assert self.values is not None
        if self.grow_delta:
            self.values["new"] = 0.0

    def current_tree_size(self) -> int:
        """Return the exact persisted synthetic node count."""
        assert self.values is not None
        return len(self.values)

    def current_tree_branch_count(self) -> int:
        """Expose an exhausted synthetic branch budget for no-growth tests."""
        return 1

    def save_checkpoint(self, output_path: str | Path) -> None:
        """Write a complete file or two-shard directory, with no durability shortcut."""
        assert self.values is not None
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".sharded":
            path.mkdir()
            (path / "manifest.json").write_text('{"shards": ["nodes.json"]}\n')
            path /= "nodes.json"
        path.write_text(json.dumps(self.values, sort_keys=True) + "\n")

    def export_training_tree_snapshot(self, output_path: str | Path) -> None:
        """Export exact fixture values; no Dataset or Training operation runs."""
        self.save_checkpoint(output_path)


def receipt_args(work_dir: Path) -> MorpionBootstrapArgs:
    """Keep normal save thresholds unreachable and use supported flat publication."""
    return MorpionBootstrapArgs(
        work_dir=work_dir,
        pipeline_mode="artifact_pipeline",
        training_export_mode="flat",
        runtime_checkpoint_format="json-zst",
        growth_skip_training_export=True,
        max_growth_steps_per_cycle=0,
        save_after_tree_growth_factor=100.0,
        save_after_seconds=1e15,
        reevaluation_blend_alpha=0.5,
        tree_branch_limit=1,
    )


def setup_receipt_baseline(
    work_dir: Path,
    *,
    adopt: bool = True,
) -> tuple[MorpionBootstrapArgs, MorpionBootstrapPaths]:
    """Adopt a quiescent synthetic legacy checkpoint before any patch publication."""
    args = receipt_args(work_dir)
    paths = MorpionBootstrapPaths.from_work_dir(work_dir)
    paths.ensure_directories()
    checkpoint = paths.runtime_checkpoint_path_for_generation(1)
    checkpoint.write_text(json.dumps(BASELINE, sort_keys=True) + "\n")
    model = paths.work_dir / MODEL
    model.mkdir(parents=True)
    (model / "param.pt").write_bytes(b"synthetic-model-identity\x00")
    save_pipeline_active_model(
        MorpionPipelineActiveModel(
            generation=1,
            evaluator_name="default",
            model_bundle_path=MODEL,
            updated_at_utc="2026-10-20T12:00:00Z",
        ),
        paths.pipeline_active_model_path,
    )
    state = replace(
        initialize_bootstrap_run_state(),
        generation=1,
        cycle_index=0,
        tree_size_at_last_save=2,
        last_save_unix_s=1e12,
        latest_runtime_checkpoint_path=paths.relative_to_work_dir(checkpoint),
    )
    save_bootstrap_run_state(state, paths.run_state_path)
    if adopt:
        run_pipeline_growth_stage(
            args, ColdScalarRunner(), max_cycles=0, checkpoint_bound_receipts=True
        )
    return args, paths


def fixture_patch(patch_id: str = "patch-1") -> MorpionReevaluationPatch:
    """Preserve nontrivial row order, missing rows and scientific metadata."""
    return MorpionReevaluationPatch(
        patch_id=patch_id,
        created_at_utc="2026-10-20T12:01:00Z",
        tree_generation=1,
        evaluator_generation=1,
        evaluator_name="default",
        model_bundle_path=MODEL,
        rows=(
            MorpionReevaluationPatchRow(
                node_id="leaf", direct_value=12.0, metadata={"order": 0}
            ),
            MorpionReevaluationPatchRow(
                node_id="root", direct_value=10.0, metadata={"order": 1}
            ),
            MorpionReevaluationPatchRow(node_id="missing", direct_value=30.0),
        ),
        start_cursor="leaf",
        end_cursor="missing",
        metadata={"source": "synthetic"},
    )


def publish_fixture_patch(
    paths: MorpionBootstrapPaths,
    patch: MorpionReevaluationPatch | None = None,
    *,
    cursor: bool = True,
) -> MorpionReevaluationPatch:
    """Use unchanged PR #78 producer publication, followed by the legacy cursor writer."""
    patch = fixture_patch() if patch is None else patch
    assert publish_reevaluation_patch_if_absent(
        patch, paths.pipeline_reevaluation_patch_path
    )
    if cursor:
        save_fixture_cursor(paths, patch)
    return patch


def save_fixture_cursor(
    paths: MorpionBootstrapPaths, patch: MorpionReevaluationPatch
) -> None:
    """Publish the existing producer cursor format without changing its semantics."""
    save_reevaluation_cursor(
        MorpionReevaluationCursor(
            evaluator_generation=patch.evaluator_generation,
            evaluator_name=patch.evaluator_name,
            model_bundle_path=patch.model_bundle_path,
            tree_generation=patch.tree_generation,
            last_patch_id=patch.patch_id,
            next_node_cursor="next-node",
            updated_at_utc="2026-10-20T12:02:00Z",
        ),
        paths.pipeline_reevaluation_cursor_path,
    )

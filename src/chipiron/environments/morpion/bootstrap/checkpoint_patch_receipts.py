"""Opt-in checkpoint-bound patch consumption; run state is the only commit record.

One Growth owner serializes restore, commit and acknowledgement. The unchanged
create-only producer may run independently. No final patch is removed before
both checkpoint commitment and producer cursor durability have been verified.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .cycle_timing import timestamp_utc_from_unix_s
from .derived.provenance import PROVENANCE_NAME, atomic_json, inside, sync_checkpoint
from .generation_publication import DurableGenerationPublication
from .generation_publication_io import PUBLICATION_KEY
from .generation_publication_io import (
    growth_receipt_ownership as _growth_receipt_ownership,
)
from .pipeline_artifacts import (
    DISCOVERY_METADATA_KEY,
    MissingMorpionPipelineArtifactError,
    MorpionPipelineGenerationManifest,
    MorpionReevaluationPatch,
    load_pipeline_active_model,
    load_pipeline_manifest,
    load_reevaluation_cursor,
    pipeline_manifest_to_dict,
    reevaluation_patch_from_dict,
)
from .reevaluation_patch_consumer import (
    MorpionReevaluationPatchConsumptionResult,
    apply_pending_reevaluation_patch_to_runner,
)
from .run_state import load_bootstrap_run_state, save_bootstrap_run_state

if TYPE_CHECKING:
    from collections.abc import Mapping
    from contextlib import AbstractContextManager

    from .bootstrap_args import MorpionBootstrapArgs
    from .bootstrap_paths import MorpionBootstrapPaths
    from .control import MorpionBootstrapEffectiveRuntimeConfig
    from .run_state import MorpionBootstrapRunState

RECEIPTS_METADATA_KEY = "checkpoint_bound_reevaluation_v1"


class CheckpointPatchReceiptError(ValueError):
    """Ambiguous identity or unsupported activation must preserve the mailbox."""

    @classmethod
    def malformed_fields(cls) -> CheckpointPatchReceiptError:
        """Return a schema error preserving the original parsing exception."""
        return cls("Malformed checkpoint receipt fields.")


def _require(condition: bool, message: str) -> None:
    """Fail closed rather than inferring missing transaction evidence."""
    if not condition:
        raise CheckpointPatchReceiptError(message)


def sync_directory(path: Path) -> None:
    """Persist a namespace operation on the supported local POSIX filesystem."""
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def growth_receipt_ownership(
    paths: MorpionBootstrapPaths,
) -> AbstractContextManager[None]:
    """Preserve PR #79's public ownership API and the same stable lock inode."""
    return _growth_receipt_ownership(paths)


def artifact_sha256(path: Path) -> str:
    """Hash exact file bytes or a sorted directory inventory without loading states."""
    _require(not path.is_symlink(), f"Symlinked transaction artifact: {path}")
    if path.is_file():
        with path.open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()
    _require(path.is_dir(), f"Missing transaction artifact: {path}")
    entries = sorted(path.rglob("*"))
    _require(
        not any(p.is_symlink() for p in entries), f"Symlinked artifact tree: {path}"
    )
    files = [p for p in entries if p.is_file()]
    _require(bool(files), f"Empty transaction artifact: {path}")
    inventory = [
        [p.relative_to(path).as_posix(), p.stat().st_size, artifact_sha256(p)]
        for p in files
    ]
    return hashlib.sha256(
        json.dumps(inventory, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class PendingPatch:
    """The exact validated bytes observed once, before any runtime mutation."""

    patch: MorpionReevaluationPatch
    content_sha256: str


def observe_pending_patch(path: Path) -> PendingPatch | None:
    """Read one immutable publication, rejecting an in-place mutation during capture."""
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            contents = handle.read()
            after = os.fstat(handle.fileno())
    except FileNotFoundError:
        return None
    _require(
        (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        == (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
        "Pending patch changed during observation.",
    )
    return PendingPatch(
        reevaluation_patch_from_dict(json.loads(contents)),
        hashlib.sha256(contents).hexdigest(),
    )


@dataclass(frozen=True)
class CheckpointPatchReceipt:
    """Application evidence stored inside the same run-state checkpoint commit."""

    patch_id: str
    content_sha256: str
    tree_generation: int
    evaluator_generation: int
    evaluator_name: str
    model_bundle_path: str
    model_sha256: str
    checkpoint_path: str
    checkpoint_sha256: str
    committed_generation: int
    rows_requested: int
    rows_applied: int
    blend_alpha: float

    def __post_init__(self) -> None:
        """Validate recovery evidence instead of accepting default scientific values."""
        for value in [self.content_sha256, self.model_sha256, self.checkpoint_sha256]:
            _require(
                isinstance(value, str)
                and len(value) == 64
                and all(c in "0123456789abcdef" for c in value),
                "Malformed receipt content identity.",
            )
        for count in [
            self.tree_generation,
            self.evaluator_generation,
            self.committed_generation,
            self.rows_requested,
            self.rows_applied,
        ]:
            _require(
                isinstance(count, int) and not isinstance(count, bool) and count >= 0,
                "Malformed receipt count/generation.",
            )
        _require(
            self.rows_applied <= self.rows_requested,
            "Invalid receipt application count.",
        )
        _require(
            self.tree_generation < self.committed_generation,
            "Invalid receipt source generation.",
        )
        _require(
            all(
                isinstance(v, str) and bool(v)
                for v in [
                    self.patch_id,
                    self.evaluator_name,
                    self.model_bundle_path,
                    self.checkpoint_path,
                ]
            ),
            "Malformed receipt identity.",
        )
        _require(
            isinstance(self.blend_alpha, (int, float))
            and not isinstance(self.blend_alpha, bool)
            and 0 <= self.blend_alpha <= 1,
            "Malformed receipt blend alpha.",
        )


def receipt_ledger(
    state: MorpionBootstrapRunState,
) -> dict[str, CheckpointPatchReceipt]:
    """Parse the cumulative receipt ledger carried by the committed checkpoint head."""
    data = state.metadata.get(RECEIPTS_METADATA_KEY)
    _require(
        isinstance(data, dict) and data.get("schema_version") == 1,
        "Missing or unsupported checkpoint receipt protocol.",
    )
    assert isinstance(data, dict)
    raw_receipts = data.get("receipts")
    _require(isinstance(raw_receipts, list), "Malformed receipt ledger.")
    assert isinstance(raw_receipts, list)
    ledger: dict[str, CheckpointPatchReceipt] = {}
    for value in raw_receipts:
        _require(isinstance(value, dict), "Malformed checkpoint receipt.")
        try:
            receipt = CheckpointPatchReceipt(**value)
        except TypeError as exc:
            raise CheckpointPatchReceiptError.malformed_fields() from exc
        _require(receipt.patch_id not in ledger, "Duplicate receipt ID in run state.")
        _require(
            receipt.committed_generation <= state.generation,
            "Receipt is ahead of checkpoint.",
        )
        ledger[receipt.patch_id] = receipt
    return ledger


def _discovery_metadata(state: MorpionBootstrapRunState) -> dict[str, object]:
    """Project one committed reference, retaining diagnostics but no history ledgers."""
    metadata = dict(state.metadata)
    publication = metadata.pop(PUBLICATION_KEY, None)
    receipts = metadata.pop(RECEIPTS_METADATA_KEY)
    metadata[DISCOVERY_METADATA_KEY] = {
        "schema_version": 2,
        "generation": state.generation,
        "publication": None if publication is None else publication["generations"][-1],
        "checkpoint": {
            "path": state.latest_runtime_checkpoint_path,
            "sha256": receipts["checkpoint_sha256"],
        }
        if publication is None
        else None,
    }
    return metadata


class GrowthPatchReceiptSession:
    """One locked Growth owner; immutable checkpoints and run-state receipts only."""

    def __init__(
        self,
        paths: MorpionBootstrapPaths,
        runner: object,
        *,
        durable_generation_publication: bool = False,
        scientific_v2_producer: Mapping[str, str] | None = None,
        closed_scientific_snapshot: bool = False,
    ) -> None:
        """Create volatile transaction context; this is never a recovery authority."""
        _require(
            scientific_v2_producer is None or durable_generation_publication,
            "Scientific v2 requires durable publication.",
        )
        self.paths = paths
        self.runner = runner
        self.pending: PendingPatch | None = None
        self.model: dict[str, Any] | None = None
        self.runtime_model: dict[str, Any] | None = None
        self.applied_count: int | None = None
        self.blend_alpha = 1.0
        self.attempt_id = uuid.uuid4().hex
        self.publication = (
            DurableGenerationPublication(
                paths,
                scientific_v2_producer=scientific_v2_producer,
                closed_scientific_snapshot=closed_scientific_snapshot,
            )
            if durable_generation_publication
            else None
        )
        self.frozen_model_path: Path | None = None

    def begin_cycle(self, model_bundle_path: str | Path | None) -> None:
        """Cold restore only; bind the selected bundle before loading or applying."""
        self.discard_runtime()
        self.pending = None
        self.model = None
        self.applied_count = None
        self.attempt_id = uuid.uuid4().hex
        self.runtime_model = (
            self.current_model() if model_bundle_path is not None else None
        )
        _require(
            (
                self.runtime_model is None
                and not self.paths.pipeline_active_model_path.exists()
            )
            or (
                self.runtime_model is not None
                and model_bundle_path is not None
                and Path(model_bundle_path)
                == inside(self.paths.work_dir, self.runtime_model["path"])
            ),
            "Selected runtime model differs from the active model identity.",
        )
        if self.publication is not None:
            self.frozen_model_path = self.publication.begin_cycle(self.runtime_model)

    def runtime_model_path(self, original: str | Path | None) -> str | Path | None:
        """Load frozen model bytes in publication mode; preserve original logical provenance."""
        return self.frozen_model_path if self.publication is not None else original

    def discard_runtime(self) -> None:
        """Do not retain partial or uncommitted values across restore boundaries."""
        discard = getattr(self.runner, "discard_uncommitted_runtime", None)
        _require(
            callable(discard), "Receipt mode requires a cold-runtime discard hook."
        )
        assert callable(discard)
        discard()

    def sync_artifact(self, path: Path) -> None:
        """Sync payload and every new ancestor entry through the existing run root."""
        sync_checkpoint(path)
        parent = path.parent
        while parent != self.paths.work_dir:
            _require(
                parent.is_relative_to(self.paths.work_dir),
                "External transaction artifact.",
            )
            sync_directory(parent)
            parent = parent.parent
        sync_directory(self.paths.work_dir)

    def validate_state(self, state: MorpionBootstrapRunState) -> Path:
        """Never fall back to an older checkpoint while retaining newer receipts."""
        ledger = receipt_ledger(state)
        data = state.metadata[RECEIPTS_METADATA_KEY]
        _require(
            data.get("generation") == state.generation, "Run-state generation mismatch."
        )
        _require(
            data.get("checkpoint_path") == state.latest_runtime_checkpoint_path,
            "Run-state checkpoint identity mismatch.",
        )
        _require(
            isinstance(state.latest_runtime_checkpoint_path, str),
            "Missing committed checkpoint.",
        )
        assert state.latest_runtime_checkpoint_path is not None
        checkpoint = inside(self.paths.work_dir, state.latest_runtime_checkpoint_path)
        _require(
            artifact_sha256(checkpoint) == data.get("checkpoint_sha256"),
            "Committed checkpoint is missing, corrupt or changed.",
        )
        for receipt in ledger.values():
            if receipt.committed_generation == state.generation:
                _require(
                    (receipt.checkpoint_path, receipt.checkpoint_sha256)
                    == (
                        state.latest_runtime_checkpoint_path,
                        data["checkpoint_sha256"],
                    ),
                    "Head receipt does not identify the committed checkpoint.",
                )
        if self.publication is not None and PUBLICATION_KEY in state.metadata:
            self.publication.validate_state(state)
        return checkpoint

    def validate_options(self, args: MorpionBootstrapArgs) -> None:
        """Reject unsupported modes before modifying any configuration or baseline."""
        if self.publication is not None and self.publication.version in (2, 3):
            _require(
                args.training_export_mode == "sharded"
                and not args.growth_skip_training_export,
                "Scientific v2 publication requires a sharded export at every saved Growth generation.",
            )
            _require(
                callable(
                    getattr(
                        self.runner,
                        "export_closed_scientific_snapshot"
                        if self.publication.version == 3
                        else "export_scientific_training_snapshot",
                        None,
                    )
                ),
                "Runner has no explicit scientific v2 export adapter.",
            )
        _require(
            not (self.paths.work_dir / PROVENANCE_NAME).exists(),
            "Ordinary receipt activation cannot modify a derived workspace.",
        )
        _require(
            args.evaluator_update_policy == "future_only",
            "Receipt recovery requires future_only; restore reevaluation is not supported.",
        )
        _require(
            self.publication is not None
            or args.growth_skip_training_export
            or args.training_export_mode == "flat",
            "Activation blocked: additive sharded exports need independent immutable publication.",
        )
        _require(
            callable(getattr(self.runner, "save_checkpoint", None)),
            "Receipt mode requires checkpoint creation.",
        )

    def validate_publication_version(self, state: MorpionBootstrapRunState) -> None:
        """Reject downgrade or implicit initialization before changing config or receipts."""
        data = state.metadata.get(PUBLICATION_KEY)
        expected = self.publication.version if self.publication is not None else None
        if data is not None:
            _require(
                isinstance(data, dict) and data.get("schema_version") == expected,
                "Explicit matching publication version required; downgrade refused.",
            )
        elif expected in (2, 3):
            _require(
                False,
                "Scientific v2 requires explicit initialization in a new isolated workspace.",
            )

    def export_scientific(self, generation: int) -> Path:
        """Invoke the v2 writer only in the selected private capsule attempt."""
        assert self.publication is not None and self.publication.producer is not None
        export = getattr(
            self.runner,
            "export_closed_scientific_snapshot"
            if self.publication.version == 3
            else "export_scientific_training_snapshot",
            None,
        )
        _require(callable(export), "Runner has no scientific v2 export adapter.")
        assert callable(export)
        if self.publication.version == 3:
            from .generation_publication_io import private_generation_validation

            assert (
                self.publication.staging is not None
                and self.publication.state is not None
            )
            assert self.publication.state.latest_tree_snapshot_path is not None
            with private_generation_validation(self.publication.staging):
                result = export(
                    self.publication.staging / "scientific_snapshot",
                    generation=generation,
                    producer=self.publication.producer,
                    committed_source=inside(
                        self.paths.work_dir,
                        self.publication.state.latest_tree_snapshot_path,
                    ),
                    _validation_result=self.publication.closed_validation,
                )
                _require(
                    isinstance(result, str | Path),
                    "Closed export adapter returned no path.",
                )
                assert isinstance(result, str | Path)
                return Path(result)
        paths = self.export_paths(sharded=True)
        result = export(
            paths.sharded_tree_snapshot_dir,
            generation=generation,
            producer=self.publication.producer,
        )
        _require(
            isinstance(result, str | Path),
            "Scientific export adapter returned no path.",
        )
        assert isinstance(result, str | Path)
        return Path(result)

    def prepare(
        self, state: MorpionBootstrapRunState, args: MorpionBootstrapArgs
    ) -> MorpionBootstrapRunState:
        """Adopt only a quiescent explicit baseline; leave production activation gated."""
        self.validate_options(args)
        self.validate_publication_version(state)
        self.discard_runtime()
        if RECEIPTS_METADATA_KEY not in state.metadata:
            _require(
                observe_pending_patch(self.paths.pipeline_reevaluation_patch_path)
                is None,
                "Activation requires a quiescent baseline without a legacy pending patch.",
            )
            _require(
                state.generation > 0
                and isinstance(state.latest_runtime_checkpoint_path, str),
                "Activation requires an existing committed checkpoint and run state.",
            )
            assert state.latest_runtime_checkpoint_path is not None
            checkpoint = inside(
                self.paths.work_dir, state.latest_runtime_checkpoint_path
            )
            self.sync_artifact(checkpoint)
            if state.latest_tree_snapshot_path is not None:
                self.sync_artifact(
                    inside(self.paths.work_dir, state.latest_tree_snapshot_path)
                )
            for manifest_path in self.paths.pipeline_dir.glob(
                "generation_*/manifest.json"
            ):
                _require(
                    load_pipeline_manifest(manifest_path).generation
                    <= state.generation,
                    "Activation found a generation ahead of the baseline; reconcile it first.",
                )
            metadata = dict(state.metadata)
            metadata[RECEIPTS_METADATA_KEY] = {
                "schema_version": 1,
                "generation": state.generation,
                "checkpoint_path": state.latest_runtime_checkpoint_path,
                "checkpoint_sha256": artifact_sha256(checkpoint),
                "receipts": [],
            }
            state = replace(state, metadata=metadata)
            save_bootstrap_run_state(state, self.paths.run_state_path, durable=True)
        checkpoint = self.validate_state(state)
        self.sync_artifact(checkpoint)
        if self.publication is not None:
            state = self.publication.prepare(state)
        # Adopt a complete state observed after an interrupted replace/fsync.
        self.sync_artifact(self.paths.run_state_path)
        self.publish_manifest(state)
        self.acknowledge_pending(state)
        if self.publication is not None:
            self.publication.prune_uncommitted(state)
        return state

    def current_model(self) -> dict[str, Any]:
        """Bind model identity to exact bundle bytes, not a mutable directory name."""
        active = load_pipeline_active_model(self.paths.pipeline_active_model_path)
        bundle = inside(self.paths.work_dir, active.model_bundle_path)
        return {
            "generation": active.generation,
            "name": active.evaluator_name,
            "path": active.model_bundle_path,
            "sha256": artifact_sha256(bundle),
            "source": active.source,
            "source_generation": active.source_generation,
            "local_trained_generation": active.local_trained_generation,
        }

    def apply(
        self,
        state: MorpionBootstrapRunState,
        runtime_config: MorpionBootstrapEffectiveRuntimeConfig,
    ) -> MorpionReevaluationPatchConsumptionResult:
        """Apply one captured identity once to this restored runtime, retaining the file."""
        pending = observe_pending_patch(self.paths.pipeline_reevaluation_patch_path)
        if pending is None:
            return MorpionReevaluationPatchConsumptionResult(
                False, False, None, 0, "missing_patch"
            )
        existing = receipt_ledger(state).get(pending.patch.patch_id)
        if existing is not None:
            _require(
                existing.content_sha256 == pending.content_sha256,
                "Conflicting reused patch ID.",
            )
            self.acknowledge_pending(state)
            return MorpionReevaluationPatchConsumptionResult(
                True, False, existing.patch_id, 0, "already_committed"
            )
        _require(
            self.pending is None, "Patch already provisionally applied to this runtime."
        )
        # Atomic visibility precedes the producer's directory durability barrier.
        # Preserve replay input even if that producer dies before confirming it.
        self.sync_artifact(self.paths.pipeline_reevaluation_patch_path)
        _require(
            observe_pending_patch(self.paths.pipeline_reevaluation_patch_path)
            == pending,
            "Pending patch changed before durable application capture.",
        )
        patch = pending.patch
        _require(
            patch.tree_generation is not None
            and 0 <= patch.tree_generation <= state.generation,
            "Patch source generation is missing or ahead of committed Growth.",
        )
        if self.publication is not None and self.publication.version in (2, 3):
            from .generation_publication_io import (
                committed_scientific_source,
                publication_records,
            )

            matching = [
                r
                for r in publication_records(state.metadata)
                if r["generation"] == patch.tree_generation
            ]
            _require(len(matching) == 1, "Patch has no committed scientific source.")
            source = inside(self.paths.work_dir, matching[0]["tree_snapshot_path"])
            _require(
                patch.metadata.get("scientific_publication")
                == committed_scientific_source(source),
                "Patch scientific publication identity mismatch.",
            )
        model = self.current_model()
        _require(
            model == self.runtime_model, "Active model changed since runtime restore."
        )
        _require(
            (patch.evaluator_generation, patch.evaluator_name, patch.model_bundle_path)
            == (model["generation"], model["name"], model["path"]),
            "Patch model/generation does not match the selected evaluator.",
        )
        self.pending, self.model = pending, model
        self.blend_alpha = runtime_config.reevaluation_blend_alpha
        try:
            result = apply_pending_reevaluation_patch_to_runner(
                paths=self.paths,
                runner=self.runner,
                delete_after_apply=False,
                patch_snapshot=patch,
            )
            _require(
                result.num_rows <= len(patch.rows), "Invalid runner application count."
            )
            self.applied_count = result.num_rows
        except BaseException:
            self.discard_runtime()
            raise
        return result

    def checkpoint_path(self, generation: int, checkpoint_format: str) -> Path:
        """Keep generation basename/codec format and never overwrite another attempt."""
        if self.publication is not None:
            return self.publication.checkpoint_path(generation, checkpoint_format)
        canonical = self.paths.runtime_checkpoint_path_for_generation_with_format(
            generation, checkpoint_format
        )
        return canonical.parent / "receipt_attempts" / self.attempt_id / canonical.name

    def export_paths(self, *, sharded: bool = False) -> MorpionBootstrapPaths:
        """Flat exports remain immutable across interrupted same-generation retries."""
        if self.publication is not None:
            return self.publication.export_paths(sharded=sharded)
        return replace(
            self.paths,
            tree_snapshot_dir=(
                self.paths.tree_snapshot_dir / "receipt_attempts" / self.attempt_id
            ),
        )

    def commit(
        self, previous: MorpionBootstrapRunState, state: MorpionBootstrapRunState
    ) -> MorpionBootstrapRunState:
        """Sync dependencies, then commit checkpoint and receipt in one run-state replace."""
        persisted = load_bootstrap_run_state(self.paths.run_state_path)
        _require(
            persisted.metadata.get(RECEIPTS_METADATA_KEY)
            == previous.metadata.get(RECEIPTS_METADATA_KEY),
            "Run state changed outside the Growth ownership lock.",
        )
        self.validate_state(previous)
        if state.generation == previous.generation:
            # RAM/diagnostic refusal: no receipt for provisional effects on the old head.
            _require(
                state.latest_runtime_checkpoint_path
                == previous.latest_runtime_checkpoint_path,
                "No-save cycle changed its checkpoint pointer.",
            )
            save_bootstrap_run_state(state, self.paths.run_state_path, durable=True)
            self.discard_runtime()
            if self.publication is not None:
                self.publication.committed(state)
                self.publication.prune_uncommitted(state)
            return state
        _require(
            state.generation == previous.generation + 1, "Non-sequential Growth commit."
        )
        _require(
            not self.paths.pipeline_manifest_path_for_generation(
                state.generation
            ).exists(),
            "Uncommitted generation already has a discovery manifest; reconcile it first.",
        )
        _require(
            isinstance(state.latest_runtime_checkpoint_path, str),
            "No resulting checkpoint.",
        )
        if self.publication is not None:
            state = self.publication.publish(state)
        assert state.latest_runtime_checkpoint_path is not None
        checkpoint = inside(self.paths.work_dir, state.latest_runtime_checkpoint_path)
        self.sync_artifact(checkpoint)
        if state.latest_tree_snapshot_path is not None:
            self.sync_artifact(
                inside(self.paths.work_dir, state.latest_tree_snapshot_path)
            )
        checkpoint_hash = artifact_sha256(checkpoint)
        ledger = receipt_ledger(previous)
        if self.pending is not None:
            _require(
                self.applied_count is not None and self.model is not None,
                "Patch application did not complete.",
            )
            assert self.applied_count is not None and self.model is not None
            _require(
                self.current_model() == self.model,
                "Active model changed before checkpoint commit.",
            )
            self.sync_artifact(inside(self.paths.work_dir, self.model["path"]))
            _require(
                observe_pending_patch(self.paths.pipeline_reevaluation_patch_path)
                == self.pending,
                "Pending patch changed before checkpoint commit.",
            )
            patch = self.pending.patch
            assert patch.tree_generation is not None
            receipt = CheckpointPatchReceipt(
                patch.patch_id,
                self.pending.content_sha256,
                patch.tree_generation,
                patch.evaluator_generation,
                patch.evaluator_name,
                patch.model_bundle_path,
                self.model["sha256"],
                state.latest_runtime_checkpoint_path,
                checkpoint_hash,
                state.generation,
                len(patch.rows),
                self.applied_count,
                self.blend_alpha,
            )
            ledger[receipt.patch_id] = receipt
        metadata = dict(state.metadata)
        metadata[RECEIPTS_METADATA_KEY] = {
            "schema_version": 1,
            "generation": state.generation,
            "checkpoint_path": state.latest_runtime_checkpoint_path,
            "checkpoint_sha256": checkpoint_hash,
            "receipts": [asdict(r) for r in ledger.values()],
        }
        state = replace(state, metadata=metadata)
        # This durable replacement, not checkpoint creation, is the commit point.
        save_bootstrap_run_state(state, self.paths.run_state_path, durable=True)
        if self.publication is not None:
            self.publication.committed(state)
        self.publish_manifest(state)
        self.acknowledge_pending(state)
        self.discard_runtime()
        if self.publication is not None:
            self.publication.prune_uncommitted(state)
        return state

    def publish_manifest(self, state: MorpionBootstrapRunState) -> None:
        """Repair a missing discovery projection after the authoritative state commit."""
        if state.latest_tree_snapshot_path is None:
            return
        path = self.paths.pipeline_manifest_path_for_generation(state.generation)
        if path.exists():
            existing = load_pipeline_manifest(path)
            _require(
                (
                    existing.generation,
                    existing.runtime_checkpoint_path,
                    existing.tree_snapshot_path,
                )
                == (
                    state.generation,
                    state.latest_runtime_checkpoint_path,
                    state.latest_tree_snapshot_path,
                ),
                "Existing generation manifest conflicts with committed run state.",
            )
            if DISCOVERY_METADATA_KEY in existing.metadata:
                projection = existing.metadata[DISCOVERY_METADATA_KEY]
                assert isinstance(projection, dict)  # Validated by the manifest type.
                # Receipt-only discovery can predate same-generation capsule adoption.
                expected = _discovery_metadata(state)[DISCOVERY_METADATA_KEY]
                assert isinstance(expected, dict)
                if projection["publication"] is None:
                    expected = {
                        **expected,
                        "publication": None,
                        "checkpoint": {
                            "path": state.latest_runtime_checkpoint_path,
                            "sha256": state.metadata[RECEIPTS_METADATA_KEY][
                                "checkpoint_sha256"
                            ],
                        },
                    }
                _require(
                    projection == expected,
                    "Existing discovery projection conflicts with committed run state.",
                )
            return
        _require(
            state.last_save_unix_s is not None,
            "Missing committed generation timestamp.",
        )
        assert state.last_save_unix_s is not None
        manifest = MorpionPipelineGenerationManifest(
            generation=state.generation,
            created_at_utc=timestamp_utc_from_unix_s(state.last_save_unix_s),
            runtime_checkpoint_path=state.latest_runtime_checkpoint_path,
            tree_snapshot_path=state.latest_tree_snapshot_path,
            metadata=_discovery_metadata(state),
        )
        atomic_json(path, pipeline_manifest_to_dict(manifest))
        self.sync_artifact(path)

    def acknowledge_pending(self, state: MorpionBootstrapRunState) -> bool:
        """Acknowledge only exact committed bytes, after durable producer cursor observation."""
        persisted = load_bootstrap_run_state(self.paths.run_state_path)
        _require(
            persisted.metadata.get(RECEIPTS_METADATA_KEY)
            == state.metadata.get(RECEIPTS_METADATA_KEY),
            "Run state changed outside the Growth ownership lock.",
        )
        self.validate_state(persisted)
        self.sync_artifact(self.paths.run_state_path)
        pending = observe_pending_patch(self.paths.pipeline_reevaluation_patch_path)
        if pending is None:
            sync_directory(self.paths.pipeline_dir)
            return False
        receipt = receipt_ledger(persisted).get(pending.patch.patch_id)
        if receipt is None:
            return False
        _require(
            receipt.content_sha256 == pending.content_sha256,
            "Conflicting reused patch ID.",
        )
        _require(
            artifact_sha256(
                self.publication.model_pin(
                    persisted, receipt.model_bundle_path, receipt.model_sha256
                )
                if self.publication is not None
                else inside(self.paths.work_dir, receipt.model_bundle_path)
            )
            == receipt.model_sha256,
            "Committed patch model content changed.",
        )
        try:
            cursor = load_reevaluation_cursor(
                self.paths.pipeline_reevaluation_cursor_path
            )
        except MissingMorpionPipelineArtifactError:
            return False
        if (
            cursor.last_patch_id,
            cursor.tree_generation,
            cursor.evaluator_generation,
            cursor.evaluator_name,
            cursor.model_bundle_path,
        ) != (
            receipt.patch_id,
            receipt.tree_generation,
            receipt.evaluator_generation,
            receipt.evaluator_name,
            receipt.model_bundle_path,
        ):
            return False
        self.sync_artifact(self.paths.pipeline_reevaluation_cursor_path)
        _require(
            load_reevaluation_cursor(self.paths.pipeline_reevaluation_cursor_path)
            == cursor,
            "Producer cursor changed during acknowledgement.",
        )
        # No other consumer can unlink under this lease; create-only producers
        # cannot replace this entry between the identity check and unlink.
        _require(
            observe_pending_patch(self.paths.pipeline_reevaluation_patch_path)
            == pending,
            "Pending patch changed during acknowledgement.",
        )
        self.paths.pipeline_reevaluation_patch_path.unlink()
        sync_directory(self.paths.pipeline_dir)
        return True

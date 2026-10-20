"""Conservative identity tracking for committed, unchanged Growth runtimes.

This owns small scalar identities only, never a tree, state, evaluator, or payload.
Eligibility is armed after checkpoint save AND pipeline run-state publication.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path

    from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
        MorpionBootstrapPaths,
    )
    from chipiron.environments.morpion.bootstrap.control import (
        MorpionBootstrapEffectiveRuntimeConfig,
    )
    from chipiron.environments.morpion.bootstrap.run_state import (
        MorpionBootstrapRunState,
    )


class GrowthRuntimeIdentityError(RuntimeError):
    """Refuse continuing from externally changed or inconsistent committed state."""

    def __init__(self, reason: str) -> None:
        """Explain why committed identity could not be established."""
        super().__init__(
            f"Growth runtime identity cannot be established: {reason}. "
            "Restart the worker from validated committed artifacts."
        )


def _identity_error(reason: str) -> GrowthRuntimeIdentityError:
    return GrowthRuntimeIdentityError(reason)


def _file_digest(path: Path) -> str | None:
    """Hash a small input artifact, preserving absence as part of identity."""
    if not path.exists():
        return None
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    """Path plus content/stat identity; a mutable path alone is insufficient."""

    path: Path
    digest: str


def artifact_identity(
    path: Path | None, *, checkpoint: bool = False
) -> ArtifactIdentity | None:
    """Fingerprint files, not live nodes; hash model contents and checkpoint manifest.

    Checkpoint member identities include inode, size, mtime and ctime. A sharded
    checkpoint also needs a complete manifest and all declared members at their
    declared sizes. This avoids reading/decompressing a tree-sized payload just
    to decide whether its just-saved immutable artifact has changed.
    """
    if path is None:
        return None
    root = path.resolve()
    if not root.exists():
        reason = f"missing artifact {root}"
        raise _identity_error(reason)
    if root.is_dir() and checkpoint:
        manifest = json.loads((root / "manifest.json").read_text())
        for shard in manifest["shards"]:
            member = (root / shard["path"]).resolve()
            if not member.is_relative_to(root) or not member.is_file():
                reason = f"incomplete checkpoint member {member}"
                raise _identity_error(reason)
            if (
                shard.get("compressed_bytes") is not None
                and member.stat().st_size != shard["compressed_bytes"]
            ):
                reason = f"checkpoint member size changed: {member}"
                raise _identity_error(reason)
    members = (
        sorted(p for p in root.rglob("*") if p.is_file()) if root.is_dir() else [root]
    )
    digest = hashlib.sha256()
    for member in members:
        stat = member.stat()
        relative = member.relative_to(root).as_posix() if root.is_dir() else member.name
        identity = (
            relative,
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )
        digest.update(repr(identity).encode())
        if not checkpoint or member.name == "manifest.json":
            digest.update((_file_digest(member) or "missing").encode())
    return ArtifactIdentity(root, digest.hexdigest())


@dataclass(frozen=True, slots=True)
class GrowthRuntimeIdentity:
    """Cycle inputs that must match before a committed runtime can be reused."""

    work_dir: Path
    generation: int
    cycle_index: int
    run_state_digest: str | None
    checkpoint: ArtifactIdentity | None
    evaluator: ArtifactIdentity | None
    active_model_digest: str | None
    control_digest: str | None
    cursor_digest: str | None
    pending_patch: bool
    force_evaluator: str | None
    runtime_config: MorpionBootstrapEffectiveRuntimeConfig
    reevaluate_tree: bool

    def same_cycle_inputs(self, other: GrowthRuntimeIdentity) -> bool:
        """Ignore the generation/checkpoint advanced by our own completed save."""
        return (
            replace(
                self,
                generation=other.generation,
                cycle_index=other.cycle_index,
                run_state_digest=other.run_state_digest,
                checkpoint=other.checkpoint,
            )
            == other
        )


def growth_runtime_identity(
    *,
    paths: MorpionBootstrapPaths,
    run_state: MorpionBootstrapRunState,
    checkpoint: Path | None,
    evaluator: Path | None,
    runtime_config: MorpionBootstrapEffectiveRuntimeConfig,
    force_evaluator: str | None,
    reevaluate_tree: bool,
) -> GrowthRuntimeIdentity:
    """Read bounded filesystem/cursor identities without loading checkpoints."""
    return GrowthRuntimeIdentity(
        work_dir=paths.work_dir.resolve(),
        generation=run_state.generation,
        cycle_index=run_state.cycle_index,
        run_state_digest=_file_digest(paths.run_state_path),
        checkpoint=artifact_identity(checkpoint, checkpoint=True),
        evaluator=artifact_identity(evaluator),
        active_model_digest=_file_digest(paths.pipeline_active_model_path),
        control_digest=_file_digest(paths.control_path),
        cursor_digest=_file_digest(paths.pipeline_reevaluation_cursor_path),
        pending_patch=paths.pipeline_reevaluation_patch_path.exists(),
        force_evaluator=force_evaluator,
        runtime_config=runtime_config,
        reevaluate_tree=reevaluate_tree,
    )


@dataclass(frozen=True, slots=True)
class LiveRuntimeStamp:
    """Bounded mutation checks for the runner-owned runtime and both RNGs."""

    runtime_id: int
    node_count: int
    branch_count: int | None
    evaluator_version: int
    search_rng_state: object
    rollout_rng_state: object
    runtime_config: MorpionBootstrapEffectiveRuntimeConfig


@dataclass(slots=True)
class WarmGrowthRuntime:
    """Track load/save/publication lifecycle without retaining the runtime graph."""

    loaded: GrowthRuntimeIdentity | None = None
    committed: GrowthRuntimeIdentity | None = None
    saved_checkpoint: ArtifactIdentity | None = None
    saved_stamp: LiveRuntimeStamp | None = None

    def invalidate(self) -> None:
        """Disarm reuse before any supported runtime mutation or save attempt."""
        self.committed = None
        self.saved_checkpoint = None
        self.saved_stamp = None

    def clear(self) -> None:
        """Forget context when replacing the live runtime."""
        self.invalidate()
        self.loaded = None

    def decision(
        self, identity: GrowthRuntimeIdentity, stamp: LiveRuntimeStamp | None
    ) -> str:
        """Return an explicit reason; external publication drift refuses safely."""
        previous = self.committed
        if previous is None:
            return "no_committed_live_runtime"
        if identity.work_dir != previous.work_dir:
            return "different_work_dir"
        if identity.run_state_digest != previous.run_state_digest:
            reason = "run_state changed outside the acknowledged publication"
            raise _identity_error(reason)
        if identity.pending_patch:
            return "pending_reevaluation_patch"
        if identity.reevaluate_tree:
            return "reevaluation_requested"
        if identity.checkpoint != previous.checkpoint:
            return "different_checkpoint"
        if (
            identity.evaluator != previous.evaluator
            or identity.active_model_digest != previous.active_model_digest
        ):
            return "evaluator_identity_changed"
        if identity.force_evaluator != previous.force_evaluator:
            return "force_evaluator_changed"
        if identity.runtime_config != previous.runtime_config:
            return "runtime_config_changed"
        if identity.cursor_digest != previous.cursor_digest:
            return "reevaluation_cursor_changed"
        if identity != previous:
            return "cycle_inputs_changed"
        if stamp is None or stamp != self.saved_stamp:
            return "live_runtime_changed_since_save"
        return "compatible_committed_checkpoint"

    def validate_before_publication(self, paths: MorpionBootstrapPaths) -> None:
        """Refuse overwriting another publisher, including during this cycle."""
        if (
            self.loaded is not None
            and self.loaded.work_dir == paths.work_dir.resolve()
            and _file_digest(paths.run_state_path) != self.loaded.run_state_digest
        ):
            reason = "run_state changed outside the acknowledged publication"
            raise _identity_error(reason)

    def record_save(self, path: Path, stamp: LiveRuntimeStamp) -> None:
        """Record only a completed checkpoint write; publication is still required."""
        self.saved_checkpoint = artifact_identity(path, checkpoint=True)
        self.saved_stamp = stamp

    def acknowledge_publication(
        self,
        *,
        paths: MorpionBootstrapPaths,
        run_state: MorpionBootstrapRunState,
        generation_published: bool,
        stamp: LiveRuntimeStamp,
    ) -> None:
        """Arm reuse only after successful save/export/manifest/run-state publication."""
        from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
            load_pipeline_manifest,
        )
        from chipiron.environments.morpion.bootstrap.run_state import (
            load_bootstrap_run_state,
        )

        loaded = self.loaded
        if loaded is not None:
            self.loaded = replace(
                loaded, run_state_digest=_file_digest(paths.run_state_path)
            )
        checkpoint = self.saved_checkpoint
        if (
            not generation_published
            or loaded is None
            or checkpoint is None
            or stamp != self.saved_stamp
        ):
            self.invalidate()
            return
        latest = paths.resolve_work_dir_path(run_state.latest_runtime_checkpoint_path)
        if (
            latest is None
            or latest.resolve() != checkpoint.path
            or stamp.node_count != run_state.tree_size_at_last_save
        ):
            self.invalidate()
            return
        if json.loads(
            json.dumps(asdict(load_bootstrap_run_state(paths.run_state_path)))
        ) != json.loads(json.dumps(asdict(run_state))):
            reason = "published run_state differs from this cycle"
            raise _identity_error(reason)
        manifest = load_pipeline_manifest(
            paths.pipeline_manifest_path_for_generation(run_state.generation)
        )
        if (
            manifest.generation != run_state.generation
            or manifest.runtime_checkpoint_path
            != run_state.latest_runtime_checkpoint_path
        ):
            reason = "generation manifest/checkpoint publication is incomplete"
            raise _identity_error(reason)
        candidate = growth_runtime_identity(
            paths=paths,
            run_state=run_state,
            checkpoint=latest,
            evaluator=None if loaded.evaluator is None else loaded.evaluator.path,
            runtime_config=loaded.runtime_config,
            force_evaluator=loaded.force_evaluator,
            reevaluate_tree=loaded.reevaluate_tree,
        )
        if (
            candidate.checkpoint != checkpoint
            or not loaded.same_cycle_inputs(candidate)
            or candidate.pending_patch
            or candidate.reevaluate_tree
        ):
            self.invalidate()
            return
        self.committed = candidate


@runtime_checkable
class WarmGrowthRunner(Protocol):
    """Optional pipeline lifecycle; third-party runners keep their existing path."""

    def load_growth_runtime(
        self,
        tree_snapshot_path: Path | None,
        model_bundle_path: Path | None,
        effective_runtime_config: MorpionBootstrapEffectiveRuntimeConfig,
        *,
        paths: MorpionBootstrapPaths,
        run_state: MorpionBootstrapRunState,
        force_evaluator: str | None,
        reevaluate_tree: bool,
    ) -> None:
        """Restore or reuse only a compatible, explicitly committed live runtime."""
        ...

    def validate_growth_publication(self, paths: MorpionBootstrapPaths) -> None:
        """Refuse stale run-state publication before writing any run-state bytes."""
        ...

    def acknowledge_growth_publication(
        self,
        *,
        paths: MorpionBootstrapPaths,
        run_state: MorpionBootstrapRunState,
        generation_published: bool,
    ) -> None:
        """Acknowledge a completed pipeline commit, never merely a checkpoint save."""
        ...

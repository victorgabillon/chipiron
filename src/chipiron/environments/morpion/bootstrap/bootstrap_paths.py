"""Path and retention helpers for Morpion bootstrap artifacts."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .bootstrap_errors import (
    InvalidBootstrapArtifactPathError,
    InvalidGenerationRetentionCountError,
)
from .history import MorpionBootstrapHistoryPaths

LOGGER = logging.getLogger(__name__)

DEFAULT_KEEP_LATEST_RUNTIME_CHECKPOINTS = 2
DEFAULT_KEEP_LATEST_TREE_EXPORTS = 2


@dataclass(frozen=True, slots=True)
class MorpionBootstrapPaths:
    """Canonical artifact locations for one Morpion bootstrap work directory."""

    work_dir: Path
    bootstrap_config_path: Path
    control_path: Path
    run_state_path: Path
    history_jsonl_path: Path
    latest_status_path: Path
    launcher_pid_path: Path
    launcher_process_state_path: Path
    launcher_stdout_log_path: Path
    launcher_stderr_log_path: Path
    tree_snapshot_dir: Path
    runtime_checkpoint_dir: Path
    rows_dir: Path
    model_dir: Path
    pipeline_dir: Path
    pipeline_active_model_path: Path

    @classmethod
    def from_work_dir(
        cls,
        work_dir: str | Path,
    ) -> MorpionBootstrapPaths:
        """Build canonical bootstrap paths for one work directory."""
        root = Path(work_dir).resolve()
        return cls(
            work_dir=root,
            bootstrap_config_path=root / "bootstrap_config.json",
            control_path=root / "control.json",
            run_state_path=root / "run_state.json",
            history_jsonl_path=root / "history.jsonl",
            latest_status_path=root / "latest_status.json",
            launcher_pid_path=root / "launcher.pid",
            launcher_process_state_path=root / "launcher_process_state.json",
            launcher_stdout_log_path=root / "launcher.out.log",
            launcher_stderr_log_path=root / "launcher.err.log",
            tree_snapshot_dir=root / "tree_exports",
            runtime_checkpoint_dir=root / "search_checkpoints",
            rows_dir=root / "rows",
            model_dir=root / "models",
            pipeline_dir=root / "pipeline",
            pipeline_active_model_path=root / "pipeline" / "active_model.json",
        )

    def ensure_directories(self) -> None:
        """Create the canonical bootstrap directories if they do not exist."""
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.tree_snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.rows_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline_dir.mkdir(parents=True, exist_ok=True)

    def tree_snapshot_path_for_generation(self, generation: int) -> Path:
        """Return the tree export path for one saved generation."""
        return self.tree_snapshot_dir / f"generation_{generation:06d}.json"

    def rows_path_for_generation(self, generation: int) -> Path:
        """Return the raw Morpion rows path for one saved generation."""
        return self.rows_dir / f"generation_{generation:06d}.json"

    def runtime_checkpoint_path_for_generation(self, generation: int) -> Path:
        """Return the runtime checkpoint path for one saved generation."""
        return self.runtime_checkpoint_dir / f"generation_{generation:06d}.json"

    def model_generation_dir_for_generation(self, generation: int) -> Path:
        """Return the model root directory for one saved generation."""
        return self.model_dir / f"generation_{generation:06d}"

    def model_bundle_path_for_generation(
        self,
        generation: int,
        evaluator_name: str,
    ) -> Path:
        """Return the model bundle directory for one evaluator and generation."""
        return self.model_generation_dir_for_generation(generation) / evaluator_name

    def pipeline_generation_dir_for_generation(self, generation: int) -> Path:
        """Return the pipeline artifact directory for one saved generation."""
        return self.pipeline_dir / f"generation_{generation:06d}"

    def pipeline_manifest_path_for_generation(self, generation: int) -> Path:
        """Return the pipeline manifest path for one saved generation."""
        return self.pipeline_generation_dir_for_generation(generation) / "manifest.json"

    def pipeline_dataset_status_path_for_generation(self, generation: int) -> Path:
        """Return the dataset-stage status path for one saved generation."""
        return (
            self.pipeline_generation_dir_for_generation(generation)
            / "dataset_status.json"
        )

    def pipeline_dataset_claim_path_for_generation(self, generation: int) -> Path:
        """Return the dataset-stage claim path for one saved generation."""
        return (
            self.pipeline_generation_dir_for_generation(generation)
            / "dataset_claim.json"
        )

    def pipeline_training_status_path_for_generation(self, generation: int) -> Path:
        """Return the training-stage status path for one saved generation."""
        return (
            self.pipeline_generation_dir_for_generation(generation)
            / "training_status.json"
        )

    def pipeline_training_claim_path_for_generation(self, generation: int) -> Path:
        """Return the training-stage claim path for one saved generation."""
        return (
            self.pipeline_generation_dir_for_generation(generation)
            / "training_claim.json"
        )

    @property
    def pipeline_reevaluation_patch_path(self) -> Path:
        """Return the singleton reevaluation-patch artifact path."""
        return self.pipeline_dir / "reevaluation_patch.json"

    @property
    def pipeline_reevaluation_cursor_path(self) -> Path:
        """Return the singleton reevaluation-cursor artifact path."""
        return self.pipeline_dir / "reevaluation_cursor.json"

    def history_paths(self) -> MorpionBootstrapHistoryPaths:
        """Return the canonical bootstrap history artifact paths."""
        return MorpionBootstrapHistoryPaths(
            work_dir=self.work_dir,
            history_jsonl_path=self.history_jsonl_path,
            latest_status_path=self.latest_status_path,
        )

    def relative_to_work_dir(self, path: str | Path) -> str:
        """Return one persisted path relative to ``work_dir`` or fail clearly."""
        raw_path = Path(path)
        if not raw_path.is_absolute():
            return raw_path.as_posix()
        try:
            return raw_path.relative_to(self.work_dir).as_posix()
        except ValueError as exc:
            raise InvalidBootstrapArtifactPathError(raw_path, self.work_dir) from exc

    def resolve_work_dir_path(self, path: str | Path | None) -> Path | None:
        """Resolve one possibly-relative persisted path against ``work_dir``."""
        if path is None:
            return None
        raw_path = Path(path)
        if raw_path.is_absolute():
            return raw_path
        return self.work_dir / raw_path


def _generation_file_sort_key(path: Path) -> int | None:
    """Return the parsed generation index for ``generation_XXXXXX.json`` files."""
    stem = path.stem
    prefix = "generation_"
    if path.suffix != ".json" or not stem.startswith(prefix):
        return None
    generation_text = stem.removeprefix(prefix)
    if not generation_text.isdigit():
        return None
    return int(generation_text)


def prune_generation_files(directory: Path, keep_latest: int = 1) -> None:
    """Delete old ``generation_*.json`` files while keeping the newest ones."""
    if keep_latest < 1:
        raise InvalidGenerationRetentionCountError(keep_latest)

    generation_files = [
        (generation, path)
        for path in directory.iterdir()
        if path.is_file()
        for generation in [_generation_file_sort_key(path)]
        if generation is not None
    ]
    generation_files.sort(key=lambda item: item[0], reverse=True)
    deleted_count = 0
    for _generation, path in generation_files[keep_latest:]:
        path.unlink()
        deleted_count += 1
        LOGGER.info("[retention] deleted path=%s", str(path))
    LOGGER.info(
        "[retention] prune_done kept=%s deleted=%s",
        min(keep_latest, len(generation_files)),
        deleted_count,
    )


__all__ = [
    "DEFAULT_KEEP_LATEST_RUNTIME_CHECKPOINTS",
    "DEFAULT_KEEP_LATEST_TREE_EXPORTS",
    "MorpionBootstrapPaths",
    "prune_generation_files",
]

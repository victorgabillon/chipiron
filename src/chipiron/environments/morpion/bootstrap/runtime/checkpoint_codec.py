"""Checkpoint codec and payload normalization for Morpion runtime."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, cast

from anemone.checkpoints import (
    AnchorCheckpointStatePayload,
    CheckpointNodeStatePayload,
    DeltaCheckpointStatePayload,
    LinooSelectorCheckpointPayload,
    SearchRuntimeCheckpointPayload,
    load_checkpoint_json_payload,
)
from atomheart.games.morpion import MorpionStateCheckpointCodec
from dacite import Config, from_dict

from .checkpoint_io import (
    CheckpointIoMetrics,
    _checkpoint_node_counts,
    _log_checkpoint_metrics,
)
from .restore_memory_logging import (
    RestoreMemoryLogger,
    current_rss_mb,
    log_morpion_checkpoint_memory_phase,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from chipiron.environments.morpion.types import MorpionDynamics, MorpionState

LOGGER = logging.getLogger(
    ".".join(("chipiron.environments.morpion.bootstrap.runtime", "runner"))
)

__all__ = [
    "CheckpointCodecProfile",
    "ChipironMorpionStateCheckpointCodec",
    "InvalidMorpionSearchCheckpointError",
    "checkpoint_profile_average_ms",
    "generation_from_checkpoint_path",
    "load_morpion_search_checkpoint_payload",
    "new_morpion_state_checkpoint_codec",
]


class InvalidMorpionSearchCheckpointError(ValueError):
    """Raised when a persisted Anemone checkpoint payload is invalid."""

    def __init__(self, path: Path, reason: str) -> None:
        """Initialize the checkpoint validation error."""
        super().__init__(f"Invalid Morpion search checkpoint at {path!s}: {reason}")


@dataclass(slots=True)
class CheckpointCodecProfile:
    """Aggregate fallback profiling for Morpion checkpoint codec calls."""

    anchor_calls: int = 0
    anchor_total_s: float = 0.0
    delta_calls: int = 0
    delta_total_s: float = 0.0
    summary_calls: int = 0
    summary_total_s: float = 0.0


@dataclass(slots=True)
class ChipironMorpionStateCheckpointCodec:
    """Thin adapter from Atomheart checkpoint codecs to Chipiron Morpion states."""

    inner: MorpionStateCheckpointCodec
    dynamics: MorpionDynamics
    profile_checkpoint: bool = False
    _profile: CheckpointCodecProfile = field(
        default_factory=CheckpointCodecProfile,
        init=False,
    )

    def dump_state_ref(self, state: MorpionState) -> object:
        """Serialize a state-ref payload for training-export compatibility only."""
        return self.inner.dump_state_ref(state.to_atomheart_state())

    def load_state_ref(self, payload: object) -> MorpionState:
        """Restore one legacy state-ref payload into Chipiron state form."""
        return self.dynamics.wrap_atomheart_state(self.inner.load_state_ref(payload))

    def dump_anchor_ref(self, state: MorpionState) -> object:
        """Serialize one full anchor snapshot for the incremental checkpoint path."""
        started_at = perf_counter()
        result = self.inner.dump_anchor_ref(state.to_atomheart_state())
        self._record_profile(
            call_count_attr="anchor_calls",
            total_s_attr="anchor_total_s",
            started_at=started_at,
        )
        return result

    def dump_delta_from_parent(
        self,
        *,
        parent_state: MorpionState,
        child_state: MorpionState,
        branch_from_parent: object | None = None,
    ) -> object:
        """Serialize one child state as a parent-relative delta."""
        _ = branch_from_parent
        started_at = perf_counter()
        result = self.inner.dump_delta_from_parent(
            parent_state=parent_state.to_atomheart_state(),
            child_state=child_state.to_atomheart_state(),
            # Chipiron branch keys may differ in orientation from Atomheart move
            # encoding; delta payload carries the canonical move already.
            branch_from_parent=None,
        )
        self._record_profile(
            call_count_attr="delta_calls",
            total_s_attr="delta_total_s",
            started_at=started_at,
        )
        return result

    def load_anchor_ref(self, anchor_ref: object) -> MorpionState:
        """Restore one anchor snapshot through Atomheart, then wrap it for Chipiron."""
        return self.dynamics.wrap_atomheart_state(
            self.inner.load_anchor_ref(anchor_ref)
        )

    def load_child_from_delta(
        self,
        *,
        parent_state: MorpionState,
        delta_ref: object,
        branch_from_parent: object | None = None,
    ) -> MorpionState:
        """Restore one child state from its parent's concrete Chipiron state."""
        _ = branch_from_parent
        return self.dynamics.wrap_atomheart_state(
            self.inner.load_child_from_delta(
                parent_state=parent_state.to_atomheart_state(),
                delta_ref=delta_ref,
                # Chipiron branch keys may differ in orientation from Atomheart move
                # encoding; delta payload carries the canonical move already.
                branch_from_parent=None,
            )
        )

    def dump_state_summary(self, state: MorpionState) -> object:
        """Serialize optional lightweight checkpoint summary metadata."""
        started_at = perf_counter()
        result = self.inner.dump_state_summary(state.to_atomheart_state())
        self._record_profile(
            call_count_attr="summary_calls",
            total_s_attr="summary_total_s",
            started_at=started_at,
        )
        return result

    def dump_state_parent_branch_for_checkpoint(
        self,
        branch_from_parent: object | None,
    ) -> object | None:
        """Bridge optional compact state-parent branch payload serialization.

        This wrapper must preserve the hook contract and forward the actual
        branch argument. Morpion's Atomheart codec may still choose to ignore it
        and return ``None`` because its compact delta payload already stores the
        canonical move needed for reconstruction.
        """
        inner_hook = getattr(
            self.inner, "dump_state_parent_branch_for_checkpoint", None
        )
        if callable(inner_hook):
            return cast("object | None", inner_hook(branch_from_parent))
        return None

    def checkpoint_profile_snapshot(self) -> dict[str, object]:
        """Return aggregate checkpoint profiling from the inner codec or fallback."""
        inner_snapshot = getattr(self.inner, "checkpoint_profile_snapshot", None)
        if callable(inner_snapshot):
            snapshot = inner_snapshot()
            snapshot_mapping = _string_key_mapping_or_none(snapshot)
            if snapshot_mapping is not None:
                return dict(snapshot_mapping)
        return {
            "chipiron_morpion_anchor_avg_ms": checkpoint_profile_average_ms(
                self._profile.anchor_total_s,
                self._profile.anchor_calls,
            ),
            "chipiron_morpion_anchor_calls": self._profile.anchor_calls,
            "chipiron_morpion_anchor_total_s": self._profile.anchor_total_s,
            "chipiron_morpion_delta_avg_ms": checkpoint_profile_average_ms(
                self._profile.delta_total_s,
                self._profile.delta_calls,
            ),
            "chipiron_morpion_delta_calls": self._profile.delta_calls,
            "chipiron_morpion_delta_total_s": self._profile.delta_total_s,
            "chipiron_morpion_summary_avg_ms": checkpoint_profile_average_ms(
                self._profile.summary_total_s,
                self._profile.summary_calls,
            ),
            "chipiron_morpion_summary_calls": self._profile.summary_calls,
            "chipiron_morpion_summary_total_s": self._profile.summary_total_s,
        }

    def reset_checkpoint_profile(self) -> None:
        """Clear aggregate checkpoint profiling counters between builds."""
        inner_reset = getattr(self.inner, "reset_checkpoint_profile", None)
        if callable(inner_reset):
            inner_reset()
            return
        self._profile = CheckpointCodecProfile()

    def _record_profile(
        self,
        *,
        call_count_attr: str,
        total_s_attr: str,
        started_at: float,
    ) -> None:
        """Record one fallback codec timing when Atomheart profiling is absent."""
        if not self.profile_checkpoint:
            return
        if callable(getattr(self.inner, "checkpoint_profile_snapshot", None)):
            return
        elapsed_s = perf_counter() - started_at
        setattr(
            self._profile,
            call_count_attr,
            getattr(self._profile, call_count_attr) + 1,
        )
        setattr(
            self._profile,
            total_s_attr,
            getattr(self._profile, total_s_attr) + elapsed_s,
        )


def checkpoint_profile_average_ms(total_s: float, count: int) -> float:
    """Return a stable milliseconds average for checkpoint profile logs."""
    if count <= 0:
        return 0.0
    return 1000.0 * total_s / count


def new_morpion_state_checkpoint_codec(
    *, profile_checkpoint: bool
) -> MorpionStateCheckpointCodec:
    """Create the Morpion checkpoint codec with optional profiling when supported."""
    try:
        return MorpionStateCheckpointCodec(profile_checkpoint=profile_checkpoint)
    except TypeError:
        return MorpionStateCheckpointCodec()


def load_morpion_search_checkpoint_payload(
    path: str | Path,
    *,
    restore_memory_logger: RestoreMemoryLogger | None = None,
) -> SearchRuntimeCheckpointPayload:
    """Load a persisted search checkpoint payload and validate shape."""
    resolved_path = Path(path)
    LOGGER.info("[checkpoint] load_start path=%s", str(resolved_path))
    rss_before_mb = current_rss_mb()
    log_morpion_checkpoint_memory_phase(
        "before_candidate_payload_load",
        path=resolved_path,
        generation=generation_from_checkpoint_path(resolved_path),
    )
    started_at = time.perf_counter()
    try:
        raw_payload: object | None = None

        def log_read_phase(phase: str, metadata: Mapping[str, object]) -> None:
            if restore_memory_logger is None:
                return
            phase_raw_payload = metadata.get("raw_payload")
            log_metadata = {
                key: value for key, value in metadata.items() if key != "raw_payload"
            }
            restore_memory_logger.log(
                phase,
                raw_payload=(
                    phase_raw_payload if phase == "after_raw_json_decode" else None
                ),
                raw_checkpoint_referenced=phase == "after_raw_json_decode",
                typed_checkpoint_referenced=False,
                **cast("dict[str, Any]", log_metadata),
            )

        raw_payload, read_stats = load_checkpoint_json_payload(
            resolved_path,
            read_phase_logger=log_read_phase if restore_memory_logger else None,
        )
        LOGGER.info(
            "[checkpoint] json_load_done path=%s format=%s elapsed=%.3fs bytes=%s",
            str(resolved_path),
            read_stats.file_format,
            read_stats.json_load_s,
            read_stats.compressed_bytes,
        )
    except FileNotFoundError as exc:
        raise InvalidMorpionSearchCheckpointError(
            resolved_path,
            "file does not exist",
        ) from exc
    except Exception as exc:
        raise InvalidMorpionSearchCheckpointError(
            resolved_path,
            f"invalid checkpoint payload: {exc}",
        ) from exc

    try:
        payload_decode_started_at = time.perf_counter()
        normalized_payload = _normalize_search_checkpoint_payload_for_dacite(
            raw_payload
        )
        payload = from_dict(
            data_class=SearchRuntimeCheckpointPayload,
            data=normalized_payload,
            config=Config(cast=[tuple], check_types=False),
        )
        if restore_memory_logger is not None:
            restore_memory_logger.log(
                "after_typed_checkpoint_payload_build",
                raw_payload=raw_payload,
                typed_payload=payload,
                raw_checkpoint_referenced=True,
                typed_checkpoint_referenced=True,
            )
        payload_decode_elapsed_s = time.perf_counter() - payload_decode_started_at
        LOGGER.info(
            "[checkpoint] payload_decode_done path=%s elapsed=%.3fs",
            str(resolved_path),
            payload_decode_elapsed_s,
        )
        total_s = time.perf_counter() - started_at
        rss_after_mb = current_rss_mb()
        node_count, anchor_count, delta_count = _checkpoint_node_counts(payload)
        log_morpion_checkpoint_memory_phase(
            "after_candidate_payload_load",
            path=resolved_path,
            nodes=node_count,
            generation=generation_from_checkpoint_path(resolved_path),
        )
        _log_checkpoint_metrics(
            "payload_load",
            CheckpointIoMetrics(
                path=str(resolved_path),
                bytes=read_stats.compressed_bytes,
                file_format=read_stats.file_format,
                json_load_s=read_stats.json_load_s,
                payload_decode_s=payload_decode_elapsed_s,
                total_s=total_s,
                rss_before_mb=rss_before_mb,
                rss_after_mb=rss_after_mb,
                node_count=node_count,
                anchor_count=anchor_count,
                delta_count=delta_count,
                cache="miss",
            ),
        )
        del raw_payload
        del normalized_payload
        if restore_memory_logger is not None:
            restore_memory_logger.log(
                "after_drop_raw_checkpoint_payload_if_applicable",
                typed_payload=payload,
                raw_checkpoint_referenced=False,
                typed_checkpoint_referenced=True,
            )
    except Exception as exc:
        raise InvalidMorpionSearchCheckpointError(
            resolved_path,
            f"payload shape is invalid: {exc}",
        ) from exc
    return payload


def generation_from_checkpoint_path(path: str | Path) -> int | None:
    """Return generation number from canonical checkpoint filenames when present."""
    stem_parts = Path(path).name.split(".")
    if not stem_parts:
        return None
    stem = stem_parts[0]
    if not stem.startswith("generation_"):
        return None
    try:
        return int(stem.removeprefix("generation_"))
    except ValueError:
        return None


def _mapping(data: object) -> dict[str, object] | None:
    """Return ``data`` as a mutable mapping when possible."""
    mapping = _string_key_mapping_or_none(data)
    if mapping is None:
        return None
    return dict(mapping)


def _string_key_mapping_or_none(data: object) -> Mapping[str, object] | None:
    """Return ``data`` as a string-keyed mapping when possible."""
    if not isinstance(data, dict):
        return None
    raw_mapping = cast("dict[object, object]", data)
    if not all(isinstance(key, str) for key in raw_mapping):
        return None
    return cast("Mapping[str, object]", raw_mapping)


def _normalize_search_checkpoint_payload_for_dacite(
    raw_payload: object,
) -> dict[str, object]:
    """Normalize union payload fields so dacite can rebuild checkpoint dataclasses."""
    normalized_payload = _mapping(raw_payload)
    if normalized_payload is None:
        raise _invalid_checkpoint_payload_mapping_error()

    normalized_selector_state = _mapping(normalized_payload.get("selector_state"))
    if normalized_selector_state is not None:
        normalized_payload["selector_state"] = from_dict(
            data_class=LinooSelectorCheckpointPayload,
            data=normalized_selector_state,
            config=Config(cast=[tuple], check_types=False),
        )

    normalized_tree = _mapping(normalized_payload.get("tree"))
    if normalized_tree is None:
        return normalized_payload

    raw_nodes = normalized_tree.get("nodes")
    if not isinstance(raw_nodes, list):
        return normalized_payload
    node_payloads = cast("list[object]", raw_nodes)

    normalized_tree["nodes"] = [
        _normalize_algorithm_node_payload_for_dacite(node_payload)
        for node_payload in node_payloads
    ]
    normalized_payload["tree"] = normalized_tree
    return normalized_payload


def _normalize_algorithm_node_payload_for_dacite(node_payload: object) -> object:
    """Normalize one algorithm-node payload before dacite reconstruction."""
    normalized_node_payload = _mapping(node_payload)
    if normalized_node_payload is None:
        return node_payload
    normalized_node_payload["state_payload"] = _checkpoint_state_payload_from_dict(
        normalized_node_payload.get("state_payload")
    )
    return normalized_node_payload


def _checkpoint_state_payload_from_dict(
    raw_state_payload: object,
) -> CheckpointNodeStatePayload | object:
    """Decode the explicit state-payload union used by Anemone checkpoints.

    Delta payloads carry their own state-parent edge fields; those are distinct
    from the graph/debug representative parent fields on algorithm nodes.
    """
    normalized_state_payload = _mapping(raw_state_payload)
    if normalized_state_payload is None:
        return raw_state_payload
    if "anchor_ref" in normalized_state_payload:
        return from_dict(
            data_class=AnchorCheckpointStatePayload,
            data=cast("Any", normalized_state_payload),
            config=Config(cast=[tuple], check_types=False),
        )
    if "delta_ref" in normalized_state_payload:
        return from_dict(
            data_class=DeltaCheckpointStatePayload,
            data=cast("Any", normalized_state_payload),
            config=Config(cast=[tuple], check_types=False),
        )
    return raw_state_payload


def _live_compact_state_payload_cycle_error(node_id: int) -> RuntimeError:
    """Return the stable live compact payload-cycle error."""
    return RuntimeError(f"Cycle in live compact state payload chain at {node_id}.")


def _invalid_checkpoint_payload_mapping_error() -> TypeError:
    """Return the stable invalid checkpoint-payload mapping error."""
    return TypeError("checkpoint payload must be a string-keyed mapping")

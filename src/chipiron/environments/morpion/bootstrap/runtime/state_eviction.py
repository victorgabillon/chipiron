"""Live state-eviction helpers for Morpion bootstrap runtimes."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING, Any, cast

from anemone.checkpoints import (
    AnchorCheckpointStatePayload,
    CheckpointNodeStatePayload,
    DeltaCheckpointStatePayload,
)

if TYPE_CHECKING:
    from chipiron.environments.morpion.types import MorpionState


def _live_compact_state_payload_cycle_error(node_id: int) -> RuntimeError:
    """Return the stable live compact payload-cycle error."""
    return RuntimeError(f"Cycle in live compact state payload chain at {node_id}.")


def _phase_delta(
    before: Mapping[str, int] | Mapping[str, float],
    after: Mapping[str, int] | Mapping[str, float],
    *,
    prefix: str,
) -> dict[str, int | float]:
    """Return changed phase counters for phases matching ``prefix``."""
    delta: dict[str, int | float] = {}
    phase_names = {
        phase
        for phase in set(before) | set(after)
        if phase == prefix or phase.startswith(f"{prefix}.")
    }
    for phase in sorted(phase_names):
        phase_delta = after.get(phase, 0) - before.get(phase, 0)
        if phase_delta:
            delta[phase] = phase_delta
    return delta


@dataclass(frozen=True, slots=True)
class _ParentDeltaContext:
    """Single unambiguous parent edge eligible for live delta payloads."""

    parent_node: object
    parent_node_id: int
    branch_from_parent: object


@dataclass(frozen=True, slots=True)
class _LiveEvictionPayload:
    """Built payload plus live-chain metadata for one evicted node."""

    payload: CheckpointNodeStatePayload
    kind: str
    chain_depth: int


@dataclass(slots=True)
class MorpionGrowthStateEvictionMetrics:
    """Counters for experimental growth-time state eviction."""

    state_eviction_policy: str = "none"
    state_eviction_payload_mode: str = "anchor"
    state_eviction_delta_chain_max_depth: int = 32
    eviction_attempt_count: int = 0
    eviction_success_count: int = 0
    evicted_materialized_state_count: int = 0
    compact_payload_count: int = 0
    anchor_payload_count: int = 0
    delta_payload_count: int = 0
    rematerialization_count: int = 0
    rematerialization_cache_hit: int = 0
    rematerialization_cache_miss: int = 0
    rematerialization_count_by_phase: dict[str, int] = field(default_factory=dict)
    rematerialization_cache_hit_by_phase: dict[str, int] = field(default_factory=dict)
    rematerialization_cache_miss_by_phase: dict[str, int] = field(default_factory=dict)
    rematerialization_total_s_by_phase: dict[str, float] = field(default_factory=dict)
    rematerialization_count_by_node_id: dict[int, int] = field(default_factory=dict)
    current_cache_size: int = 0
    cache_evictions: int = 0
    total_reconstruction_depth: int = 0
    eviction_scan_count: int = 0
    eviction_nodes_scanned_count: int = 0
    eviction_total_s: float = 0.0
    eviction_payload_build_s: float = 0.0
    rematerialization_total_s: float = 0.0
    eviction_skipped_count_by_reason: dict[str, int] = field(default_factory=dict)
    delta_payload_attempt_count: int = 0
    delta_payload_success_count: int = 0
    delta_payload_fallback_count: int = 0
    delta_payload_fallback_count_by_reason: dict[str, int] = field(default_factory=dict)

    def skip(self, reason: str) -> None:
        """Record one skipped eviction attempt."""
        self.eviction_skipped_count_by_reason[reason] = (
            self.eviction_skipped_count_by_reason.get(reason, 0) + 1
        )

    def record_delta_payload_attempt(self) -> None:
        """Record one attempt to build a live delta eviction payload."""
        self.delta_payload_attempt_count += 1

    def record_delta_payload_success(self) -> None:
        """Record one successful live delta eviction payload build."""
        self.delta_payload_success_count += 1

    def record_delta_payload_fallback(self, reason: str) -> None:
        """Record one live delta ineligibility reason before anchor fallback."""
        self.delta_payload_fallback_count += 1
        self.delta_payload_fallback_count_by_reason[reason] = (
            self.delta_payload_fallback_count_by_reason.get(reason, 0) + 1
        )

    def record_rematerialization(
        self,
        *,
        node_id: int,
        phase: str,
        cache_hit: bool,
        elapsed_s: float,
    ) -> None:
        """Record one compact-state resolution under its diagnostic phase."""
        self.rematerialization_count_by_phase[phase] = (
            self.rematerialization_count_by_phase.get(phase, 0) + 1
        )
        self.rematerialization_total_s_by_phase[phase] = (
            self.rematerialization_total_s_by_phase.get(phase, 0.0) + elapsed_s
        )
        self.rematerialization_count_by_node_id[node_id] = (
            self.rematerialization_count_by_node_id.get(node_id, 0) + 1
        )
        if cache_hit:
            self.rematerialization_cache_hit_by_phase[phase] = (
                self.rematerialization_cache_hit_by_phase.get(phase, 0) + 1
            )
        else:
            self.rematerialization_cache_miss_by_phase[phase] = (
                self.rematerialization_cache_miss_by_phase.get(phase, 0) + 1
            )

    def snapshot(self) -> dict[str, object]:
        """Return a grep-friendly diagnostics payload."""
        average_reconstruction_depth = (
            0.0
            if self.rematerialization_count <= 0
            else self.total_reconstruction_depth / self.rematerialization_count
        )
        skipped_count = sum(self.eviction_skipped_count_by_reason.values())
        select_rematerialization_count_by_subphase = _phase_delta(
            {},
            self.rematerialization_count_by_phase,
            prefix="select",
        )
        select_rematerialization_cache_miss_by_subphase = _phase_delta(
            {},
            self.rematerialization_cache_miss_by_phase,
            prefix="select",
        )
        select_rematerialization_total_s_by_subphase = _phase_delta(
            {},
            self.rematerialization_total_s_by_phase,
            prefix="select",
        )
        return {
            "state_eviction_policy": self.state_eviction_policy,
            "state_eviction_payload_mode": self.state_eviction_payload_mode,
            "state_eviction_delta_chain_max_depth": (
                self.state_eviction_delta_chain_max_depth
            ),
            "eviction_attempt_count": self.eviction_attempt_count,
            "eviction_success_count": self.eviction_success_count,
            "eviction_skipped_count": skipped_count,
            "eviction_skipped_count_by_reason": dict(
                sorted(self.eviction_skipped_count_by_reason.items())
            ),
            "delta_payload_attempt_count": self.delta_payload_attempt_count,
            "delta_payload_success_count": self.delta_payload_success_count,
            "delta_payload_fallback_count": self.delta_payload_fallback_count,
            "delta_payload_fallback_count_by_reason": dict(
                sorted(self.delta_payload_fallback_count_by_reason.items())
            ),
            "evicted_materialized_state_count": self.evicted_materialized_state_count,
            "compact_payload_count": self.compact_payload_count,
            "anchor_payload_count": self.anchor_payload_count,
            "delta_payload_count": self.delta_payload_count,
            "rematerialization_count": self.rematerialization_count,
            "rematerialization_cache_hit": self.rematerialization_cache_hit,
            "rematerialization_cache_miss": self.rematerialization_cache_miss,
            "rematerialization_count_by_phase": dict(
                sorted(self.rematerialization_count_by_phase.items())
            ),
            "rematerialization_cache_hit_by_phase": dict(
                sorted(self.rematerialization_cache_hit_by_phase.items())
            ),
            "rematerialization_cache_miss_by_phase": dict(
                sorted(self.rematerialization_cache_miss_by_phase.items())
            ),
            "rematerialization_total_s_by_phase": dict(
                sorted(self.rematerialization_total_s_by_phase.items())
            ),
            "select_rematerialization_count_by_subphase": (
                select_rematerialization_count_by_subphase
            ),
            "select_rematerialization_cache_miss_by_subphase": (
                select_rematerialization_cache_miss_by_subphase
            ),
            "select_rematerialization_total_s_by_subphase": (
                select_rematerialization_total_s_by_subphase
            ),
            "top_rematerialized_node_ids": tuple(
                node_id
                for node_id, _count in sorted(
                    self.rematerialization_count_by_node_id.items(),
                    key=lambda item: (-item[1], item[0]),
                )[:10]
            ),
            "current_cache_size": self.current_cache_size,
            "cache_evictions": self.cache_evictions,
            "average_reconstruction_depth": average_reconstruction_depth,
            "eviction_scan_count": self.eviction_scan_count,
            "eviction_nodes_scanned_count": self.eviction_nodes_scanned_count,
            "eviction_total_s": self.eviction_total_s,
            "eviction_payload_build_s": self.eviction_payload_build_s,
            "rematerialization_total_s": self.rematerialization_total_s,
        }


@dataclass(slots=True)
class _LiveCompactStateResolver:
    """In-RAM compact state resolver for experimental growth eviction."""

    state_codec: object
    state_payloads_by_node_id: dict[int, CheckpointNodeStatePayload] = field(
        default_factory=dict
    )
    payload_chain_depth_by_node_id: dict[int, int] = field(default_factory=dict)
    metrics: MorpionGrowthStateEvictionMetrics = field(
        default_factory=MorpionGrowthStateEvictionMetrics
    )
    cache_size: int = 10000
    current_phase: str = "unknown"
    _decoded_state_cache: OrderedDict[int, MorpionState] = field(
        default_factory=OrderedDict
    )
    _resolving_node_ids: set[int] = field(default_factory=set)

    @contextmanager
    def phase(self, phase: str) -> Iterator[None]:
        """Temporarily attribute rematerializations to ``phase``."""
        previous_phase = self.current_phase
        self.current_phase = phase
        try:
            yield
        finally:
            self.current_phase = previous_phase

    def resolve(self, node_id: int) -> MorpionState:
        """Resolve one compact payload into a concrete Morpion state."""
        started_at = perf_counter()
        phase = self.current_phase
        self.metrics.rematerialization_count += 1
        cached_state = self._decoded_state_cache.get(node_id)
        if cached_state is not None:
            self.metrics.rematerialization_cache_hit += 1
            self._decoded_state_cache.move_to_end(node_id)
            self.metrics.current_cache_size = len(self._decoded_state_cache)
            elapsed_s = perf_counter() - started_at
            self.metrics.rematerialization_total_s += elapsed_s
            self.metrics.record_rematerialization(
                node_id=node_id,
                phase=phase,
                cache_hit=True,
                elapsed_s=elapsed_s,
            )
            return cached_state
        self.metrics.rematerialization_cache_miss += 1
        if node_id in self._resolving_node_ids:
            raise _live_compact_state_payload_cycle_error(node_id)
        self._resolving_node_ids.add(node_id)
        try:
            state = self._resolve_uncached(node_id, depth=1)
        finally:
            self._resolving_node_ids.discard(node_id)
            elapsed_s = perf_counter() - started_at
            self.metrics.rematerialization_total_s += elapsed_s
            self.metrics.record_rematerialization(
                node_id=node_id,
                phase=phase,
                cache_hit=False,
                elapsed_s=elapsed_s,
            )
        self._cache_state(node_id, state)
        return state

    def _resolve_uncached(self, node_id: int, *, depth: int) -> MorpionState:
        """Resolve one payload without checking the top-level decoded cache."""
        self.metrics.total_reconstruction_depth += depth
        payload = self.state_payloads_by_node_id[node_id]
        if isinstance(payload, AnchorCheckpointStatePayload):
            return cast(
                "MorpionState",
                cast("Any", self.state_codec).load_anchor_ref(payload.anchor_ref),
            )
        if isinstance(payload, DeltaCheckpointStatePayload):
            parent_state = self.resolve(payload.state_parent_node_id)
            return cast(
                "MorpionState",
                cast("Any", self.state_codec).load_child_from_delta(
                    parent_state=parent_state,
                    delta_ref=payload.delta_ref,
                    branch_from_parent=None,
                ),
            )
        raise KeyError(node_id)

    def store_payload(
        self,
        *,
        node_id: int,
        payload: CheckpointNodeStatePayload,
        chain_depth: int,
    ) -> None:
        """Store one live compact payload and its bounded delta-chain depth."""
        self.state_payloads_by_node_id[node_id] = payload
        self.payload_chain_depth_by_node_id[node_id] = chain_depth

    def payload_for_node_id_or_none(
        self,
        node_id: int,
    ) -> CheckpointNodeStatePayload | None:
        """Return one compact payload without resolving concrete state."""
        return self.state_payloads_by_node_id.get(node_id)

    def _cache_state(self, node_id: int, state: MorpionState) -> None:
        """Store one decoded state in the bounded LRU cache."""
        if self.cache_size <= 0:
            self.metrics.current_cache_size = 0
            return
        self._decoded_state_cache[node_id] = state
        self._decoded_state_cache.move_to_end(node_id)
        while len(self._decoded_state_cache) > self.cache_size:
            self._decoded_state_cache.popitem(last=False)
            self.metrics.cache_evictions += 1
        self.metrics.current_cache_size = len(self._decoded_state_cache)

    def summary(self, node_id: int) -> object | None:
        """Return optional state summary metadata for ``node_id``."""
        return self.state_payloads_by_node_id[node_id].state_summary


def _effective_growth_state_eviction_policy(policy: str) -> str:
    """Return the runtime semantics for a configured eviction policy."""
    if policy == "expanded":
        return "cold_expanded"
    return policy


def _single_parent_link_for_live_delta(node: object) -> _ParentDeltaContext | None:
    """Return the sole parent edge for live deltas, or ``None`` if ambiguous."""
    iter_parent_items = getattr(node, "iter_parent_items", None)
    if callable(iter_parent_items):
        try:
            parent_items = tuple(iter_parent_items())
        except Exception:  # pylint: disable=broad-exception-caught
            return None
    else:
        parent_nodes = getattr(node, "parent_nodes", None)
        if not isinstance(parent_nodes, Mapping):
            return None
        parent_items = tuple(parent_nodes.items())
    if len(parent_items) != 1:
        return None
    parent_node, branch_keys = parent_items[0]
    if not isinstance(branch_keys, set) or len(branch_keys) != 1:
        return None
    parent_node_id = getattr(parent_node, "id", None)
    if not isinstance(parent_node_id, int):
        return None
    return _ParentDeltaContext(
        parent_node=parent_node,
        parent_node_id=parent_node_id,
        branch_from_parent=next(iter(branch_keys)),
    )


def _dump_live_state_parent_branch_for_checkpoint(
    *,
    state_codec: object,
    branch_from_parent: object,
) -> object | None:
    """Return the optional compact branch payload stored beside a live delta."""
    hook = getattr(state_codec, "dump_state_parent_branch_for_checkpoint", None)
    if callable(hook):
        return cast("object | None", hook(branch_from_parent))
    return branch_from_parent

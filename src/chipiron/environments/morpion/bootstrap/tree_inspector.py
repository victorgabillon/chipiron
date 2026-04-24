"""Read-only tree and state inspection helpers for the Morpion bootstrap dashboard."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import time

from anemone.checkpoints import (
    AlgorithmNodeCheckpointPayload,
    AnchorCheckpointStatePayload,
    DeltaCheckpointStatePayload,
    NodeEvaluationCheckpointPayload,
    SearchRuntimeCheckpointPayload,
    SerializedValuePayload,
    deserialize_checkpoint_atom,
)
from atomheart.games.morpion import MorpionStateCheckpointCodec

from chipiron.environments.morpion.types import MorpionDynamics, MorpionState

from .anemone_runner import (
    InvalidMorpionSearchCheckpointError,
    load_morpion_search_checkpoint_payload,
)
from .bootstrap_loop import RUNTIME_CHECKPOINT_METADATA_KEY, MorpionBootstrapPaths
from .run_state import load_bootstrap_run_state

LOGGER = logging.getLogger(__name__)
TREE_INSPECTOR_TIMING_PREFIX = "[tree-inspector-timing]"
TREE_INSPECTOR_CACHE_PREFIX = "[tree-inspector-cache]"
_INDEXED_CHECKPOINT_TREE_CACHE: dict[tuple[str, int], "_IndexedCheckpointTree"] = {}


@dataclass(frozen=True, slots=True)
class MorpionBootstrapChildSummary:
    """Dashboard-facing summary for one outgoing branch from the selected node."""

    branch_label: str
    child_node_id: str | None
    visit_count: int | None
    is_terminal: bool | None
    is_exact: bool | None
    direct_value_scalar: float | None
    backed_up_value_scalar: float | None
    display_value_scalar: float | None


@dataclass(frozen=True, slots=True)
class MorpionBootstrapNodeSummary:
    """Dashboard-facing summary for one selected checkpoint node."""

    node_id: str
    depth: int | None
    parent_ids: tuple[str, ...]
    child_ids: tuple[str, ...]
    num_children: int
    visit_count: int | None
    is_terminal: bool | None
    is_exact: bool | None
    direct_value_scalar: float | None
    backed_up_value_scalar: float | None
    best_child_id: str | None
    best_branch_label: str | None


@dataclass(frozen=True, slots=True)
class MorpionBootstrapStateView:
    """Decoded Morpion state and board render for one selected node."""

    node_id: str
    variant: str
    moves: int
    point_count: int
    total_points: int
    is_terminal: bool
    board_text: str
    board_svg: str
    board_click_targets: tuple["MorpionBootstrapBoardClickTarget", ...]
    board_click_radius: float
    board_render_size: int


@dataclass(frozen=True, slots=True)
class MorpionBootstrapBoardClickTarget:
    """One clickable board target expressed in rendered SVG coordinates."""

    action_name: str
    center_x: float
    center_y: float


@dataclass(frozen=True, slots=True)
class MorpionBootstrapLocalTreeView:
    """Bounded local neighborhood view around the selected node."""

    root_node_id: str
    selected_node_id: str
    parent_node_ids: tuple[str, ...]
    sibling_node_ids: tuple[str, ...]
    child_node_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MorpionBootstrapTreeInspectorSnapshot:
    """Complete dashboard-facing snapshot for one selected tree node."""

    checkpoint_path: Path | None
    checkpoint_source: str | None
    root_node_id: str | None
    selected_node_id: str | None
    status_message: str | None
    error_message: str | None
    selection_warning: str | None
    node_summary: MorpionBootstrapNodeSummary | None
    child_summaries: tuple[MorpionBootstrapChildSummary, ...]
    state_view: MorpionBootstrapStateView | None
    local_tree_view: MorpionBootstrapLocalTreeView | None


@dataclass(frozen=True, slots=True)
class _ResolvedCheckpointReference:
    """Resolved runtime checkpoint path and source for inspector loading."""

    checkpoint_path: Path | None
    checkpoint_source: str | None
    status_message: str | None = None


@dataclass(frozen=True, slots=True)
class _IndexedChildLink:
    """Indexed branch edge from one node to one expanded child."""

    branch_key: object
    child_node_id: int


@dataclass(frozen=True, slots=True)
class _IndexedCheckpointTree:
    """Read-only indexed checkpoint state for local dashboard navigation."""

    root_node_id: int
    nodes_by_id: dict[int, AlgorithmNodeCheckpointPayload]
    parent_ids_by_node_id: dict[int, tuple[int, ...]]
    child_links_by_node_id: dict[int, tuple[_IndexedChildLink, ...]]


@dataclass(frozen=True, slots=True)
class _SelectedNodeSnapshotParts:
    """Cached selected-node inspector payload for one checkpoint node."""

    node_summary: MorpionBootstrapNodeSummary
    child_summaries: tuple[MorpionBootstrapChildSummary, ...]
    local_tree_view: MorpionBootstrapLocalTreeView | None
    state_view: MorpionBootstrapStateView | None


def build_morpion_bootstrap_tree_inspector_snapshot(
    work_dir: str | Path,
    *,
    selected_node_id: str | None = None,
    include_child_summaries: bool = True,
    include_local_tree_view: bool = True,
    include_state_view: bool = True,
) -> MorpionBootstrapTreeInspectorSnapshot:
    """Build one bounded dashboard snapshot from the latest runtime checkpoint."""
    snapshot_start_time = time.perf_counter()
    paths = MorpionBootstrapPaths.from_work_dir(work_dir)
    resolved_checkpoint = resolve_latest_runtime_checkpoint(paths)
    if resolved_checkpoint.checkpoint_path is None:
        return MorpionBootstrapTreeInspectorSnapshot(
            checkpoint_path=None,
            checkpoint_source=None,
            root_node_id=None,
            selected_node_id=None,
            status_message=resolved_checkpoint.status_message,
            error_message=None,
            selection_warning=None,
            node_summary=None,
            child_summaries=(),
            state_view=None,
            local_tree_view=None,
        )

    try:
        load_index_start_time = time.perf_counter()
        indexed_checkpoint = _load_indexed_checkpoint_tree_cached(
            checkpoint_path=resolved_checkpoint.checkpoint_path,
        )
        load_index_duration = time.perf_counter() - load_index_start_time
        print(
            f"{TREE_INSPECTOR_TIMING_PREFIX} load_indexed_checkpoint_tree_call "
            f"checkpoint={resolved_checkpoint.checkpoint_path.name} total_s={load_index_duration:.6f}",
            flush=True,
        )
    except InvalidMorpionSearchCheckpointError as exc:
        return MorpionBootstrapTreeInspectorSnapshot(
            checkpoint_path=resolved_checkpoint.checkpoint_path,
            checkpoint_source=resolved_checkpoint.checkpoint_source,
            root_node_id=None,
            selected_node_id=None,
            status_message=resolved_checkpoint.status_message,
            error_message=str(exc),
            selection_warning=None,
            node_summary=None,
            child_summaries=(),
            state_view=None,
            local_tree_view=None,
        )

    resolved_selected_node_id, selection_warning = _resolve_selected_node_id(
        indexed_checkpoint=indexed_checkpoint,
        selected_node_id=selected_node_id,
    )
    selected_parts_start_time = time.perf_counter()
    selected_node_parts = _build_selected_node_snapshot_parts(
        checkpoint_path=resolved_checkpoint.checkpoint_path,
        checkpoint_mtime_ns=_checkpoint_mtime_ns(resolved_checkpoint.checkpoint_path),
        selected_node_id=resolved_selected_node_id,
        include_child_summaries=include_child_summaries,
        include_local_tree_view=include_local_tree_view,
        include_state_view=include_state_view,
    )
    selected_parts_duration = time.perf_counter() - selected_parts_start_time

    snapshot_result = MorpionBootstrapTreeInspectorSnapshot(
        checkpoint_path=resolved_checkpoint.checkpoint_path,
        checkpoint_source=resolved_checkpoint.checkpoint_source,
        root_node_id=str(indexed_checkpoint.root_node_id),
        selected_node_id=str(resolved_selected_node_id),
        status_message=resolved_checkpoint.status_message,
        error_message=None,
        selection_warning=selection_warning,
        node_summary=selected_node_parts.node_summary,
        child_summaries=selected_node_parts.child_summaries,
        state_view=selected_node_parts.state_view,
        local_tree_view=selected_node_parts.local_tree_view,
    )
    snapshot_duration = time.perf_counter() - snapshot_start_time
    print(
        f"{TREE_INSPECTOR_TIMING_PREFIX} build_morpion_bootstrap_tree_inspector_snapshot "
        f"checkpoint={resolved_checkpoint.checkpoint_path.name} "
        f"selected_node_id={resolved_selected_node_id} "
        f"selected_parts_s={selected_parts_duration:.6f} total_s={snapshot_duration:.6f}",
        flush=True,
    )
    return snapshot_result


def resolve_latest_runtime_checkpoint(
    paths: MorpionBootstrapPaths,
) -> _ResolvedCheckpointReference:
    """Resolve the latest usable runtime checkpoint path for one work dir."""
    metadata_warning: str | None = None
    if paths.run_state_path.is_file():
        run_state = load_bootstrap_run_state(paths.run_state_path)
        dedicated_checkpoint_path = paths.resolve_work_dir_path(
            run_state.latest_runtime_checkpoint_path
        )
        if (
            dedicated_checkpoint_path is not None
            and dedicated_checkpoint_path.is_file()
        ):
            return _ResolvedCheckpointReference(
                checkpoint_path=dedicated_checkpoint_path,
                checkpoint_source="run_state_latest_runtime_checkpoint_path",
            )
        metadata_path = run_state.metadata.get(RUNTIME_CHECKPOINT_METADATA_KEY)
        if isinstance(metadata_path, str):
            resolved_path = paths.resolve_work_dir_path(metadata_path)
            if resolved_path is not None and resolved_path.is_file():
                return _ResolvedCheckpointReference(
                    checkpoint_path=resolved_path,
                    checkpoint_source="run_state_metadata",
                )
            metadata_warning = (
                "Run state points to a runtime checkpoint that is not currently "
                "available; falling back to the latest checkpoint file on disk if one exists."
            )

    checkpoint_candidates = sorted(paths.runtime_checkpoint_dir.glob("generation_*.json"))
    if checkpoint_candidates:
        fallback_message = metadata_warning
        if fallback_message is None:
            fallback_message = (
                "Using the latest runtime checkpoint file discovered on disk."
            )
        if metadata_warning is not None:
            LOGGER.warning(
                "[dashboard] latest_runtime_checkpoint_missing fallback_path=%s",
                str(checkpoint_candidates[-1]),
            )
        return _ResolvedCheckpointReference(
            checkpoint_path=checkpoint_candidates[-1],
            checkpoint_source="runtime_checkpoint_dir",
            status_message=fallback_message,
        )

    if metadata_warning is not None:
        return _ResolvedCheckpointReference(
            checkpoint_path=None,
            checkpoint_source=None,
            status_message=metadata_warning,
        )

    return _ResolvedCheckpointReference(
        checkpoint_path=None,
        checkpoint_source=None,
        status_message="No persisted runtime checkpoint available yet.",
    )


def _checkpoint_mtime_ns(checkpoint_path: Path) -> int:
    """Return one stable checkpoint freshness token for cache invalidation."""
    try:
        return checkpoint_path.stat().st_mtime_ns
    except OSError:
        return 0


def _load_indexed_checkpoint_tree(
    checkpoint_path: Path,
) -> _IndexedCheckpointTree:
    """Load and index one runtime checkpoint for bounded local inspection."""
    load_start_time = time.perf_counter()
    payload = load_morpion_search_checkpoint_payload(checkpoint_path)
    indexed_tree = _index_checkpoint_payload(payload)
    load_duration = time.perf_counter() - load_start_time
    print(
        f"{TREE_INSPECTOR_TIMING_PREFIX} _load_indexed_checkpoint_tree "
        f"checkpoint={checkpoint_path.name} total_s={load_duration:.6f}",
        flush=True,
    )
    return indexed_tree


def _load_indexed_checkpoint_tree_cached(
    checkpoint_path: Path,
) -> _IndexedCheckpointTree:
    """Return the indexed checkpoint tree cached by path and file freshness."""
    checkpoint_path = checkpoint_path.resolve()
    checkpoint_mtime_ns = _checkpoint_mtime_ns(checkpoint_path)
    cache_key = (str(checkpoint_path), checkpoint_mtime_ns)
    cached_tree = _INDEXED_CHECKPOINT_TREE_CACHE.get(cache_key)
    if cached_tree is not None:
        print(
            f"{TREE_INSPECTOR_CACHE_PREFIX} indexed_checkpoint hit "
            f"checkpoint={checkpoint_path.name}",
            flush=True,
        )
        return cached_tree

    print(
        f"{TREE_INSPECTOR_CACHE_PREFIX} indexed_checkpoint miss "
        f"checkpoint={checkpoint_path.name}",
        flush=True,
    )
    indexed_tree = _load_indexed_checkpoint_tree(checkpoint_path)
    _INDEXED_CHECKPOINT_TREE_CACHE.clear()
    _INDEXED_CHECKPOINT_TREE_CACHE[cache_key] = indexed_tree
    return indexed_tree


@lru_cache(maxsize=128)
def _build_selected_node_snapshot_parts(
    *,
    checkpoint_path: Path,
    checkpoint_mtime_ns: int,
    selected_node_id: int,
    include_child_summaries: bool,
    include_local_tree_view: bool,
    include_state_view: bool,
) -> _SelectedNodeSnapshotParts:
    """Build and cache the selected-node inspector payload for one checkpoint."""
    _ = checkpoint_mtime_ns
    snapshot_parts_start_time = time.perf_counter()
    indexed_checkpoint = _load_indexed_checkpoint_tree_cached(checkpoint_path)
    node_payload = indexed_checkpoint.nodes_by_id[selected_node_id]
    decoded_states_by_node_id: dict[int, MorpionState] = {}
    decode_start_time = time.perf_counter()
    state = _decode_node_state(
        node_payload,
        indexed_checkpoint=indexed_checkpoint,
        decoded_states_by_node_id=decoded_states_by_node_id,
    )
    decode_duration = time.perf_counter() - decode_start_time

    node_summary_start_time = time.perf_counter()
    node_summary = _build_node_summary(
        indexed_checkpoint=indexed_checkpoint,
        node_payload=node_payload,
        state=state,
    )
    node_summary_duration = time.perf_counter() - node_summary_start_time
    child_summaries_duration = 0.0
    child_summaries = (
        (
            lambda start_time: (
                _build_child_summaries(
                    indexed_checkpoint=indexed_checkpoint,
                    node_payload=node_payload,
                    state=state,
                    decoded_states_by_node_id=decoded_states_by_node_id,
                ),
                time.perf_counter() - start_time,
            )
        )(time.perf_counter())
        if include_child_summaries
        else ((), 0.0)
    )
    child_summaries, child_summaries_duration = child_summaries
    local_tree_view_duration = 0.0
    local_tree_view = (
        (
            lambda start_time: (
                _build_local_tree_view(
                    indexed_checkpoint=indexed_checkpoint,
                    selected_node_id=selected_node_id,
                ),
                time.perf_counter() - start_time,
            )
        )(time.perf_counter())
        if include_local_tree_view
        else (None, 0.0)
    )
    local_tree_view, local_tree_view_duration = local_tree_view
    state_view_duration = 0.0
    state_view = (
        (
            lambda start_time: (
                _build_state_view(
                    node_id=selected_node_id,
                    state=state,
                ),
                time.perf_counter() - start_time,
            )
        )(time.perf_counter())
        if include_state_view
        else (None, 0.0)
    )
    state_view, state_view_duration = state_view
    total_duration = time.perf_counter() - snapshot_parts_start_time
    print(
        f"{TREE_INSPECTOR_TIMING_PREFIX} _build_selected_node_snapshot_parts "
        f"checkpoint={checkpoint_path.name} selected_node_id={selected_node_id} "
        f"decode_s={decode_duration:.6f} node_summary_s={node_summary_duration:.6f} "
        f"child_summaries_s={child_summaries_duration:.6f} "
        f"local_tree_view_s={local_tree_view_duration:.6f} "
        f"state_view_s={state_view_duration:.6f} total_s={total_duration:.6f}",
        flush=True,
    )
    return _SelectedNodeSnapshotParts(
        node_summary=node_summary,
        child_summaries=child_summaries,
        local_tree_view=local_tree_view,
        state_view=state_view,
    )


def _index_checkpoint_payload(
    payload: SearchRuntimeCheckpointPayload,
) -> _IndexedCheckpointTree:
    """Index checkpoint nodes and reverse edges for navigation."""
    nodes_by_id = {
        node_payload.node_id: node_payload for node_payload in payload.tree.nodes
    }
    parent_ids_by_node_id = {node_id: [] for node_id in nodes_by_id}
    child_links_by_node_id: dict[int, tuple[_IndexedChildLink, ...]] = {}

    for node_payload in payload.tree.nodes:
        child_links: list[_IndexedChildLink] = []
        for linked_child in node_payload.linked_children:
            parent_ids_by_node_id.setdefault(linked_child.child_node_id, []).append(
                node_payload.node_id
            )
            child_links.append(
                _IndexedChildLink(
                    branch_key=deserialize_checkpoint_atom(linked_child.branch_key),
                    child_node_id=linked_child.child_node_id,
                )
            )
        child_links_by_node_id[node_payload.node_id] = tuple(child_links)

    return _IndexedCheckpointTree(
        root_node_id=payload.tree.root_node_id,
        nodes_by_id=nodes_by_id,
        parent_ids_by_node_id={
            node_id: tuple(sorted(parent_ids))
            for node_id, parent_ids in parent_ids_by_node_id.items()
        },
        child_links_by_node_id=child_links_by_node_id,
    )


def _resolve_selected_node_id(
    *,
    indexed_checkpoint: _IndexedCheckpointTree,
    selected_node_id: str | None,
) -> tuple[int, str | None]:
    """Resolve the requested selected node id or fall back to the root."""
    if selected_node_id is None:
        return indexed_checkpoint.root_node_id, None
    try:
        requested_node_id = int(selected_node_id)
    except ValueError:
        return (
            indexed_checkpoint.root_node_id,
            f"Selected node id {selected_node_id!r} is invalid; showing the root node instead.",
        )
    if requested_node_id in indexed_checkpoint.nodes_by_id:
        return requested_node_id, None
    return (
        indexed_checkpoint.root_node_id,
        f"Selected node id {selected_node_id!r} is not present in the current checkpoint; showing the root node instead.",
    )


def _build_node_summary(
    *,
    indexed_checkpoint: _IndexedCheckpointTree,
    node_payload: AlgorithmNodeCheckpointPayload,
    state: MorpionState,
) -> MorpionBootstrapNodeSummary:
    """Build one selected-node summary from the indexed checkpoint."""
    direct_value = _value_score(_direct_value_payload(node_payload.evaluation))
    backed_up_value = _value_score(_backed_up_value_payload(node_payload.evaluation))
    best_branch_label = _best_branch_label(state, node_payload)
    best_child_id = _best_child_id(
        indexed_checkpoint=indexed_checkpoint,
        node_id=node_payload.node_id,
        state=state,
        best_branch_label=best_branch_label,
    )
    child_ids = tuple(
        str(child_link.child_node_id)
        for child_link in indexed_checkpoint.child_links_by_node_id[node_payload.node_id]
    )
    return MorpionBootstrapNodeSummary(
        node_id=str(node_payload.node_id),
        depth=node_payload.depth,
        parent_ids=tuple(
            str(parent_id)
            for parent_id in indexed_checkpoint.parent_ids_by_node_id.get(
                node_payload.node_id,
                (),
            )
        ),
        child_ids=child_ids,
        num_children=len(child_ids),
        visit_count=None,
        is_terminal=state.is_terminal,
        is_exact=_node_is_exact(state, node_payload.evaluation),
        direct_value_scalar=direct_value,
        backed_up_value_scalar=backed_up_value,
        best_child_id=best_child_id,
        best_branch_label=best_branch_label,
    )


def _build_child_summaries(
    *,
    indexed_checkpoint: _IndexedCheckpointTree,
    node_payload: AlgorithmNodeCheckpointPayload,
    state: MorpionState,
    decoded_states_by_node_id: dict[int, MorpionState],
) -> tuple[MorpionBootstrapChildSummary, ...]:
    """Build one summary row per symmetry-unique legal action from the selected node."""
    child_summaries_start_time = time.perf_counter()
    dynamics = MorpionDynamics()
    child_links = {
        _child_link_branch_label(state=state, child_link=child_link): child_link.child_node_id
        for child_link in indexed_checkpoint.child_links_by_node_id[node_payload.node_id]
    }
    child_summaries: list[MorpionBootstrapChildSummary] = []
    expanded_child_count = 0
    unique_actions = _symmetry_unique_actions(state)
    for action in unique_actions:
        branch_label = dynamics.action_name(state, action)
        child_node_id = child_links.get(branch_label)
        if child_node_id is None:
            child_summaries.append(
                MorpionBootstrapChildSummary(
                    branch_label=branch_label,
                    child_node_id=None,
                    visit_count=None,
                    is_terminal=None,
                    is_exact=None,
                    direct_value_scalar=None,
                    backed_up_value_scalar=None,
                    display_value_scalar=None,
                )
            )
            continue
        child_payload = indexed_checkpoint.nodes_by_id[child_node_id]
        expanded_child_count += 1
        child_state = _decode_node_state(
            child_payload,
            indexed_checkpoint=indexed_checkpoint,
            decoded_states_by_node_id=decoded_states_by_node_id,
        )
        direct_value = _value_score(_direct_value_payload(child_payload.evaluation))
        backed_up_value = _value_score(_backed_up_value_payload(child_payload.evaluation))
        child_summaries.append(
            MorpionBootstrapChildSummary(
                branch_label=branch_label,
                child_node_id=str(child_node_id),
                visit_count=None,
                is_terminal=child_state.is_terminal,
                is_exact=_node_is_exact(child_state, child_payload.evaluation),
                direct_value_scalar=direct_value,
                backed_up_value_scalar=backed_up_value,
                display_value_scalar=_display_value_scalar(
                    backed_up_value=backed_up_value,
                    direct_value=direct_value,
                ),
            )
        )
    total_duration = time.perf_counter() - child_summaries_start_time
    print(
        f"{TREE_INSPECTOR_TIMING_PREFIX} _build_child_summaries "
        f"selected_node_id={node_payload.node_id} unique_actions={len(unique_actions)} "
        f"expanded_children={expanded_child_count} total_s={total_duration:.6f}",
        flush=True,
    )
    return tuple(child_summaries)


def _build_local_tree_view(
    *,
    indexed_checkpoint: _IndexedCheckpointTree,
    selected_node_id: int,
) -> MorpionBootstrapLocalTreeView:
    """Build the bounded local tree neighborhood around the selected node."""
    parent_ids = indexed_checkpoint.parent_ids_by_node_id.get(selected_node_id, ())
    child_ids = tuple(
        str(child_link.child_node_id)
        for child_link in indexed_checkpoint.child_links_by_node_id[selected_node_id]
    )
    sibling_ids: tuple[str, ...] = ()
    if parent_ids:
        first_parent_id = parent_ids[0]
        sibling_ids = tuple(
            str(child_link.child_node_id)
            for child_link in indexed_checkpoint.child_links_by_node_id[first_parent_id]
            if child_link.child_node_id != selected_node_id
        )
    return MorpionBootstrapLocalTreeView(
        root_node_id=str(indexed_checkpoint.root_node_id),
        selected_node_id=str(selected_node_id),
        parent_node_ids=tuple(str(parent_id) for parent_id in parent_ids),
        sibling_node_ids=sibling_ids,
        child_node_ids=child_ids,
    )


def _build_state_view(
    *,
    node_id: int,
    state: MorpionState,
) -> MorpionBootstrapStateView:
    """Build one selected-state view with a rendered Morpion SVG board."""
    from chipiron.displays.morpion_svg_adapter import MorpionSvgAdapter
    from chipiron.environments.morpion.morpion_display import (
        build_morpion_display_payload,
    )

    state_view_start_time = time.perf_counter()
    dynamics = MorpionDynamics()
    svg_adapter = MorpionSvgAdapter()
    render_size = 720
    payload_start_time = time.perf_counter()
    adapter_payload = build_morpion_display_payload(
        state=state,
        dynamics=dynamics,
    )
    payload_duration = time.perf_counter() - payload_start_time
    position_start_time = time.perf_counter()
    position = svg_adapter.position_from_update(
        state_tag=state.tag,
        adapter_payload=adapter_payload,
    )
    position_duration = time.perf_counter() - position_start_time
    render_start_time = time.perf_counter()
    render_result = svg_adapter.render_svg(position, render_size, margin=8)
    render_duration = time.perf_counter() - render_start_time
    state_view = MorpionBootstrapStateView(
        node_id=str(node_id),
        variant=state.variant.value,
        moves=state.moves,
        point_count=len(state.points),
        total_points=len(state.points),
        is_terminal=state.is_terminal,
        board_text=state.pprint(),
        board_svg=render_result.svg_bytes.decode("utf-8"),
        board_click_targets=tuple(
            MorpionBootstrapBoardClickTarget(
                action_name=action_name,
                center_x=center_x,
                center_y=center_y,
            )
            for action_name, center_x, center_y in svg_adapter.click_targets_snapshot()
        ),
        board_click_radius=svg_adapter.click_radius,
        board_render_size=render_size,
    )
    total_duration = time.perf_counter() - state_view_start_time
    print(
        f"{TREE_INSPECTOR_TIMING_PREFIX} _build_state_view "
        f"selected_node_id={node_id} payload_s={payload_duration:.6f} "
        f"position_s={position_duration:.6f} render_svg_s={render_duration:.6f} "
        f"total_s={total_duration:.6f}",
        flush=True,
    )
    return state_view


def _decode_node_state(
    node_payload: AlgorithmNodeCheckpointPayload,
    *,
    indexed_checkpoint: _IndexedCheckpointTree,
    decoded_states_by_node_id: dict[int, MorpionState],
) -> MorpionState:
    """Decode one checkpoint node state on demand from Anemone state payloads."""
    decode_start_time = time.perf_counter()
    cached_state = decoded_states_by_node_id.get(node_payload.node_id)
    if cached_state is not None:
        return cached_state

    dynamics = MorpionDynamics()
    state_codec = MorpionStateCheckpointCodec()
    state_payload = node_payload.state_payload
    if isinstance(state_payload, AnchorCheckpointStatePayload):
        decoded_state = dynamics.wrap_atomheart_state(
            state_codec.load_anchor_ref(state_payload.anchor_ref)
        )
    elif isinstance(state_payload, DeltaCheckpointStatePayload):
        if node_payload.parent_node_id is None:
            raise InvalidMorpionSearchCheckpointError(
                Path("<in-memory>"),
                f"delta node {node_payload.node_id} is missing its parent node id",
            )
        parent_payload = indexed_checkpoint.nodes_by_id.get(node_payload.parent_node_id)
        if parent_payload is None:
            raise InvalidMorpionSearchCheckpointError(
                Path("<in-memory>"),
                f"delta node {node_payload.node_id} references unknown parent "
                f"{node_payload.parent_node_id}",
            )
        parent_state = _decode_node_state(
            parent_payload,
            indexed_checkpoint=indexed_checkpoint,
            decoded_states_by_node_id=decoded_states_by_node_id,
        )
        branch_from_parent = (
            None
            if node_payload.branch_from_parent is None
            else deserialize_checkpoint_atom(node_payload.branch_from_parent)
        )
        decoded_state = dynamics.wrap_atomheart_state(
            state_codec.load_child_from_delta(
                parent_state=parent_state.to_atomheart_state(),
                delta_ref=state_payload.delta_ref,
                branch_from_parent=branch_from_parent,
            )
        )
    else:
        raise InvalidMorpionSearchCheckpointError(
            Path("<in-memory>"),
            f"node {node_payload.node_id} has unsupported state payload "
            f"{type(state_payload).__name__}",
        )
    decoded_states_by_node_id[node_payload.node_id] = decoded_state
    decode_duration = time.perf_counter() - decode_start_time
    print(
        f"{TREE_INSPECTOR_TIMING_PREFIX} _decode_node_state "
        f"node_id={node_payload.node_id} total_s={decode_duration:.6f}",
        flush=True,
    )
    return decoded_state


def _symmetry_unique_actions(state: MorpionState) -> tuple[object, ...]:
    """Return one canonical representative per legal-action symmetry orbit."""
    dynamics = MorpionDynamics()
    return tuple(
        dynamics.canonical_action_in_state(state, orbit[0])
        for orbit in dynamics.legal_action_orbits(state)
        if orbit
    )


def _render_branch_label(state: MorpionState, branch_key: object) -> str:
    """Render one branch label for the current state or fall back safely."""
    dynamics = MorpionDynamics()
    try:
        return dynamics.action_name(state, branch_key)
    except Exception:
        return repr(branch_key)


def _child_link_branch_label(
    *,
    state: MorpionState,
    child_link: _IndexedChildLink,
) -> str:
    """Render one child-link branch label for the selected node only."""
    return _render_branch_label(state, child_link.branch_key)


def _direct_value_payload(
    evaluation: NodeEvaluationCheckpointPayload | None,
) -> SerializedValuePayload | None:
    """Return the direct serialized value payload when present."""
    if evaluation is None:
        return None
    return evaluation.direct_value


def _backed_up_value_payload(
    evaluation: NodeEvaluationCheckpointPayload | None,
) -> SerializedValuePayload | None:
    """Return the backed-up serialized value payload when present."""
    if evaluation is None:
        return None
    return evaluation.backed_up_value


def _value_score(value_payload: SerializedValuePayload | None) -> float | None:
    """Return the scalar score from one serialized value payload."""
    if value_payload is None:
        return None
    return value_payload.score


def _node_is_exact(
    state: MorpionState,
    evaluation: NodeEvaluationCheckpointPayload | None,
) -> bool | None:
    """Return whether the node has a non-estimate or terminal value."""
    if state.is_terminal:
        return True
    for value_payload in (_backed_up_value_payload(evaluation), _direct_value_payload(evaluation)):
        if value_payload is None:
            continue
        return value_payload.certainty != "ESTIMATE"
    return None


def _best_branch_label(
    state: MorpionState,
    node_payload: AlgorithmNodeCheckpointPayload,
) -> str | None:
    """Return the selected node's best known branch label when present."""
    evaluation = node_payload.evaluation
    if evaluation is None:
        return None
    best_branch_payload = None
    if evaluation.backup_runtime is not None:
        best_branch_payload = evaluation.backup_runtime.best_branch
    if (
        best_branch_payload is None
        and evaluation.principal_variation is not None
        and evaluation.principal_variation.best_branch_sequence
    ):
        best_branch_payload = evaluation.principal_variation.best_branch_sequence[0]
    if best_branch_payload is None:
        return None
    best_branch = deserialize_checkpoint_atom(best_branch_payload)
    return _render_branch_label(state, best_branch)


def _best_child_id(
    *,
    indexed_checkpoint: _IndexedCheckpointTree,
    node_id: int,
    state: MorpionState,
    best_branch_label: str | None,
) -> str | None:
    """Resolve the expanded child id for the best known branch when present."""
    if best_branch_label is None:
        return None
    for child_link in indexed_checkpoint.child_links_by_node_id[node_id]:
        if _child_link_branch_label(state=state, child_link=child_link) == best_branch_label:
            return str(child_link.child_node_id)
    return None


def _display_value_scalar(
    *,
    backed_up_value: float | None,
    direct_value: float | None,
) -> float | None:
    """Return the operator-facing display value with backed-up priority."""
    if backed_up_value is not None:
        return backed_up_value
    return direct_value


__all__ = [
    "MorpionBootstrapChildSummary",
    "MorpionBootstrapLocalTreeView",
    "MorpionBootstrapNodeSummary",
    "MorpionBootstrapStateView",
    "MorpionBootstrapTreeInspectorSnapshot",
    "build_morpion_bootstrap_tree_inspector_snapshot",
    "resolve_latest_runtime_checkpoint",
]

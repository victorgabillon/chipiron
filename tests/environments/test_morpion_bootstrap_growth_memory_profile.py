"""Tests for opt-in Morpion growth runtime memory profiling."""
# ruff: noqa: E402, D102, D105, D107, ANN201, ANN204, TRY003

from __future__ import annotations

import gc
import importlib
import logging
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHIPIRON_PACKAGE_ROOT = _REPO_ROOT / "src" / "chipiron"
_BOOTSTRAP_PACKAGE_ROOT = (
    _CHIPIRON_PACKAGE_ROOT / "environments" / "morpion" / "bootstrap"
)

if "chipiron" not in sys.modules:
    _chipiron_stub = ModuleType("chipiron")
    _chipiron_stub.__path__ = [str(_CHIPIRON_PACKAGE_ROOT)]
    sys.modules["chipiron"] = _chipiron_stub

if "chipiron.environments.morpion.bootstrap" not in sys.modules:
    _bootstrap_stub = ModuleType("chipiron.environments.morpion.bootstrap")
    _bootstrap_stub.__path__ = [str(_BOOTSTRAP_PACKAGE_ROOT)]
    sys.modules["chipiron.environments.morpion.bootstrap"] = _bootstrap_stub

from chipiron.environments.morpion.bootstrap.growth_memory_profile import (
    log_growth_runtime_memory_profile,
)
from chipiron.environments.morpion.bootstrap.recursive_memory_profile import (
    build_recursive_profile_context,
    checkpoint_payload_lifetime_histograms,
    checkpoint_state_histograms,
    deep_size,
    frozenset_ownership_histogram,
    gc_shallow_size_summary,
    linoo_deep_breakdown_histograms,
    linoo_node_state_slots_histogram,
    linoo_node_state_table_histogram,
    linoo_state_histograms,
    log_growth_recursive_memory_profile,
    node_evaluation_runtime_histograms,
    slot_names,
    tree_topology_histograms,
)
from anemone.checkpoints.state_handles import DenseCheckpointPayloadStore

recursive_memory_profile_module = importlib.import_module(
    "chipiron.environments.morpion.bootstrap.recursive_memory_profile"
)

if TYPE_CHECKING:
    from _pytest.logging import LogCaptureFixture


class FakeNode:
    """Small node-like object with the fields sampled by the profile helper."""

    def __init__(self, index: int) -> None:
        """Initialize a fake node with representative shallow fields."""
        self.metadata = {"index": index, "tag": "profile-test"}
        self.state_ref_payload = {"state": [index, index + 1]}
        self.children = [index * 2, index * 2 + 1]
        self.parents = [index - 1] if index > 0 else []
        self.legal_actions = [(index, index + 1)]
        self.value = float(index)
        self.is_terminal = index % 2 == 0
        self.is_exact = index % 3 == 0


class FakeAttrChild:
    """Nested object with a regular __dict__ for one-level profiling."""

    def __init__(self, index: int) -> None:
        self.score = float(index)
        self.tag = f"child-{index}"


class RaisingDictAttr:
    """Object whose __dict__ access raises to keep profiling failure-proof."""

    def __getattribute__(self, name: str):
        if name == "__dict__":
            raise RuntimeError("broken __dict__")
        return super().__getattribute__(name)


class FakeSlottedAttr:
    """Slotted nested object for slots-aware anatomy profiling."""

    __slots__ = ("a", "b")

    def __init__(self, index: int) -> None:
        self.a = index
        self.b = FakeAttrChild(index)


class RaisingSlotAttr:
    """Slotted object whose slot getter raises for failure-proof profiling."""

    __slots__ = ("a", "b")

    def __init__(self, index: int) -> None:
        self.a = index
        self.b = index + 1

    def __getattribute__(self, name: str):
        if name == "b":
            raise RuntimeError("broken slot getter")
        return super().__getattribute__(name)


class FakeNodeWithNestedAttrs:
    """Node-like object whose direct attrs should get one-level anatomy logs."""

    def __init__(self, index: int, *, broken_attr: bool = False) -> None:
        self.tree_node = FakeAttrChild(index)
        self.tree_evaluation = (
            RaisingDictAttr() if broken_attr else FakeAttrChild(index + 100)
        )
        self.is_terminal = False
        self.is_exact = False


class FakeNodeWithSlottedAttrs:
    """Node-like object whose direct attrs point to slotted objects."""

    def __init__(self, index: int, *, broken_slot: bool = False) -> None:
        self.tree_node = FakeSlottedAttr(index)
        self.tree_evaluation = (
            RaisingSlotAttr(index + 100)
            if broken_slot
            else FakeSlottedAttr(index + 100)
        )
        self.is_terminal = False
        self.is_exact = False


class FakeRunner:
    """Runner-like object exposing nodes through a simple attribute."""

    def __init__(self) -> None:
        """Initialize a runner with discoverable node and cache attributes."""
        self.nodes = [FakeNode(index) for index in range(5)]
        self.patch_cache = {"pending": [1, 2, 3]}


class FakeRunnerWithProfileIterator:
    """Runner-like object exposing a dedicated profiling iterator."""

    def __init__(self) -> None:
        self._nodes = [FakeNode(index) for index in range(4)]

    def iter_profile_nodes(self):
        return iter(self._nodes)


class FakeRunnerWithCheckpointStoreAndProfileIterator:
    """Runner exposing profile nodes and checkpoint payload stores."""

    def __init__(self) -> None:
        self._nodes = [FakeNode(index) for index in range(4)]
        self.profile_checkpoint_state_resolver = SimpleNamespace(
            owner=FakeCheckpointPayloadOwner()
        )

    def iter_profile_nodes(self):
        return iter(self._nodes)


class FakeRunnerWithDirectCheckpointStoreAndDangerousHandles:
    """Runner with a direct checkpoint store and handles that must not be scanned."""

    def __init__(self, *, node_count: int = 4) -> None:
        self._runtime = SimpleNamespace(
            state_codec=SimpleNamespace(owner=FakeCheckpointPayloadOwner())
        )
        self._nodes = [
            FakeAlgorithmNode(
                FakeTreeNode(state_handle=RaisingCheckpointHandle()),
                FakeNodeEvaluation(),
            )
            for _ in range(node_count)
        ]

    def iter_profile_nodes(self):
        return iter(self._nodes)


class FakeRunnerWithFallbackCheckpointHandles:
    """Runner with no direct store, forcing capped handle fallback discovery."""

    def __init__(self, *, node_count: int) -> None:
        self._nodes = [
            FakeAlgorithmNode(
                FakeTreeNode(state_handle=RaisingCheckpointHandle()),
                FakeNodeEvaluation(),
            )
            for _ in range(node_count)
        ]

    def iter_profile_nodes(self):
        return iter(self._nodes)


class FakeResolverWithPayloads:
    """Resolver-like object exposing payloads through explicit known paths."""

    def __init__(self) -> None:
        self.owner = FakeCheckpointPayloadOwner()


class RecursivelyDangerousRuntime:
    """Runtime-like object that must not be generically traversed."""

    def __init__(self, resolver: object | None = None) -> None:
        self.checkpoint_state_resolver = resolver

    def __iter__(self):
        raise AssertionError("runtime should not be recursively iterated")


class FakeRunnerWithPrivateRuntime:
    """Runner-like object exposing nodes only under a private runtime path."""

    def __init__(self) -> None:
        self._runtime = SimpleNamespace(
            node_store=SimpleNamespace(nodes=[FakeNode(index) for index in range(3)])
        )


class FakeRunnerWithNestedAttrs:
    """Runner-like object exposing nodes with nested anatomy attributes."""

    def __init__(self, *, broken_attr: bool = False) -> None:
        self.nodes = [
            FakeNodeWithNestedAttrs(index, broken_attr=broken_attr)
            for index in range(3)
        ]


class FakeRunnerWithSlottedAttrs:
    """Runner-like object exposing nodes with slotted nested attrs."""

    def __init__(self, *, broken_slot: bool = False) -> None:
        self.nodes = [
            FakeNodeWithSlottedAttrs(index, broken_slot=broken_slot)
            for index in range(3)
        ]


class RunnerWithoutNodes:
    """Runner-like object with no discoverable node store."""

    def __init__(self) -> None:
        """Initialize a runner without node-like attributes."""
        self.name = "no-nodes"


class RecursiveSlotObject:
    """Slotted object used by recursive-size tests."""

    __slots__ = ("child",)

    def __init__(self, child: object) -> None:
        self.child = child


class RecursiveDictObject:
    """Dict-backed object used by recursive-size tests."""

    def __init__(self, child: object) -> None:
        self.child = child


class RecursivePropertyObject:
    """Object whose property must never be touched by the profiler."""

    __slots__ = ("safe",)

    def __init__(self) -> None:
        self.safe = 1

    @property
    def dangerous(self) -> int:
        raise AssertionError("recursive profiler called a property")


class RecursionDictObject:
    """Object used to simulate recursion failures during attribute traversal."""

    def __init__(self) -> None:
        self.child = 1


class MorpionState:
    """Synthetic atomheart-like state used by frozenset ownership tests."""

    def __init__(
        self,
        *,
        points: frozenset[object],
        used_unit_segments: frozenset[object],
        played_moves: frozenset[object],
    ) -> None:
        self.points = points
        self.used_unit_segments = used_unit_segments
        self.played_moves = played_moves


MorpionState.__module__ = "atomheart.games.morpion.state"


class FakeCheckpointPayload:
    """Synthetic checkpoint payload owner used by referrer type tests."""

    def __init__(self, state_ref: object) -> None:
        self.state_ref = state_ref


class FakeTreeNode:
    """Small TreeNode-shaped object for recursive profile histograms."""

    __slots__ = (
        "branches_children_",
        "non_opened_branches_",
        "parent_nodes_",
        "state_handle_",
    )

    def __init__(
        self,
        *,
        branches_children: object | None = None,
        parent_nodes: object | None = None,
        state_handle: object | None = None,
    ) -> None:
        self.branches_children_ = branches_children
        self.non_opened_branches_ = None
        self.parent_nodes_ = parent_nodes if parent_nodes is not None else {}
        self.state_handle_ = state_handle


class FakeNodeEvaluation:
    """Small NodeMaxEvaluation-shaped object for recursive histograms."""

    __slots__ = (
        "_backed_up_value",
        "backup_runtime_",
        "branch_frontier_",
        "decision_ordering_",
        "direct_value",
        "pv_state_",
    )

    def __init__(self, *, runtime_state: object | None = None) -> None:
        self.direct_value = 1.0
        self._backed_up_value = None
        self.decision_ordering_ = runtime_state
        self.pv_state_ = None
        self.branch_frontier_ = None
        self.backup_runtime_ = None


class FakeAlgorithmNode:
    """Small AlgorithmNode-shaped object for recursive profile tests."""

    def __init__(self, tree_node: FakeTreeNode, node_eval: FakeNodeEvaluation) -> None:
        self.tree_node = tree_node
        self.tree_evaluation = node_eval
        self._state_representation = None


class RaisingCheckpointHandle:
    """Handle that proves checkpoint histogram scanning stopped before access."""

    def __getattribute__(self, name: str):
        if name in {"state_", "node_id", "resolver"}:
            raise AssertionError("checkpoint histogram scanned past the handle cap")
        return super().__getattribute__(name)


class FakeLinooNodeState:
    """Small sparse Linoo state for histogram tests."""

    __slots__ = ("depth", "node", "status")

    def __init__(self, node: object, *, status: str) -> None:
        self.node = node
        self.depth = 1
        self.status = status

    def is_default(self) -> bool:
        return self.status == "opened"


class FakeLinooSelector:
    """Small Linoo selector-shaped object for histogram tests."""

    def __init__(self, states: dict[int, FakeLinooNodeState]) -> None:
        self._node_state_by_id = states
        self.labels = {key: state.status for key, state in states.items()}
        self.active_status = "opened"
        self.extra_metadata = [len(states), "selector"]


class FakeComposedSelector:
    """Composed selector wrapper that hides the concrete Linoo selector."""

    __slots__ = ("base", "metadata")

    def __init__(self, selector: FakeLinooSelector) -> None:
        self.base = [{"selector": selector}]
        self.metadata = {"name": "composed"}


class FakeAnchorCheckpointStatePayload:
    """Fake anchor payload with the real suffix used by diagnostics."""

    def __init__(self, state: object) -> None:
        self.state = state


class FakeDeltaCheckpointStatePayload:
    """Fake delta payload with the real suffix used by diagnostics."""

    def __init__(self, delta: object) -> None:
        self.delta = delta


class FakeCheckpointPayloadOwner:
    """Object with a raw mapping attribute that owns checkpoint payloads."""

    __slots__ = ("_payloads_by_node_id",)

    def __init__(self) -> None:
        self._payloads_by_node_id = {
            1: FakeAnchorCheckpointStatePayload({"board": [1, 2, 3]}),
            2: FakeDeltaCheckpointStatePayload({"move": 4}),
            3: object(),
        }


class FakeCheckpointStateResolver:
    """Resolver-like object with checkpoint payloads and resolved-state cache."""

    __slots__ = ("_resolved_states", "owner", "state_payloads_by_node_id")

    def __init__(self) -> None:
        self.owner = FakeCheckpointPayloadOwner()
        self.state_payloads_by_node_id = self.owner._payloads_by_node_id
        self._resolved_states = {1: {"board": [9, 9, 9]}}


class FakeDenseCheckpointStateResolver:
    """Resolver-like object exposing a dense mapping-compatible payload store."""

    __slots__ = ("_resolved_states", "state_payloads_by_node_id")

    def __init__(self) -> None:
        self.state_payloads_by_node_id = DenseCheckpointPayloadStore(
            [
                FakeAnchorCheckpointStatePayload({"board": [1, 2, 3]}),
                FakeDeltaCheckpointStatePayload({"move": 4}),
            ]
        )
        self._resolved_states = {}


class FakeCheckpointBackedStateHandle:
    """Checkpoint-handle-shaped object whose get() must never be called."""

    def __init__(self, resolver: object, node_id: int) -> None:
        self.resolver = resolver
        self.node_id = node_id

    def get(self) -> object:
        raise AssertionError("checkpoint lifetime diagnostics must not materialize")


class FakeRunnerWithCheckpointBackedHandles:
    """Runner exposing one shared checkpoint resolver and lazy handles."""

    def __init__(self) -> None:
        resolver = FakeCheckpointStateResolver()
        self.profile_checkpoint_state_resolver = resolver
        self.nodes = [
            FakeAlgorithmNode(
                FakeTreeNode(
                    state_handle=FakeCheckpointBackedStateHandle(resolver, 1)
                ),
                FakeNodeEvaluation(),
            ),
            FakeAlgorithmNode(
                FakeTreeNode(
                    state_handle=FakeCheckpointBackedStateHandle(resolver, 2)
                ),
                FakeNodeEvaluation(),
            ),
        ]


class FakeRunnerWithCheckpointPayloadStore:
    """Runner whose checkpoint payloads sit under a nested runtime codec."""

    def __init__(self) -> None:
        self._runtime = SimpleNamespace(
            state_codec=SimpleNamespace(owner=FakeCheckpointPayloadOwner())
        )
        self.nodes = []


def test_growth_runtime_memory_profile_logs_node_sample(
    caplog: LogCaptureFixture,
) -> None:
    """Growth profiles should include density, GC, attrs, and node anatomy."""
    caplog.set_level(logging.INFO)

    log_growth_runtime_memory_profile(
        runner=FakeRunner(),
        generation=431,
        event="after_growth",
        node_count=5,
        branch_count=10,
        sample_nodes=3,
        top_n=5,
    )

    text = caplog.text
    assert "[growth-profile] event=after_growth generation=431" in text
    assert "node_count=5" in text
    assert "branch_count=10" in text
    assert "top_gc_types=" in text
    assert "top_project_gc_types=" in text
    assert "runner_attrs=" in text
    assert "node_sample" in text
    assert "sample_size=3" in text
    assert "avg_dict_len=" in text
    assert "top_node_attrs=" in text
    assert "avg_node_shallow_bytes=" in text
    assert "avg_state_payload_shallow_bytes=" in text
    assert "state_payload_fraction=" in text


def test_growth_runtime_memory_profile_prefers_profile_iterator(
    caplog: LogCaptureFixture,
) -> None:
    """Dedicated runner profiling iterators should win over fallback discovery."""
    caplog.set_level(logging.INFO)

    log_growth_runtime_memory_profile(
        runner=FakeRunnerWithProfileIterator(),
        generation=7,
        event="after_checkpoint_load",
        node_count=4,
        branch_count=None,
        sample_nodes=2,
        top_n=5,
    )

    text = caplog.text
    assert "node_sample source=iter_profile_nodes" in text
    assert "sample_size=2" in text


def test_growth_runtime_memory_profile_uses_private_runtime_fallback(
    caplog: LogCaptureFixture,
) -> None:
    """Private runtime node stores should still be discoverable for profiling."""
    caplog.set_level(logging.INFO)

    log_growth_runtime_memory_profile(
        runner=FakeRunnerWithPrivateRuntime(),
        generation=8,
        event="after_checkpoint_load",
        node_count=3,
        branch_count=None,
        sample_nodes=3,
        top_n=5,
    )

    text = caplog.text
    assert "node_sample source=_runtime.node_store.nodes" in text
    assert "sample_size=3" in text


def test_growth_runtime_memory_profile_logs_node_attr_sample(
    caplog: LogCaptureFixture,
) -> None:
    """One-level direct node attributes should get bounded anatomy logs."""
    caplog.set_level(logging.INFO)

    log_growth_runtime_memory_profile(
        runner=FakeRunnerWithNestedAttrs(),
        generation=11,
        event="after_checkpoint_load",
        node_count=3,
        branch_count=None,
        sample_nodes=3,
        top_n=5,
    )

    text = caplog.text
    assert "node_attr_sample source=nodes attr=tree_node" in text
    assert "node_attr_sample source=nodes attr=tree_evaluation" in text
    assert "avg_shallow_bytes=" in text
    assert "avg_dict_shallow_bytes=" in text
    assert "avg_dict_len=" in text
    assert "top_types=" in text
    assert "top_child_attrs=" in text


def test_growth_runtime_memory_profile_tolerates_broken_attr_getters(
    caplog: LogCaptureFixture,
) -> None:
    """Profiler instrumentation should not crash when nested attr access breaks."""
    caplog.set_level(logging.INFO)

    log_growth_runtime_memory_profile(
        runner=FakeRunnerWithNestedAttrs(broken_attr=True),
        generation=12,
        event="after_checkpoint_load",
        node_count=3,
        branch_count=None,
        sample_nodes=3,
        top_n=5,
    )

    text = caplog.text
    assert "[growth-profile] event=after_checkpoint_load node_sample" in text
    assert "node_attr_sample source=nodes attr=tree_evaluation" in text


def test_growth_runtime_memory_profile_logs_node_attr_slot_sample(
    caplog: LogCaptureFixture,
) -> None:
    """Slotted nested attrs should get bounded slot anatomy logs."""
    caplog.set_level(logging.INFO)

    log_growth_runtime_memory_profile(
        runner=FakeRunnerWithSlottedAttrs(),
        generation=13,
        event="after_checkpoint_load",
        node_count=3,
        branch_count=None,
        sample_nodes=3,
        top_n=5,
    )

    text = caplog.text
    assert "node_attr_sample source=nodes attr=tree_node" in text
    assert "avg_slots_count=" in text
    assert "top_slot_names=" in text
    assert "top_slot_value_types=" in text
    assert "node_attr_slot_sample source=nodes attr=tree_node slot=a" in text
    assert "node_attr_slot_sample source=nodes attr=tree_node slot=b" in text


def test_growth_runtime_memory_profile_tolerates_broken_slot_getters(
    caplog: LogCaptureFixture,
) -> None:
    """Slot getter failures should be logged without crashing profiling."""
    caplog.set_level(logging.INFO)

    log_growth_runtime_memory_profile(
        runner=FakeRunnerWithSlottedAttrs(broken_slot=True),
        generation=14,
        event="after_checkpoint_load",
        node_count=3,
        branch_count=None,
        sample_nodes=3,
        top_n=5,
    )

    text = caplog.text
    assert "[growth-profile] event=after_checkpoint_load node_sample" in text
    assert "node_attr_slot_sample source=nodes attr=tree_evaluation slot=b" in text


def test_growth_runtime_memory_profile_handles_missing_nodes(
    caplog: LogCaptureFixture,
) -> None:
    """Growth profiles should not crash when runner internals are unknown."""
    caplog.set_level(logging.INFO)

    log_growth_runtime_memory_profile(
        runner=RunnerWithoutNodes(),
        generation=3,
        event="after_checkpoint_load",
        node_count=None,
        branch_count=None,
        sample_nodes=10,
        top_n=3,
    )

    assert "[growth-profile] event=after_checkpoint_load generation=3" in caplog.text
    assert "top_gc_types=" in caplog.text
    assert "top_project_gc_types=" in caplog.text
    assert "node_sample unavailable reason=no_node_iterator" in caplog.text


def test_deep_size_handles_cycles_and_shared_references() -> None:
    """Recursive sizing should terminate and count shared objects once."""
    shared: list[object] = ["shared"]
    cycle: list[object] = [shared, shared]
    cycle.append(cycle)

    seen: set[int] = set()
    first_size = deep_size(cycle, seen=seen)
    second_size = deep_size(shared, seen=seen)

    assert first_size > 0
    assert second_size == 0


def test_deep_size_traverses_containers_dicts_and_slots() -> None:
    """Recursive sizing should include builtin containers, dict attrs, and slots."""
    payload = {"items": [1, (2, 3), {4, 5}]}
    root = RecursiveDictObject(RecursiveSlotObject(payload))

    root_size = deep_size(root, seen=set())
    payload_size = deep_size(payload, seen=set())

    assert root_size >= payload_size
    assert "child" in slot_names(root.child)


def test_deep_size_does_not_call_properties() -> None:
    """Recursive sizing must avoid properties that may materialize lazy state."""
    size = deep_size(RecursivePropertyObject(), seen=set())

    assert size > 0


def test_deep_size_skips_modules_functions_and_types() -> None:
    """Skipped runtime globals should be shallow-only and not deeply traversed."""
    seen: set[int] = set()
    module_size = deep_size(math, seen=seen)
    function_size = deep_size(
        test_deep_size_skips_modules_functions_and_types,
        seen=seen,
    )
    type_size = deep_size(RecursiveSlotObject, seen=seen)

    assert module_size == sys.getsizeof(math)
    assert function_size == sys.getsizeof(
        test_deep_size_skips_modules_functions_and_types
    )
    assert type_size == sys.getsizeof(RecursiveSlotObject)


def test_deep_size_caps_on_default_max_depth() -> None:
    """Recursive sizing should stop descending once the default depth cap is hit."""
    root = RecursiveDictObject(None)
    current = root
    for _ in range(80):
        child = RecursiveDictObject(None)
        current.child = child
        current = child

    stats = recursive_memory_profile_module.DeepSizeStats()
    size = deep_size(root, seen=set(), stats=stats)

    assert size > 0
    assert stats.max_depth_reached_count > 0
    assert stats.capped is True


def test_deep_size_honors_explicit_none_max_depth() -> None:
    """Explicit max_depth=None should disable depth-based capping."""
    root = RecursiveDictObject(None)
    current = root
    for _ in range(80):
        child = RecursiveDictObject(None)
        current.child = child
        current = child

    stats = recursive_memory_profile_module.DeepSizeStats()
    size = deep_size(root, seen=set(), max_depth=None, stats=stats)

    assert size > 0
    assert stats.max_depth_reached_count == 0
    assert stats.capped is False


def test_deep_size_handles_deeper_than_python_recursion_limit_iteratively() -> None:
    """Uncapped traversal should finish past Python's recursion limit."""
    root = RecursiveDictObject(None)
    current = root
    for _ in range(sys.getrecursionlimit() + 100):
        child = RecursiveDictObject(None)
        current.child = child
        current = child

    stats = recursive_memory_profile_module.DeepSizeStats()
    size = deep_size(root, seen=set(), max_depth=None, stats=stats)

    assert size > 0
    assert stats.capped is False
    assert stats.max_depth_reached_count == 0
    assert stats.recursion_error_count == 0


def test_deep_size_honors_explicit_max_depth() -> None:
    """Explicit max_depth should cap iterative traversal by queued depth."""
    root = RecursiveDictObject(None)
    current = root
    for _ in range(10):
        child = RecursiveDictObject(None)
        current.child = child
        current = child

    stats = recursive_memory_profile_module.DeepSizeStats()
    size = deep_size(root, seen=set(), max_depth=3, stats=stats)

    assert size > 0
    assert stats.capped is True
    assert stats.max_depth_reached_count > 0


def test_deep_size_honors_max_objects_cap() -> None:
    """Iterative traversal should preserve the object-visit cap."""
    root = [RecursiveDictObject(index) for index in range(10)]
    stats = recursive_memory_profile_module.DeepSizeStats(max_objects=3)

    size = deep_size(root, seen=set(), max_objects=3, stats=stats)

    assert size > 0
    assert stats.capped is True
    assert stats.visited_objects == 3


def test_exclusive_deep_size_respects_shared_seen_across_roots() -> None:
    """Exclusive deep sizing should count shared nested objects only once."""
    shared = RecursiveDictObject([1, 2, 3])
    roots = (
        RecursiveDictObject(shared),
        RecursiveDictObject(shared),
    )

    exclusive_size = recursive_memory_profile_module._exclusive_deep_size(
        roots,
        seen=set(),
        max_depth=None,
        stats=recursive_memory_profile_module.DeepSizeStats(),
    )
    seen: set[int] = set()
    combined_size = deep_size(roots[0], seen=seen, max_depth=None) + deep_size(
        roots[1],
        seen=seen,
        max_depth=None,
    )

    assert exclusive_size == combined_size


def test_deep_size_catches_recursion_error_from_attribute_iteration(
    monkeypatch,
) -> None:
    """Recursive sizing should convert attribute-iteration recursion failures to stats."""
    root = RecursionDictObject()
    original_iter = recursive_memory_profile_module._iter_object_attribute_values

    def raising_iter(value: object):
        if value is root:
            raise RecursionError("boom")
        yield from original_iter(value)

    monkeypatch.setattr(
        recursive_memory_profile_module,
        "_iter_object_attribute_values",
        raising_iter,
    )

    stats = recursive_memory_profile_module.DeepSizeStats()
    size = deep_size(root, seen=set(), stats=stats)

    assert size == sys.getsizeof(root)
    assert stats.recursion_error_count == 1
    assert stats.capped is True


def test_tree_topology_histograms_with_fake_nodes() -> None:
    """Topology histograms should use raw storage and count links."""
    parent = object()
    nodes = [
        FakeAlgorithmNode(
            FakeTreeNode(branches_children={"a": object()}, parent_nodes={}),
            FakeNodeEvaluation(),
        ),
        FakeAlgorithmNode(
            FakeTreeNode(parent_nodes={parent: {"b", "c"}}),
            FakeNodeEvaluation(runtime_state=object()),
        ),
    ]

    histograms = tree_topology_histograms(nodes)

    assert histograms["child_link_count"] == {"1": 1, "0": 1}
    assert histograms["parent_nodes_len"] == {"0": 1, "1": 1}
    assert histograms["parent_branch_refs"] == {"0": 1, "2": 1}
    assert "dict" in histograms["branches_children_types"]
    assert "NoneType" in histograms["branches_children_types"]


def test_node_evaluation_runtime_histograms_with_fake_nodes() -> None:
    """Node-evaluation histograms should count materialized runtime slots."""
    nodes = [
        FakeAlgorithmNode(FakeTreeNode(), FakeNodeEvaluation(runtime_state=object())),
        FakeAlgorithmNode(FakeTreeNode(), FakeNodeEvaluation()),
    ]

    histograms = node_evaluation_runtime_histograms(nodes)

    assert histograms["runtime_state_counts"]["decision_ordering__non_none"] == 1
    assert histograms["runtime_state_counts"]["direct_value_non_none"] == 2


def test_linoo_state_histograms_with_fake_selector() -> None:
    """Linoo histograms should classify default and non-default states."""
    selector = FakeLinooSelector(
        {
            1: FakeLinooNodeState(object(), status="opened"),
            2: FakeLinooNodeState(object(), status="frontier"),
        }
    )

    histograms = linoo_state_histograms(selector)

    assert histograms["present"] is True
    assert histograms["selector_type"].endswith("FakeLinooSelector")
    assert histograms["node_state_count"] == 2
    assert histograms["default_count"] == 1
    assert histograms["non_default_count"] == 1
    assert histograms["node_state_table_shallow_bytes"] > 0
    assert histograms["container_shallow_bytes"] >= (
        histograms["node_state_table_shallow_bytes"]
    )


def test_linoo_state_histograms_finds_nested_selector() -> None:
    """Linoo histograms should traverse composed selector wrappers."""
    selector = FakeLinooSelector(
        {
            1: FakeLinooNodeState(object(), status="opened"),
            2: FakeLinooNodeState(object(), status="frontier"),
        }
    )

    histograms = linoo_state_histograms(FakeComposedSelector(selector))

    assert histograms["present"] is True
    assert histograms["selector_type"].endswith("FakeLinooSelector")
    assert histograms["node_state_count"] == 2
    assert histograms["default_count"] == 1
    assert histograms["non_default_count"] == 1
    assert histograms["node_state_table_shallow_bytes"] > 0


def test_linoo_deep_breakdown_histograms_report_direct_fields() -> None:
    """Linoo deep breakdown should report direct selector fields safely."""
    selector = FakeLinooSelector(
        {
            1: FakeLinooNodeState(object(), status="opened"),
            2: FakeLinooNodeState(object(), status="frontier"),
        }
    )

    histograms = linoo_deep_breakdown_histograms(selector, max_depth=None)

    field_names = {histogram["field_name"] for histogram in histograms}
    assert "_node_state_by_id" in field_names
    assert "labels" in field_names
    assert "active_status" in field_names
    table_histogram = next(
        histogram
        for histogram in histograms
        if histogram["field_name"] == "_node_state_by_id"
    )
    assert table_histogram["recursive_reachable_bytes"] > 0
    assert table_histogram["recursion_error_count"] == 0


def test_linoo_node_state_table_histogram_detects_table() -> None:
    """Linoo table breakdown should expose table and state totals."""
    selector = FakeLinooSelector(
        {
            1: FakeLinooNodeState(object(), status="opened"),
            2: FakeLinooNodeState(object(), status="frontier"),
        }
    )

    histogram = linoo_node_state_table_histogram(selector, max_depth=None)

    assert histogram["present"] is True
    assert histogram["table_attr_name"] == "_node_state_by_id"
    assert histogram["table_length"] == 2
    assert histogram["table_recursive_reachable_bytes"] > 0
    assert histogram["node_state_count"] == 2
    assert histogram["node_states_shallow_bytes"] > 0
    assert histogram["node_states_recursive_reachable_bytes"] > 0


def test_linoo_node_state_slots_histogram_samples_slots() -> None:
    """Linoo slot breakdown should sample state slots and classify values."""
    selector = FakeLinooSelector(
        {
            1: FakeLinooNodeState(
                FakeAlgorithmNode(FakeTreeNode(), FakeNodeEvaluation()),
                status="opened",
            ),
            2: FakeLinooNodeState(
                FakeAlgorithmNode(FakeTreeNode(), FakeNodeEvaluation()),
                status="frontier",
            ),
        }
    )

    histogram = linoo_node_state_slots_histogram(selector, max_depth=None)

    assert histogram["present"] is True
    assert set(histogram["slot_names"]) == {"depth", "node", "status"}
    assert histogram["sampled_state_count"] == 2
    assert histogram["slot_value_kind_counts"]["AlgorithmNode"] == 2
    assert histogram["slot_value_kind_counts"]["int"] == 2
    assert histogram["slot_value_kind_counts"]["str"] == 2
    assert histogram["slot_recursive_reachable_bytes"]["node"] > 0


def test_linoo_breakdowns_do_not_crash_when_selector_missing() -> None:
    """Linoo breakdown helpers should stay safe when no selector is present."""
    assert linoo_deep_breakdown_histograms(None) == ({"present": False},)
    assert linoo_node_state_table_histogram(None) == {"present": False}
    assert linoo_node_state_slots_histogram(None) == {"present": False}


def test_linoo_breakdowns_do_not_crash_for_non_linoo_selector() -> None:
    """Linoo breakdown helpers should stay safe for unrelated selectors."""
    selector = SimpleNamespace(other={"value": 1})

    assert linoo_deep_breakdown_histograms(selector) == ({"present": False},)
    assert linoo_node_state_table_histogram(selector) == {"present": False}
    assert linoo_node_state_slots_histogram(selector) == {"present": False}


def test_linoo_breakdowns_respect_depth_and_object_caps() -> None:
    """Linoo breakdown helpers should honor recursive depth and object caps."""
    nested = RecursiveDictObject(RecursiveDictObject(RecursiveDictObject(None)))
    selector = FakeLinooSelector({1: FakeLinooNodeState(nested, status="opened")})

    deep_histograms = linoo_deep_breakdown_histograms(selector, max_depth=0)
    table_histogram = linoo_node_state_table_histogram(selector, max_objects=1)

    node_state_field = next(
        histogram
        for histogram in deep_histograms
        if histogram["field_name"] == "_node_state_by_id"
    )
    assert node_state_field["capped"] is True
    assert node_state_field["max_depth_reached_count"] > 0
    assert table_histogram["table_recursive_reachable_capped"] is True


def test_checkpoint_state_histograms_with_materialized_state_handles() -> None:
    """Checkpoint histograms should see materialized state handles without get()."""
    handle = SimpleNamespace(state_={"board": [0, 1, 2]})
    nodes = [FakeAlgorithmNode(FakeTreeNode(state_handle=handle), FakeNodeEvaluation())]

    histograms = checkpoint_state_histograms(nodes)

    assert histograms["materialized_state_count"] == 1
    assert histograms["materialized_state_recursive_bytes"] > 0


def test_checkpoint_state_histograms_caps_payload_traversal(
    caplog: LogCaptureFixture,
) -> None:
    """Checkpoint histograms should stop payload sizing once the cap is reached."""
    caplog.set_level(logging.INFO)
    payload_store = SimpleNamespace(
        payloads={
            1: FakeAnchorCheckpointStatePayload({"board": [1, 2, 3]}),
            2: FakeDeltaCheckpointStatePayload({"move": 4}),
        }
    )

    histograms = checkpoint_state_histograms(
        [],
        [payload_store],
        max_objects=1,
    )

    assert histograms["capped"] is True
    assert histograms["payloads_seen"] == 1
    assert histograms["anchors_seen"] == 1
    assert histograms["deltas_seen"] == 0
    assert histograms["payload_recursive_visited_objects"] == 1
    assert "checkpoint_state_histograms handles_seen=0" in caplog.text
    assert "payloads_seen=1" in caplog.text
    assert "capped=True" in caplog.text


def test_checkpoint_state_histograms_caps_handle_scan(
    caplog: LogCaptureFixture,
) -> None:
    """Checkpoint histograms should stop scanning handles at the scan cap."""
    caplog.set_level(logging.INFO)
    safe_nodes = [
        FakeAlgorithmNode(
            FakeTreeNode(state_handle=SimpleNamespace(state_={"index": index})),
            FakeNodeEvaluation(),
        )
        for index in range(3)
    ]
    dangerous_nodes = [
        FakeAlgorithmNode(
            FakeTreeNode(state_handle=RaisingCheckpointHandle()),
            FakeNodeEvaluation(),
        )
        for _ in range(5)
    ]

    histograms = checkpoint_state_histograms(
        [*safe_nodes, *dangerous_nodes],
        max_objects=3_000_000,
        checkpoint_max_handles=3,
    )

    assert histograms["handles_seen"] == 3
    assert histograms["handle_scan_capped"] is True
    assert histograms["handles_scanned_cap_reached"] is True
    assert histograms["capped"] is True
    assert "handles_scanned_cap_reached=True" in caplog.text
    assert "capped=True" in caplog.text


def test_checkpoint_payload_lifetime_histograms_detect_shared_resolver() -> None:
    """Lifetime histograms should join one payload store with shared-handle counts."""
    runner = FakeRunnerWithCheckpointBackedHandles()
    context = build_recursive_profile_context(
        runner,
        event="after_checkpoint_load",
        branch_count=0,
        node_cap=10,
    )
    payload_store_records = recursive_memory_profile_module._log_checkpoint_payload_stores(
        event="after_checkpoint_load",
        checkpoint_payload_stores=context.checkpoint_payload_stores,
        max_objects=None,
        max_depth=64,
        complete_map=False,
    )

    histograms = checkpoint_payload_lifetime_histograms(
        context.nodes,
        context.checkpoint_payload_stores,
        checkpoint_payload_store_records=payload_store_records,
    )

    assert len(histograms) == 1
    histogram = histograms[0]
    assert histogram["resolver_type"].endswith("FakeCheckpointStateResolver")
    assert histogram["payload_mapping_attr_name"] == "state_payloads_by_node_id"
    assert histogram["payload_mapping_length"] == 3
    assert histogram["anchor_count"] == 1
    assert histogram["delta_count"] == 1
    assert histogram["checkpoint_backed_state_handle_count"] == 2
    assert histogram["materialized_handle_count"] == 1
    assert histogram["unmaterialized_handle_count"] == 1
    assert histogram["payload_entries_still_referenced_count"] == 2
    assert histogram["all_handles_share_one_resolver"] is True
    assert histogram["payload_mapping_recursive_bytes"] > 0
    assert histogram["payload_mapping_shallow_bytes"] > 0


def test_checkpoint_payload_lifetime_histograms_do_not_materialize_states() -> None:
    """Lifetime histograms must not call handle.get() while counting lazy handles."""
    resolver = FakeCheckpointStateResolver()
    nodes = [
        FakeAlgorithmNode(
            FakeTreeNode(state_handle=FakeCheckpointBackedStateHandle(resolver, 2)),
            FakeNodeEvaluation(),
        )
    ]
    payload_store = recursive_memory_profile_module.CheckpointPayloadStore(
        resolver_type=type(resolver).__module__ + "." + type(resolver).__qualname__,
        resolver_id=id(resolver),
        owner_type=type(resolver).__module__ + "." + type(resolver).__qualname__,
        attr_name="state_payloads_by_node_id",
        payloads=resolver.state_payloads_by_node_id,
        anchor_count=1,
        delta_count=1,
    )

    histograms = checkpoint_payload_lifetime_histograms(
        nodes,
        [payload_store],
    )

    assert histograms[0]["checkpoint_backed_state_handle_count"] == 1
    assert histograms[0]["materialized_handle_count"] == 0
    assert histograms[0]["unmaterialized_handle_count"] == 1


def test_checkpoint_payload_lifetime_histograms_tolerate_missing_attrs() -> None:
    """Lifetime histograms should not crash when resolver payload attrs are absent."""
    resolver = SimpleNamespace()
    nodes = [
        FakeAlgorithmNode(
            FakeTreeNode(state_handle=FakeCheckpointBackedStateHandle(resolver, 7)),
            FakeNodeEvaluation(),
        )
    ]

    histograms = checkpoint_payload_lifetime_histograms(nodes, [])

    assert len(histograms) == 1
    histogram = histograms[0]
    assert histogram["payload_mapping_attr_name"] is None
    assert histogram["payload_mapping_type"] is None
    assert histogram["payload_mapping_length"] == 0
    assert histogram["checkpoint_backed_state_handle_count"] == 1
    assert histogram["materialized_handle_count"] == 0
    assert histogram["unmaterialized_handle_count"] == 1


def test_checkpoint_payload_lifetime_histograms_support_dense_store() -> None:
    """Checkpoint lifetime diagnostics should accept dense mapping-compatible stores."""
    resolver = FakeDenseCheckpointStateResolver()
    nodes = [
        FakeAlgorithmNode(
            FakeTreeNode(state_handle=FakeCheckpointBackedStateHandle(resolver, 0)),
            FakeNodeEvaluation(),
        ),
        FakeAlgorithmNode(
            FakeTreeNode(state_handle=FakeCheckpointBackedStateHandle(resolver, 1)),
            FakeNodeEvaluation(),
        ),
    ]
    payload_store = recursive_memory_profile_module.CheckpointPayloadStore(
        resolver_type=type(resolver).__module__ + "." + type(resolver).__qualname__,
        resolver_id=id(resolver),
        owner_type=type(resolver).__module__ + "." + type(resolver).__qualname__,
        attr_name="state_payloads_by_node_id",
        payloads=resolver.state_payloads_by_node_id,
        anchor_count=1,
        delta_count=1,
    )

    histograms = checkpoint_payload_lifetime_histograms(nodes, [payload_store])

    assert histograms[0]["payload_mapping_length"] == 2
    assert histograms[0]["checkpoint_backed_state_handle_count"] == 2
    assert histograms[0]["materialized_handle_count"] == 0
    assert histograms[0]["unmaterialized_handle_count"] == 2


def test_frozenset_ownership_histogram_tracks_state_fields_by_identity(
    monkeypatch,
) -> None:
    """Frozenset ownership histogram should bucket sizes and attribute state fields."""
    points = frozenset()
    used_unit_segments = frozenset({1})
    played_moves = frozenset({2, 3, 4})
    same_value_different_identity = frozenset([2, 3, 4])
    state = MorpionState(
        points=points,
        used_unit_segments=used_unit_segments,
        played_moves=played_moves,
    )
    checkpoint_payload = FakeCheckpointPayload(played_moves)
    dict_owner = {"used_unit_segments": used_unit_segments}
    referrers_by_id = {
        id(points): [state],
        id(used_unit_segments): [state, dict_owner],
        id(played_moves): [state, checkpoint_payload],
        id(same_value_different_identity): [state],
    }

    monkeypatch.setattr(
        gc,
        "get_referrers",
        lambda value: list(referrers_by_id.get(id(value), [])),
    )

    histogram = frozenset_ownership_histogram(
        objects=[points, used_unit_segments, played_moves, same_value_different_identity],
        sample_cap=10,
        top_n=10,
    )

    assert histogram["total_count"] == 4
    assert histogram["total_shallow_bytes"] == sum(
        sys.getsizeof(value)
        for value in (
            points,
            used_unit_segments,
            played_moves,
            same_value_different_identity,
        )
    )
    assert dict(histogram["len_buckets"]) == {"0": 1, "1": 1, "2-4": 2}
    assert dict(histogram["morpion_state_field_refs"]) == {
        "points": 1,
        "used_unit_segments": 1,
        "played_moves": 1,
    }
    assert dict(histogram["top_referrer_types"])[
        "atomheart.games.morpion.state.MorpionState"
    ] == 4
    assert dict(histogram["top_referrer_types"])["dict"] == 1
    assert dict(histogram["top_referrer_types"])[
        f"{FakeCheckpointPayload.__module__}.{FakeCheckpointPayload.__qualname__}"
    ] == 1


def test_frozenset_ownership_histogram_attributes_morpion_state_dict_owners(
    monkeypatch,
) -> None:
    """Frozenset ownership should resolve MorpionState __dict__ referrers."""
    points = frozenset({1, 2})
    used_unit_segments = frozenset({3, 4, 5})
    played_moves = frozenset({6})
    state = MorpionState(
        points=points,
        used_unit_segments=used_unit_segments,
        played_moves=played_moves,
    )
    referrers_by_id = {
        id(points): [state.__dict__],
        id(used_unit_segments): [state.__dict__],
        id(played_moves): [state.__dict__],
        id(state.__dict__): [state],
    }

    monkeypatch.setattr(
        gc,
        "get_referrers",
        lambda value: list(referrers_by_id.get(id(value), [])),
    )

    histogram = frozenset_ownership_histogram(
        objects=[points, used_unit_segments, played_moves],
        sample_cap=10,
        top_n=10,
    )

    assert dict(histogram["morpion_state_field_refs"]) == {
        "points": 1,
        "used_unit_segments": 1,
        "played_moves": 1,
    }
    assert dict(histogram["top_referrer_types"])["dict"] == 3


def test_frozenset_ownership_histogram_respects_sample_cap(monkeypatch) -> None:
    """Frozenset ownership histogram should bound referrer scans to the sample cap."""
    scanned_ids: list[int] = []
    frozensets = [frozenset({index}) for index in range(4)]

    def fake_get_referrers(value: object) -> list[object]:
        scanned_ids.append(id(value))
        return []

    monkeypatch.setattr(gc, "get_referrers", fake_get_referrers)

    histogram = frozenset_ownership_histogram(
        objects=frozensets,
        sample_cap=2,
        top_n=5,
    )

    assert histogram["total_count"] == 4
    assert histogram["sample_count"] == 2
    assert len(scanned_ids) == 2
    assert scanned_ids == [id(frozensets[0]), id(frozensets[1])]


def test_gc_shallow_size_summary_skips_reverse_referrer_scan(monkeypatch) -> None:
    """Default shallow summary should not call gc.get_referrers."""

    def fail_get_referrers(*_args: object) -> list[object]:
        raise AssertionError("gc.get_referrers should not be used by default")

    monkeypatch.setattr(gc, "get_referrers", fail_get_referrers)

    summary = gc_shallow_size_summary(top_n=5)

    frozenset_ownership = summary["frozenset_ownership"]
    assert isinstance(summary["object_count"], int)
    assert summary["object_count"] >= 0
    assert isinstance(frozenset_ownership, dict)
    assert "total_count" in frozenset_ownership
    assert "morpion_state_field_refs" in frozenset_ownership


def test_gc_shallow_size_summary_attributes_fake_morpion_state_fields(monkeypatch) -> None:
    """Default shallow summary should attribute MorpionState frozenset fields directly."""
    points = frozenset()
    used_unit_segments = frozenset({1})
    played_moves = frozenset({2, 3, 4})
    state = MorpionState(
        points=points,
        used_unit_segments=used_unit_segments,
        played_moves=played_moves,
    )
    unrelated = FakeCheckpointPayload(played_moves)
    gc_objects = [points, used_unit_segments, played_moves, state, unrelated]

    monkeypatch.setattr(gc, "get_objects", lambda: gc_objects)
    monkeypatch.setattr(
        gc,
        "get_referrers",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("gc.get_referrers should not be used by default")
        ),
    )

    summary = gc_shallow_size_summary(top_n=5)

    frozenset_ownership = summary["frozenset_ownership"]
    assert frozenset_ownership["total_count"] == 3
    assert frozenset_ownership["total_shallow_bytes"] == sum(
        sys.getsizeof(value) for value in (points, used_unit_segments, played_moves)
    )
    assert dict(frozenset_ownership["len_buckets"]) == {"0": 1, "1": 1, "2-4": 1}
    assert frozenset_ownership["morpion_state_count"] == 1
    assert dict(frozenset_ownership["morpion_state_field_refs"]) == {
        "points": 1,
        "used_unit_segments": 1,
        "played_moves": 1,
    }
    assert frozenset_ownership["morpion_state_field_shallow_bytes"] == sum(
        sys.getsizeof(value) for value in (points, used_unit_segments, played_moves)
    )
    assert dict(frozenset_ownership["morpion_state_field_len_buckets"]) == {
        "0": 1,
        "1": 1,
        "2-4": 1,
    }


def test_growth_recursive_memory_profile_finds_checkpoint_payload_store(
    caplog: LogCaptureFixture,
) -> None:
    """Recursive profile should find nested checkpoint payload-owner mappings."""
    caplog.set_level(logging.INFO)

    log_growth_recursive_memory_profile(
        runner=FakeRunnerWithCheckpointPayloadStore(),
        generation=31,
        event="after_checkpoint_load",
        node_count=0,
        branch_count=None,
        max_objects=None,
    )

    text = caplog.text
    assert "mode=standalone component=checkpoint_state_roots" in text
    assert "histogram=checkpoint_state" in text
    assert "histogram=checkpoint_payload_lifetime" in text
    assert "payload_store_count=1" in text
    assert "anchor_payload_count=1" in text
    assert "delta_payload_count=1" in text
    assert "checkpoint_payload_store index=1" in text
    assert "owner_type=" in text
    assert "FakeCheckpointPayloadOwner" in text
    assert "attr_name=owner._payloads_by_node_id" in text
    assert "mapping_length=3" in text
    assert "anchor_count=1" in text
    assert "delta_count=1" in text
    assert "mb=" in text


def test_growth_recursive_memory_profile_logs_total_when_checkpoint_histogram_capped(
    caplog: LogCaptureFixture,
) -> None:
    """Capped checkpoint histograms should not suppress the final total log."""
    caplog.set_level(logging.INFO)

    log_growth_recursive_memory_profile(
        runner=FakeRunnerWithCheckpointPayloadStore(),
        generation=32,
        event="after_checkpoint_load",
        node_count=0,
        branch_count=None,
        max_objects=1,
    )

    text = caplog.text
    assert "histogram=checkpoint_state" in text
    assert "capped=True" in text
    assert "total_recursive_reachable_mb=" in text
    assert "[growth-recursive-profile-summary] event=after_checkpoint_load" in text
    assert "capped_components=[" in text
    assert "max_objects=1" in text


def test_growth_recursive_memory_profile_logs_components(
    caplog: LogCaptureFixture,
) -> None:
    """Recursive profiles should log standalone, exclusive, histograms, residuals."""
    caplog.set_level(logging.INFO)

    log_growth_recursive_memory_profile(
        runner=FakeRunnerWithNestedAttrs(),
        generation=21,
        event="after_checkpoint_load",
        node_count=3,
        branch_count=None,
        max_objects=None,
    )

    text = caplog.text
    assert (
        "[growth-recursive-profile] event=after_checkpoint_load generation=21" in text
    )
    assert "max_depth=64 complete_map=False" in text
    assert "mode=standalone component=all_profile_nodes" in text
    assert "mode=exclusive order=" in text
    assert "histogram=tree_topology" in text
    assert "histogram=node_evaluation_runtime" in text
    assert "histogram=linoo" in text
    assert "histogram=linoo_deep_breakdown" in text
    assert "histogram=linoo_node_state_table" in text
    assert "histogram=linoo_node_state_slots" in text
    assert "gc_shallow_size_summary_start" in text
    assert "gc_shallow_size_summary_done" in text
    assert "histogram=gc_shallow_sizes" in text
    assert "histogram=frozenset_ownership" in text
    assert "morpion_state_count=" in text
    assert "morpion_state_field_shallow_mb=" in text
    assert "top_by_bytes=" in text
    assert "histogram=checkpoint_state" in text
    assert "rss_minus_reachable_mb=" in text
    assert "[growth-recursive-profile-summary] event=after_checkpoint_load" in text
    assert "largest_components=" in text


def test_growth_recursive_memory_profile_logs_context_build_progress(
    caplog: LogCaptureFixture,
) -> None:
    """Recursive profile should emit context-build progress logs and honor node cap."""
    caplog.set_level(logging.INFO)

    log_growth_recursive_memory_profile(
        runner=FakeRunnerWithCheckpointStoreAndProfileIterator(),
        generation=22,
        event="after_checkpoint_load",
        node_count=4,
        branch_count=8,
        max_objects=1,
        context_node_cap=2,
    )

    text = caplog.text
    assert "context_build_start" in text
    assert "context_build_runtime_found" in text
    assert "context_build_nodes_start" in text
    assert "context_build_nodes_done node_count=2" in text
    assert "context_build_branches_start" in text
    assert "context_build_branches_done branch_count=8" in text
    assert "context_build_checkpoint_stores_start" in text
    assert "context_build_checkpoint_stores_known_paths_start" in text
    assert "context_build_checkpoint_stores_known_paths_done count=1" in text
    assert "context_build_checkpoint_stores_done count=1" in text
    assert "context_build_checkpoint_stores_handle_fallback_start" not in text
    assert "context_build_done total_elapsed_s=" in text
    assert "profile_node_count=2 checkpoint_payload_stores=1" in text


def test_known_path_checkpoint_store_discovery_finds_direct_resolver() -> None:
    """Known-path discovery should find direct resolver payload stores cheaply."""
    stores = recursive_memory_profile_module._find_checkpoint_payload_stores_from_known_paths_only(
        runner=SimpleNamespace(
            profile_checkpoint_state_resolver=FakeResolverWithPayloads()
        ),
        runtime=None,
    )

    assert len(stores) == 1
    assert stores[0].attr_name == "owner._payloads_by_node_id"


def test_build_recursive_profile_context_direct_checkpoint_store_skips_handles(
    monkeypatch,
    caplog: LogCaptureFixture,
) -> None:
    """Direct checkpoint store discovery should avoid scanning node handles."""
    monkeypatch.setattr(
        recursive_memory_profile_module,
        "_find_checkpoint_payload_stores",
        lambda _roots: (_ for _ in ()).throw(
            AssertionError("generic checkpoint store discovery must not run")
        ),
    )
    caplog.set_level(logging.INFO)

    context = build_recursive_profile_context(
        FakeRunnerWithDirectCheckpointStoreAndDangerousHandles(),
        event="after_checkpoint_load",
        branch_count=0,
        node_cap=4,
    )

    text = caplog.text
    assert len(context.checkpoint_payload_stores) == 1
    assert "context_build_checkpoint_stores_known_paths_done count=1" in text
    assert "context_build_checkpoint_stores_handle_fallback_start" not in text


def test_build_recursive_profile_context_caps_handle_fallback_scan(
    monkeypatch,
    caplog: LogCaptureFixture,
) -> None:
    """Fallback checkpoint store discovery should inspect only a capped handle set."""
    runner = FakeRunnerWithFallbackCheckpointHandles(node_count=1_500)
    scanned_handle_ids: list[int] = []
    original_handle_resolver = recursive_memory_profile_module._handle_resolver

    def fake_handle_resolver(handle: object) -> object | None:
        scanned_handle_ids.append(id(handle))
        return original_handle_resolver(handle)

    monkeypatch.setattr(
        recursive_memory_profile_module,
        "_handle_resolver",
        fake_handle_resolver,
    )
    caplog.set_level(logging.INFO)

    context = build_recursive_profile_context(
        runner,
        event="after_checkpoint_load",
        branch_count=0,
        node_cap=50_000,
    )

    text = caplog.text
    assert len(context.checkpoint_payload_stores) == 0
    assert (
        len(scanned_handle_ids)
        == recursive_memory_profile_module._DEFAULT_CHECKPOINT_STORE_HANDLE_DISCOVERY_CAP
    )
    assert "context_build_checkpoint_stores_known_paths_done count=0" in text
    assert "context_build_checkpoint_stores_handle_fallback_start handle_cap=100" in text
    assert "context_build_checkpoint_stores_handle_fallback_done count=0" in text


def test_build_recursive_profile_context_known_paths_do_not_walk_runtime(
    monkeypatch,
    caplog: LogCaptureFixture,
) -> None:
    """Context building should complete without recursively walking runner/runtime."""
    runner = FakeRunnerWithProfileIterator()
    runner._runtime = RecursivelyDangerousRuntime(FakeResolverWithPayloads())
    monkeypatch.setattr(
        recursive_memory_profile_module,
        "_find_checkpoint_payload_stores",
        lambda _roots: (_ for _ in ()).throw(
            AssertionError("generic checkpoint store discovery must not run")
        ),
    )
    caplog.set_level(logging.INFO)

    context = build_recursive_profile_context(
        runner,
        event="after_checkpoint_load",
        branch_count=0,
        node_cap=4,
    )

    text = caplog.text
    assert len(context.checkpoint_payload_stores) == 1
    assert "context_build_checkpoint_stores_known_paths_done count=1" in text
    assert "context_build_checkpoint_stores_done count=1" in text


def test_growth_recursive_memory_profile_logs_recursion_errors_and_continues(
    monkeypatch,
    caplog: LogCaptureFixture,
) -> None:
    """Recursive profile should log capped components instead of crashing on recursion."""
    runner = FakeRunnerWithProfileIterator()
    sentinel_runtime = RecursionDictObject()
    runner._runtime = sentinel_runtime
    original_iter = recursive_memory_profile_module._iter_object_attribute_values

    def raising_iter(value: object):
        if value is sentinel_runtime:
            raise RecursionError("runtime recursion")
        yield from original_iter(value)

    monkeypatch.setattr(
        recursive_memory_profile_module,
        "_iter_object_attribute_values",
        raising_iter,
    )
    caplog.set_level(logging.INFO)

    log_growth_recursive_memory_profile(
        runner=runner,
        generation=23,
        event="after_checkpoint_load",
        node_count=4,
        branch_count=4,
        max_objects=1,
    )

    text = caplog.text
    assert "component=remaining_runtime" in text
    assert "recursion_error_count=" in text
    assert "capped=True" in text
    assert "[growth-recursive-profile-summary] event=after_checkpoint_load" in text


def test_growth_recursive_memory_profile_complete_map_defaults_to_no_depth_cap(
    caplog: LogCaptureFixture,
) -> None:
    """Complete-map mode should default recursive depth traversal to uncapped."""
    caplog.set_level(logging.INFO)

    log_growth_recursive_memory_profile(
        runner=FakeRunnerWithNestedAttrs(),
        generation=24,
        event="after_checkpoint_load",
        node_count=3,
        branch_count=None,
        max_objects=1,
        complete_map=True,
    )

    text = caplog.text
    assert "max_depth=None complete_map=True" in text
    assert "recursion_error_count=0" in text


def test_growth_recursive_memory_profile_honors_explicit_max_depth(
    caplog: LogCaptureFixture,
) -> None:
    """Explicit recursive max_depth values should override complete-map defaults."""
    caplog.set_level(logging.INFO)

    log_growth_recursive_memory_profile(
        runner=FakeRunnerWithNestedAttrs(),
        generation=25,
        event="after_checkpoint_load",
        node_count=3,
        branch_count=None,
        max_objects=1,
        max_depth=256,
        complete_map=True,
        max_depth_explicit=True,
    )

    text = caplog.text
    assert "max_depth=256 complete_map=True" in text


def test_growth_recursive_memory_profile_honors_explicit_none_max_depth(
    caplog: LogCaptureFixture,
) -> None:
    """Explicit max_depth=None should stay uncapped outside complete-map mode."""
    caplog.set_level(logging.INFO)

    log_growth_recursive_memory_profile(
        runner=FakeRunnerWithNestedAttrs(),
        generation=26,
        event="after_checkpoint_load",
        node_count=3,
        branch_count=None,
        max_objects=1,
        max_depth=None,
        max_depth_explicit=True,
    )

    text = caplog.text
    assert "max_depth=None complete_map=False" in text

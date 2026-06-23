"""Tests for opt-in Morpion growth runtime memory profiling."""
# ruff: noqa: E402, D102, D105, D107, ANN201, ANN204, TRY003

from __future__ import annotations

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
    checkpoint_state_histograms,
    deep_size,
    linoo_state_histograms,
    log_growth_recursive_memory_profile,
    node_evaluation_runtime_histograms,
    slot_names,
    tree_topology_histograms,
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


def test_checkpoint_state_histograms_with_materialized_state_handles() -> None:
    """Checkpoint histograms should see materialized state handles without get()."""
    handle = SimpleNamespace(state_={"board": [0, 1, 2]})
    nodes = [FakeAlgorithmNode(FakeTreeNode(state_handle=handle), FakeNodeEvaluation())]

    histograms = checkpoint_state_histograms(nodes)

    assert histograms["materialized_state_count"] == 1
    assert histograms["materialized_state_recursive_bytes"] > 0


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
    assert "payload_store_count=1" in text
    assert "anchor_payload_count=1" in text
    assert "delta_payload_count=1" in text
    assert "checkpoint_payload_store index=1" in text
    assert "owner_type=" in text
    assert "FakeCheckpointPayloadOwner" in text
    assert "attr_name=_payloads_by_node_id" in text
    assert "mapping_length=3" in text
    assert "anchor_count=1" in text
    assert "delta_count=1" in text
    assert "mb=" in text


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
    assert "mode=standalone component=all_profile_nodes" in text
    assert "mode=exclusive order=" in text
    assert "histogram=tree_topology" in text
    assert "histogram=node_evaluation_runtime" in text
    assert "histogram=linoo" in text
    assert "histogram=checkpoint_state" in text
    assert "rss_minus_reachable_mb=" in text

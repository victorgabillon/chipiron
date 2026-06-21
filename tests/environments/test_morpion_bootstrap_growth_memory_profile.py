"""Tests for opt-in Morpion growth runtime memory profiling."""
# ruff: noqa: E402

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType
from typing import TYPE_CHECKING

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHIPIRON_PACKAGE_ROOT = _REPO_ROOT / "src" / "chipiron"
_BOOTSTRAP_PACKAGE_ROOT = _CHIPIRON_PACKAGE_ROOT / "environments" / "morpion" / "bootstrap"

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


class RunnerWithoutNodes:
    """Runner-like object with no discoverable node store."""

    def __init__(self) -> None:
        """Initialize a runner without node-like attributes."""
        self.name = "no-nodes"


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

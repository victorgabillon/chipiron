"""Tests for Morpion bootstrap memory diagnostics."""
# ruff: noqa: E402

from __future__ import annotations

import gc
import importlib
import logging
import sys
import weakref
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

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

from chipiron.environments.morpion.bootstrap.memory_diagnostics import (
    MemoryDiagnostics,
    MemoryDiagnosticsConfig,
)


def test_memory_diagnostics_disabled_does_not_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Disabled diagnostics should be a silent no-op."""
    caplog.set_level(logging.INFO)
    diagnostics = MemoryDiagnostics(MemoryDiagnosticsConfig(enabled=False))

    diagnostics.log("disabled")
    diagnostics.close()

    assert not caplog.records


def test_memory_diagnostics_enabled_logs_rss(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enabled diagnostics should log RSS/VMS when psutil is available."""

    class _FakeProcess:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        def memory_info(self) -> SimpleNamespace:
            return SimpleNamespace(rss=2 * 1024 * 1024, vms=5 * 1024 * 1024)

    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(Process=_FakeProcess),
    )
    caplog.set_level(logging.INFO)
    diagnostics = MemoryDiagnostics(MemoryDiagnosticsConfig(enabled=True))

    diagnostics.log("rss")
    diagnostics.close()

    assert any(
        "[memory] tag=rss rss_mb=2.000 vms_mb=5.000" in record.message
        for record in caplog.records
    )


def test_memory_diagnostics_gc_growth_does_not_retain_objects(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GC growth diagnostics should aggregate by type without object retention."""

    class _FakeProcess:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        def memory_info(self) -> SimpleNamespace:
            return SimpleNamespace(rss=1, vms=1)

    class _Marker:
        pass

    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(Process=_FakeProcess),
    )
    caplog.set_level(logging.INFO)
    diagnostics = MemoryDiagnostics(
        MemoryDiagnosticsConfig(enabled=True, gc_growth=True, top_n=5)
    )
    marker = _Marker()
    marker_ref = weakref.ref(marker)

    diagnostics.log("gc")
    del marker
    diagnostics.close()

    assert marker_ref() is None
    assert any("[memory_gc] tag=gc" in record.message for record in caplog.records)
    assert any(
        "[memory_gc_type] tag=gc" in record.message for record in caplog.records
    )


def test_memory_diagnostics_tracemalloc_logs_diffs(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tracemalloc diagnostics should accept repeated checkpoints."""

    class _FakeProcess:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        def memory_info(self) -> SimpleNamespace:
            return SimpleNamespace(rss=1, vms=1)

    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(Process=_FakeProcess),
    )
    caplog.set_level(logging.INFO)
    diagnostics = MemoryDiagnostics(
        MemoryDiagnosticsConfig(enabled=True, tracemalloc=True, top_n=3)
    )

    diagnostics.log("trace_1")
    _allocated = [object() for _ in range(10)]
    diagnostics.log("trace_2")
    diagnostics.close()

    assert any(
        "[memory_tracemalloc] tag=trace_1 baseline=true" in record.message
        for record in caplog.records
    )
    assert any(
        "[memory_tracemalloc] tag=trace_2 rank=1" in record.message
        for record in caplog.records
    )
    assert _allocated


def test_memory_diagnostics_torch_tensors_logs_retained_tensor_summary(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Torch tensor diagnostics should log tensor counts without retaining tensors."""
    torch = pytest.importorskip("torch")

    class _FakeProcess:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        def memory_info(self) -> SimpleNamespace:
            return SimpleNamespace(rss=1, vms=1)

    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(Process=_FakeProcess),
    )
    caplog.set_level(logging.INFO)
    diagnostics = MemoryDiagnostics(
        MemoryDiagnosticsConfig(enabled=True, torch_tensors=True, top_n=3)
    )
    tensor = torch.ones(4, dtype=torch.float32)

    diagnostics.log("torch")
    del tensor
    diagnostics.close()

    assert any("[memory_torch] tag=torch" in record.message for record in caplog.records)
    assert any(
        "[memory_torch_kind] tag=torch" in record.message for record in caplog.records
    )


def test_memory_diagnostics_referrers_logs_matching_object_and_referrer(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Referrer diagnostics should report a matching object and one holder."""

    class _FakeProcess:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        def memory_info(self) -> SimpleNamespace:
            return SimpleNamespace(rss=1, vms=1)

    class _SyntheticPayload:
        __slots__ = ("__weakref__", "name")

        def __init__(self) -> None:
            self.name = "payload"

    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(Process=_FakeProcess),
    )
    payload = _SyntheticPayload()
    holder = [payload]
    payload_type_name = f"{type(payload).__module__}.{type(payload).__qualname__}"
    caplog.set_level(logging.INFO)
    diagnostics = MemoryDiagnostics(
        MemoryDiagnosticsConfig(
            enabled=True,
            referrers=True,
            referrer_type_patterns=(payload_type_name,),
            referrer_max_depth=1,
            referrer_max_objects_per_type=1,
            referrer_top_n=5,
        )
    )

    diagnostics.log("refs")
    diagnostics.close()

    assert holder
    assert any(
        "[memory_referrer_object] tag=refs" in record.message
        and payload_type_name in record.message
        for record in caplog.records
    )
    assert any(
        "[memory_referrer] tag=refs" in record.message
        and "builtins.list" in record.message
        for record in caplog.records
    )


def test_memory_diagnostics_referrers_close_does_not_retain_inspected_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Referrer diagnostics should clear temporary object containers on close."""

    class _FakeProcess:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        def memory_info(self) -> SimpleNamespace:
            return SimpleNamespace(rss=1, vms=1)

    class _SyntheticPayload:
        __slots__ = ("__weakref__", "name")

        def __init__(self) -> None:
            self.name = "payload"

    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(Process=_FakeProcess),
    )

    def _inspect_and_release() -> weakref.ReferenceType[_SyntheticPayload]:
        payload = _SyntheticPayload()
        holder = [payload]
        payload_type_name = f"{type(payload).__module__}.{type(payload).__qualname__}"
        payload_ref = weakref.ref(payload)
        diagnostics = MemoryDiagnostics(
            MemoryDiagnosticsConfig(
                enabled=True,
                referrers=True,
                referrer_type_patterns=(payload_type_name,),
                referrer_max_depth=1,
                referrer_max_objects_per_type=1,
            )
        )
        diagnostics.log("refs_release")
        holder.clear()
        del payload
        diagnostics.close()
        return payload_ref

    payload_ref = _inspect_and_release()
    gc.collect()

    assert payload_ref() is None


def test_launcher_parser_accepts_memory_diagnostics_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The launcher parser should thread memory diagnostics flags into args."""
    _install_launcher_import_stubs(monkeypatch)
    try:
        launcher = importlib.import_module(
            "chipiron.environments.morpion.bootstrap.launcher"
        )
        parsed = launcher.launcher_args_from_cli(
            [
                "--work-dir",
                "/tmp/morpion-memory",
                "--memory-diagnostics",
                "--memory-diagnostics-gc-growth",
                "--memory-diagnostics-tracemalloc",
                "--memory-diagnostics-torch-tensors",
                "--memory-diagnostics-referrers",
                "--memory-diagnostics-referrer-type-pattern",
                "example.Payload",
                "--memory-diagnostics-referrer-type-pattern",
                "example.OtherPayload",
                "--memory-diagnostics-referrer-max-objects-per-type",
                "3",
                "--memory-diagnostics-referrer-max-depth",
                "4",
                "--memory-diagnostics-top-n",
                "7",
            ]
        )
    finally:
        sys.modules.pop("chipiron.environments.morpion.bootstrap.launcher", None)

    assert parsed.bootstrap_args.memory_diagnostics is True
    assert parsed.bootstrap_args.memory_diagnostics_gc_growth is True
    assert parsed.bootstrap_args.memory_diagnostics_tracemalloc is True
    assert parsed.bootstrap_args.memory_diagnostics_torch_tensors is True
    assert parsed.bootstrap_args.memory_diagnostics_referrers is True
    assert parsed.bootstrap_args.memory_diagnostics_referrer_type_patterns == (
        "example.Payload",
        "example.OtherPayload",
    )
    assert parsed.bootstrap_args.memory_diagnostics_referrer_max_objects_per_type == 3
    assert parsed.bootstrap_args.memory_diagnostics_referrer_max_depth == 4
    assert parsed.bootstrap_args.memory_diagnostics_top_n == 7


def _install_launcher_import_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    def _simple_function(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        return None

    class _BootstrapArgs:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            self.__dict__.update(kwargs)

    stubs = {
        "chipiron.environments.morpion.bootstrap.anemone_runner": {
            "AnemoneMorpionSearchRunner": object,
            "AnemoneMorpionSearchRunnerArgs": object,
            "apply_runtime_control_to_runner_args": _simple_function,
        },
        "chipiron.environments.morpion.bootstrap.bootstrap_loop": {
            "MorpionBootstrapArgs": _BootstrapArgs,
            "MorpionBootstrapPaths": object,
            "run_morpion_bootstrap_loop": _simple_function,
        },
        "chipiron.environments.morpion.bootstrap.config": {
            "DEFAULT_MORPION_TREE_BRANCH_LIMIT": 128,
            "MorpionBootstrapConfig": object,
            "bootstrap_config_from_args": _simple_function,
            "load_bootstrap_config": _simple_function,
        },
        "chipiron.environments.morpion.bootstrap.control": {
            "MorpionBootstrapControl": object,
            "effective_runtime_config_from_config_and_control": _simple_function,
            "load_bootstrap_control": _simple_function,
        },
        "chipiron.environments.morpion.bootstrap.evaluator_family": {
            "CANONICAL_MORPION_EVALUATOR_FAMILY_PRESET": "canonical",
        },
        "chipiron.environments.morpion.bootstrap.history": {
            "MorpionBootstrapLatestStatus": object,
            "load_latest_bootstrap_status": _simple_function,
        },
        "chipiron.environments.morpion.bootstrap.process_control": {
            "mark_current_launcher_process_stopped": _simple_function,
            "register_current_launcher_process": _simple_function,
        },
        "chipiron.environments.morpion.bootstrap.run_state": {
            "MorpionBootstrapRunState": object,
            "load_bootstrap_run_state": _simple_function,
        },
    }
    for module_name, attributes in stubs.items():
        module = ModuleType(module_name)
        for attribute_name, value in attributes.items():
            setattr(module, attribute_name, value)
        monkeypatch.setitem(sys.modules, module_name, module)

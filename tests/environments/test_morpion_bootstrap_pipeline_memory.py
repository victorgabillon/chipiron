"""Tests for Morpion artifact-pipeline memory guard helpers."""
# ruff: noqa: E402

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHIPIRON_PACKAGE_ROOT = _REPO_ROOT / "src" / "chipiron"

if "chipiron" not in sys.modules:
    _chipiron_stub = ModuleType("chipiron")
    _chipiron_stub.__path__ = [str(_CHIPIRON_PACKAGE_ROOT)]
    sys.modules["chipiron"] = _chipiron_stub

import chipiron.environments.morpion.bootstrap.pipeline_memory as pipeline_memory_module
from chipiron.environments.morpion.bootstrap.pipeline_memory import (
    _available_ram_mb_from_meminfo_text,
    candidate_checkpoint_load_memory_forecast,
    has_min_available_ram,
    log_available_ram_guard,
)


def test_available_ram_mb_parses_meminfo_text() -> None:
    """MemAvailable should be parsed from Linux /proc/meminfo-style content."""
    meminfo = "\n".join(
        (
            "MemTotal:       32768000 kB",
            "MemFree:         1024000 kB",
            "MemAvailable:    5120000 kB",
            "Buffers:          128000 kB",
        )
    )

    assert _available_ram_mb_from_meminfo_text(meminfo) == 5000.0


def test_disabled_ram_guard_always_allows(monkeypatch: pytest.MonkeyPatch) -> None:
    """None and zero thresholds should preserve the old always-run behavior."""
    monkeypatch.setattr(pipeline_memory_module, "available_ram_mb", lambda: 1.0)

    assert has_min_available_ram(None)
    assert has_min_available_ram(0)


def test_ram_guard_logs_skip_when_available_ram_is_low(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Insufficient available RAM should log a skip decision."""
    monkeypatch.setattr(pipeline_memory_module, "available_ram_mb", lambda: 1024.0)

    with caplog.at_level(logging.INFO):
        should_run = log_available_ram_guard(
            stage="dataset",
            generation=3,
            action="snapshot_load",
            required_mb=5000,
        )

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert not should_run
    assert (
        "[ram-guard] stage=dataset generation=3 action=snapshot_load "
        "available_mb=1024.0 required_mb=5000 decision=skip"
    ) in messages


def test_ram_guard_logs_run_when_available_ram_is_sufficient(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Sufficient available RAM should log a run decision."""
    monkeypatch.setattr(pipeline_memory_module, "available_ram_mb", lambda: 6000.0)

    with caplog.at_level(logging.INFO):
        should_run = log_available_ram_guard(
            stage="training",
            generation=4,
            action="rows_load",
            required_mb=5000,
        )

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert should_run
    assert (
        "[ram-guard] stage=training generation=4 action=rows_load "
        "available_mb=6000.0 required_mb=5000 decision=run"
    ) in messages


def test_ram_guard_allows_when_available_ram_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unknown available RAM should not block a guarded action."""
    monkeypatch.setattr(pipeline_memory_module, "available_ram_mb", lambda: None)

    with caplog.at_level(logging.INFO):
        should_run = log_available_ram_guard(
            stage="growth",
            generation=5,
            action="tree_growth",
            required_mb=5000,
        )

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert should_run
    assert (
        "[ram-guard] stage=growth generation=5 action=tree_growth "
        "available_mb=None required_mb=5000 decision=run"
    ) in messages


def test_candidate_checkpoint_load_forecast_skips_when_headroom_is_insufficient(
    tmp_path: Path,
) -> None:
    """Forecast should require min RAM plus estimated checkpoint load headroom."""
    checkpoint_path = tmp_path / "checkpoint.json.zst"
    checkpoint_path.write_bytes(b"0" * (100 * 1024 * 1024))

    forecast = candidate_checkpoint_load_memory_forecast(
        checkpoint_path=checkpoint_path,
        available_mb=9_000.0,
        min_available_ram_mb=5_000,
        headroom_factor=60.0,
        min_headroom_mb=512,
    )

    assert forecast.checkpoint_bytes == 100 * 1024 * 1024
    assert forecast.checkpoint_mb == 100.0
    assert forecast.estimated_load_mb == 6_000.0
    assert forecast.required_pre_load_available_mb == 11_000.0
    assert forecast.decision == "skip"


def test_candidate_checkpoint_load_forecast_runs_when_headroom_is_sufficient(
    tmp_path: Path,
) -> None:
    """Enough available RAM should allow candidate checkpoint payload loading."""
    checkpoint_path = tmp_path / "checkpoint.json.zst"
    checkpoint_path.write_bytes(b"0" * (100 * 1024 * 1024))

    forecast = candidate_checkpoint_load_memory_forecast(
        checkpoint_path=checkpoint_path,
        available_mb=12_000.0,
        min_available_ram_mb=5_000,
        headroom_factor=60.0,
        min_headroom_mb=512,
    )

    assert forecast.required_pre_load_available_mb == 11_000.0
    assert forecast.decision == "run"


def test_candidate_checkpoint_load_forecast_allows_when_ram_guard_disabled(
    tmp_path: Path,
) -> None:
    """Without a post-load RAM requirement the forecast should not block loads."""
    checkpoint_path = tmp_path / "checkpoint.json.zst"
    checkpoint_path.write_bytes(b"0" * (100 * 1024 * 1024))

    forecast = candidate_checkpoint_load_memory_forecast(
        checkpoint_path=checkpoint_path,
        available_mb=1.0,
        min_available_ram_mb=None,
        headroom_factor=60.0,
        min_headroom_mb=512,
    )

    assert forecast.required_pre_load_available_mb is None
    assert forecast.decision == "run"


def test_candidate_checkpoint_load_forecast_allows_unknown_available_ram(
    tmp_path: Path,
) -> None:
    """Unknown available RAM should preserve existing non-blocking guard behavior."""
    checkpoint_path = tmp_path / "checkpoint.json.zst"
    checkpoint_path.write_bytes(b"0" * (100 * 1024 * 1024))

    forecast = candidate_checkpoint_load_memory_forecast(
        checkpoint_path=checkpoint_path,
        available_mb=None,
        min_available_ram_mb=5_000,
        headroom_factor=60.0,
        min_headroom_mb=512,
    )

    assert forecast.required_pre_load_available_mb == 11_000.0
    assert forecast.decision == "run"


def test_candidate_checkpoint_load_forecast_zero_headroom_matches_simple_guard(
    tmp_path: Path,
) -> None:
    """Zero factor/minimum should reduce forecast to the old RAM threshold."""
    checkpoint_path = tmp_path / "checkpoint.json.zst"
    checkpoint_path.write_bytes(b"0" * (100 * 1024 * 1024))

    forecast = candidate_checkpoint_load_memory_forecast(
        checkpoint_path=checkpoint_path,
        available_mb=4_999.0,
        min_available_ram_mb=5_000,
        headroom_factor=0.0,
        min_headroom_mb=0,
    )

    assert forecast.estimated_load_mb == 0.0
    assert forecast.required_pre_load_available_mb == 5_000.0
    assert forecast.decision == "skip"


def test_candidate_checkpoint_load_forecast_missing_file_uses_min_headroom(
    tmp_path: Path,
) -> None:
    """If the checkpoint size is unknown, use the configured minimum estimate."""
    forecast = candidate_checkpoint_load_memory_forecast(
        checkpoint_path=tmp_path / "missing.json.zst",
        available_mb=5_511.0,
        min_available_ram_mb=5_000,
        headroom_factor=60.0,
        min_headroom_mb=512,
    )

    assert forecast.checkpoint_bytes is None
    assert forecast.checkpoint_mb is None
    assert forecast.estimated_load_mb == 512.0
    assert forecast.required_pre_load_available_mb == 5_512.0
    assert forecast.decision == "skip"

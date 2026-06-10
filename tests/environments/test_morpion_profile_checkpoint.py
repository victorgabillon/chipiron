"""Tests for the standalone Morpion checkpoint profiling script."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from anemone.checkpoints import (
    checkpoint_path_for_generation,
    default_checkpoint_file_suffix,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHIPIRON_PACKAGE_ROOT = _REPO_ROOT / "src" / "chipiron"
_ATOMHEART_PACKAGE_ROOT = _REPO_ROOT.parent / "atomheart" / "src" / "atomheart"
_ANEMONE_PACKAGE_ROOT = _REPO_ROOT.parent / "anemone" / "src" / "anemone"
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "profile_morpion_checkpoint.py"

_SPEC = importlib.util.spec_from_file_location(
    "profile_morpion_checkpoint",
    _SCRIPT_PATH,
)
assert _SPEC is not None
assert _SPEC.loader is not None
profile_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(profile_module)


def _prepend_runtime_source_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Expose sibling source trees for smoke tests without poisoning sys.modules."""
    required_roots = (
        _REPO_ROOT / "src",
        _ATOMHEART_PACKAGE_ROOT.parent,
        _ANEMONE_PACKAGE_ROOT.parent,
    )
    missing_roots = [path for path in required_roots if not path.is_dir()]
    if missing_roots:
        pytest.skip(
            "Standalone profiler smoke test requires sibling source trees: "
            + ", ".join(str(path) for path in missing_roots)
        )
    for path in required_roots:
        monkeypatch.syspath_prepend(str(path))


def test_resolve_latest_runtime_checkpoint_picks_highest_generation(
    tmp_path: Path,
) -> None:
    """Latest checkpoint resolution should pick the highest generation index."""
    runtime_checkpoint_dir = tmp_path / "search_checkpoints"
    runtime_checkpoint_dir.mkdir()
    (runtime_checkpoint_dir / "generation_000002.json").write_text(
        "{}",
        encoding="utf-8",
    )
    checkpoint_path_for_generation(runtime_checkpoint_dir, 10).write_text(
        "{}",
        encoding="utf-8",
    )
    (runtime_checkpoint_dir / "generation_latest.json").write_text(
        "{}",
        encoding="utf-8",
    )

    resolved = profile_module._resolve_latest_runtime_checkpoint(runtime_checkpoint_dir)

    assert resolved == checkpoint_path_for_generation(runtime_checkpoint_dir, 10)


def test_time_call_returns_result_and_elapsed() -> None:
    """The small phase timer should return both the result and elapsed seconds."""
    result, elapsed_s = profile_module._time_call(lambda: 123)

    assert result == 123
    assert elapsed_s >= 0.0


def test_profile_script_parser_accepts_required_cli_shape(tmp_path: Path) -> None:
    """CLI parsing should accept the standalone script's supported arguments."""
    args = profile_module.build_parser().parse_args(
        [
            "--mode",
            "load",
            "--work-dir",
            str(tmp_path),
            "--checkpoint",
            str(tmp_path / "generation_000001.json"),
            "--output",
            str(tmp_path / f"profiled_checkpoint{default_checkpoint_file_suffix()}"),
            "--profile-output",
            str(tmp_path / "checkpoint.prof"),
            "--top",
            "12",
            "--dump-json",
            "--checkpoint-format",
            "json-zst",
            "--profile-full-save",
        ]
    )

    assert args.mode == "load"
    assert args.work_dir == tmp_path
    assert args.profile_mode == "full_save"
    assert args.dump_json is True
    assert args.checkpoint_format == "json-zst"
    assert args.top == 12


def test_profile_script_smoke_grow_mode_without_json_dump(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tiny grow-mode profiling run should complete and emit a .prof file."""
    _prepend_runtime_source_paths(monkeypatch)
    profile_output = tmp_path / "morpion_checkpoint.prof"

    exit_code = profile_module.main(
        [
            "--mode",
            "grow",
            "--work-dir",
            str(tmp_path),
            "--target-nodes",
            "20",
            "--growth-steps-per-batch",
            "5",
            "--profile-output",
            str(profile_output),
            "--profile-build-only",
            "--no-dump-json",
            "--top",
            "5",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert profile_output.is_file()
    assert "anemone:" in captured.out
    assert "chipiron:" in captured.out
    assert "atomheart:" in captured.out
    assert "[profile] phase=payload_build" in captured.out
    assert "[profile] phase=checkpoint_write skipped=true" in captured.out
    assert "[profile] phase=cprofile_dump" in captured.out


def test_profile_checkpoint_save_reports_encoder_compress_and_write_fields(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checkpoint profiling output should expose encoder/compress/write timings."""

    class _FakeRunner:
        _state_codec = object()

        def current_tree_size(self) -> int:
            return 3

        def _require_runtime(self) -> object:
            return SimpleNamespace(tree=SimpleNamespace(branch_count=2))

    fake_runner_module = SimpleNamespace(
        _current_rss_mb=lambda: 1.0,
        _checkpoint_node_counts=lambda _payload: (3, 1, 2),
        _log_checkpoint_metrics=lambda *_args, **_kwargs: None,
        _checkpoint_selector_state_fields=lambda _payload, prefix: {
            f"{prefix}_selector_state_present": False,
            f"{prefix}_selector_state_type": "none",
            f"{prefix}_selector_state_version": "none",
        },
        CheckpointIoMetrics=lambda **kwargs: SimpleNamespace(**kwargs),
    )

    monkeypatch.setattr(
        profile_module,
        "write_checkpoint_json_payload",
        lambda _payload, output_path: SimpleNamespace(
            output_path=Path(output_path),
            file_format="json_zst",
            encoder="orjson",
            compressed_bytes=29,
            uncompressed_bytes=941,
            jsonable_s=None,
            json_encode_s=1.25,
            compress_s=0.75,
            write_s=0.02,
            compression_ratio=29 / 941,
        ),
    )

    args = profile_module.build_parser().parse_args(
        [
            "--mode",
            "grow",
            "--work-dir",
            str(tmp_path),
            "--output",
            str(tmp_path / f"profiled_checkpoint{default_checkpoint_file_suffix()}"),
            "--dump-json",
            "--profile-build-only",
        ]
    )
    modules = {
        "runner_module": fake_runner_module,
        "build_search_checkpoint_payload": lambda runtime, state_codec: {
            "runtime": runtime,
            "state_codec": state_codec,
        },
    }

    profile_module._profile_checkpoint_save(
        args=args,
        modules=modules,
        runner=_FakeRunner(),
    )

    captured = capsys.readouterr()

    assert "[profile] phase=payload_to_jsonable skipped=true" in captured.out
    assert "encoder=orjson" in captured.out
    assert "compress_s=0.750000" in captured.out
    assert "write_s=0.020000" in captured.out

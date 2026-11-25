"""Tests for the standalone Morpion checkpoint profiling script."""

import importlib.util
import pstats
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from anemone.checkpoints import (
    checkpoint_path_for_generation,
    default_checkpoint_file_suffix,
)

from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
    MorpionBootstrapPaths,
)
from chipiron.environments.morpion.bootstrap.run_state import (
    initialize_bootstrap_run_state,
    save_bootstrap_run_state,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "profile_morpion_checkpoint.py"

_SPEC = importlib.util.spec_from_file_location(
    "profile_morpion_checkpoint",
    _SCRIPT_PATH,
)
assert _SPEC is not None
assert _SPEC.loader is not None
profile_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(profile_module)


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
    args = profile_module.build_parser().parse_args([
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
    ])

    assert args.mode == "load"
    assert args.work_dir == tmp_path
    assert args.profile_mode == "full_save"
    assert args.dump_json is True
    assert args.checkpoint_format == "json-zst"
    assert args.top == 12


def test_profile_script_smoke_grow_mode_without_json_dump(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tiny grow-mode profiling run should complete and emit a .prof file."""
    profile_output = tmp_path / "morpion_checkpoint.prof"

    exit_code = profile_module.main([
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
    ])

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
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checkpoint profiling output should expose encoder/compress/write timings."""

    class _FakeRunner:
        _state_codec = object()

        def current_tree_size(self) -> int:
            return 3

        def _require_runtime(self) -> object:
            return SimpleNamespace(tree=SimpleNamespace(branch_count=2))

    fake_checkpoint_io_module = SimpleNamespace(
        _checkpoint_node_counts=lambda _payload: (3, 1, 2),
        _log_checkpoint_metrics=lambda *_args, **_kwargs: None,
        CheckpointIoMetrics=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    fake_restore_memory_logging_module = SimpleNamespace(current_rss_mb=lambda: 1.0)
    fake_selection_logging_module = SimpleNamespace(
        checkpoint_selector_state_fields=lambda _payload, prefix: {
            f"{prefix}_selector_state_present": False,
            f"{prefix}_selector_state_type": "none",
            f"{prefix}_selector_state_version": "none",
        },
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

    args = profile_module.build_parser().parse_args([
        "--mode",
        "grow",
        "--work-dir",
        str(tmp_path),
        "--output",
        str(tmp_path / f"profiled_checkpoint{default_checkpoint_file_suffix()}"),
        "--profile-output",
        str(tmp_path / "morpion_checkpoint_profile.prof"),
        "--dump-json",
        "--profile-build-only",
    ])
    modules = {
        "checkpoint_io_module": fake_checkpoint_io_module,
        "restore_memory_logging_module": fake_restore_memory_logging_module,
        "runner_module": SimpleNamespace(),
        "selection_logging_module": fake_selection_logging_module,
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


@pytest.fixture
def restore_calls(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Exercise the real CLI dispatch and loader without a scientific workload."""
    calls: list[Path] = []

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Restore-only must not build, write, grow or inspect the tree")

    class _FakeRunner:
        def load_or_create(
            self, *, tree_snapshot_path: Path, model_bundle_path: None
        ) -> None:
            """Record a read or propagate a simulated corrupt-checkpoint failure."""
            assert model_bundle_path is None
            calls.append(tree_snapshot_path)
            payload = (
                tree_snapshot_path / "manifest.json"
                if tree_snapshot_path.is_dir()
                else tree_snapshot_path
            )
            if payload.read_bytes() == b"corrupt":
                message = "Corrupt checkpoint"
                raise ValueError(message)

        grow = forbidden
        _require_runtime = forbidden

    modules = {
        "AnemoneMorpionSearchRunner": _FakeRunner,
        "MorpionBootstrapPaths": MorpionBootstrapPaths,
        "restore_memory_logging_module": SimpleNamespace(current_rss_mb=lambda: 12.0),
        "build_search_checkpoint_payload": forbidden,
        "chipiron": SimpleNamespace(__file__="fake-chipiron"),
        "anemone": SimpleNamespace(__file__="fake-anemone"),
        "atomheart": SimpleNamespace(__file__="fake-atomheart"),
    }
    monkeypatch.setattr(profile_module, "_load_runtime_modules", lambda: modules)
    monkeypatch.setattr(profile_module, "_profile_checkpoint_save", forbidden)
    monkeypatch.setattr(profile_module, "write_checkpoint_json_payload", forbidden)
    return calls


@pytest.mark.parametrize("layout", ["explicit-file", "latest", "explicit-directory"])
def test_restore_only_cli_stops_before_payload_construction(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    restore_calls: list[Path],
    layout: str,
) -> None:
    """File/directory inputs use the existing loader without building a payload."""
    checkpoints = tmp_path / "search_checkpoints"
    checkpoints.mkdir()
    checkpoint = checkpoint_path_for_generation(checkpoints, 1)
    payload = checkpoint
    if layout == "explicit-directory":
        checkpoint = checkpoints / "sharded"
        checkpoint.mkdir()
        payload = checkpoint / "manifest.json"
    payload.write_bytes(b"fixture")
    before = payload.stat()
    output = tmp_path / "profiles" / "restore.prof"
    args = [
        "--mode",
        "load",
        "--restore-only",
        "--work-dir",
        str(tmp_path),
        "--profile-output",
        str(output),
        "--top",
        "1",
    ]
    if layout != "latest":
        args.extend(["--checkpoint", str(checkpoint)])
    else:
        save_bootstrap_run_state(
            replace(
                initialize_bootstrap_run_state(),
                latest_runtime_checkpoint_path=str(checkpoint.relative_to(tmp_path)),
            ),
            MorpionBootstrapPaths.from_work_dir(tmp_path).run_state_path,
        )

    assert profile_module.main(args) == 0
    assert restore_calls == [checkpoint]
    assert payload.read_bytes() == b"fixture"
    assert payload.stat().st_mtime_ns == before.st_mtime_ns
    assert output.is_file()
    assert pstats.Stats(str(output)).total_calls > 0
    text = capsys.readouterr().out
    assert "phase=restore elapsed_s=" in text
    assert "rss_before_restore_mb=12.000000 rss_after_restore_mb=12.000000" in text
    assert "phase=payload_build skipped=true" in text
    assert "phase=checkpoint_write skipped=true" in text


@pytest.mark.parametrize("failure", ["missing", "corrupt", "existing-profile"])
def test_restore_only_does_not_hide_load_errors_or_overwrite_evidence(
    tmp_path: Path, restore_calls: list[Path], failure: str
) -> None:
    """Missing/corrupt input fails; an existing report is never replaced."""
    checkpoint = tmp_path / "checkpoint.json"
    output = tmp_path / "restore.prof"
    if failure != "missing":
        checkpoint.write_bytes(b"corrupt" if failure == "corrupt" else b"fixture")
    if failure == "existing-profile":
        output.write_bytes(b"prior evidence")
    expected = {
        "missing": FileNotFoundError,
        "corrupt": ValueError,
        "existing-profile": FileExistsError,
    }[failure]
    with pytest.raises(expected):
        profile_module.main([
            "--mode",
            "load",
            "--restore-only",
            "--work-dir",
            str(tmp_path),
            "--checkpoint",
            str(checkpoint),
            "--profile-output",
            str(output),
        ])
    assert restore_calls == ([checkpoint] if failure == "corrupt" else [])
    if failure == "existing-profile":
        assert output.read_bytes() == b"prior evidence"
    else:
        assert not output.exists()


@pytest.mark.parametrize(
    "options",
    [
        ["--mode", "grow"],
        ["--mode", "load", "--dump-json"],
        ["--mode", "load", "--dump-zstd"],
        ["--mode", "load", "--output", "forbidden.json"],
        ["--mode", "load", "--profile-full-save"],
    ],
)
def test_restore_only_rejects_conflicting_options_before_runtime_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, options: list[str]
) -> None:
    """Conflicting modes must fail before importing or restoring any runtime."""

    def forbidden() -> None:
        pytest.fail("Invalid restore-only arguments must fail before imports")

    monkeypatch.setattr(profile_module, "_load_runtime_modules", forbidden)
    with pytest.raises(SystemExit) as error:
        profile_module.main(["--work-dir", str(tmp_path), "--restore-only", *options])
    assert error.value.code == 2


def test_load_without_restore_only_keeps_existing_payload_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restore_calls: list[Path]
) -> None:
    """The new opt-in mode does not alter the existing load/save profiling path."""
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_bytes(b"fixture")
    saved: list[object] = []
    monkeypatch.setattr(
        profile_module,
        "_profile_checkpoint_save",
        lambda **kwargs: saved.append(kwargs),
    )
    assert (
        profile_module.main([
            "--mode",
            "load",
            "--work-dir",
            str(tmp_path),
            "--checkpoint",
            str(checkpoint),
        ])
        == 0
    )
    assert restore_calls == [checkpoint]
    assert len(saved) == 1


@pytest.mark.parametrize(
    "state_kind",
    ["committed", "sharded-only", "uncommitted", "missing", "malformed", "unset"],
)
def test_restore_only_implicit_selection_uses_committed_sharded_generation(
    tmp_path: Path, restore_calls: list[Path], state_kind: str
) -> None:
    """G40 must not silently replace committed G41 or ambiguous workspace state."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.runtime_checkpoint_dir.mkdir()
    if state_kind != "sharded-only":
        checkpoint_path_for_generation(paths.runtime_checkpoint_dir, 40).write_bytes(
            b"fixture"
        )
    checkpoint = paths.runtime_checkpoint_dir / "generation_000041.sharded"
    if state_kind != "missing":
        checkpoint.mkdir()
        (checkpoint / "manifest.json").write_bytes(b"fixture")
    if state_kind != "uncommitted":
        save_bootstrap_run_state(
            replace(
                initialize_bootstrap_run_state(),
                generation=41,
                latest_runtime_checkpoint_path=(
                    None
                    if state_kind == "unset"
                    else str(checkpoint.relative_to(tmp_path))
                ),
            ),
            paths.run_state_path,
        )
    if state_kind == "malformed":
        paths.run_state_path.write_text("{", encoding="utf-8")
    output = tmp_path / "restore.prof"
    args = [
        "--mode",
        "load",
        "--restore-only",
        "--work-dir",
        str(tmp_path),
        "--profile-output",
        str(output),
    ]
    if state_kind in {"committed", "sharded-only"}:
        assert profile_module.main(args) == 0
        assert restore_calls == [checkpoint]
    else:
        expected = FileNotFoundError if state_kind == "missing" else ValueError
        with pytest.raises(expected):
            profile_module.main(args)
        assert restore_calls == []
        assert not output.exists()


@pytest.mark.parametrize("rival_kind", ["file", "symlink"])
def test_restore_only_preserves_profile_created_during_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    restore_calls: list[Path],
    rival_kind: str,
) -> None:
    """Exclusive publication preserves a competing file or symlink byte for byte."""
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_bytes(b"fixture")
    output = tmp_path / "restore.prof"
    evidence = tmp_path / "other.prof"
    original_loader = profile_module._load_runner_for_mode

    def load_and_publish(*args: object) -> object:
        result = original_loader(*args)
        if rival_kind == "symlink":
            evidence.write_bytes(b"concurrent evidence")
            output.symlink_to(evidence)
        else:
            output.write_bytes(b"concurrent evidence")
        return result

    monkeypatch.setattr(profile_module, "_load_runner_for_mode", load_and_publish)
    with pytest.raises(FileExistsError):
        profile_module.main([
            "--mode",
            "load",
            "--restore-only",
            "--work-dir",
            str(tmp_path),
            "--checkpoint",
            str(checkpoint),
            "--profile-output",
            str(output),
        ])
    assert restore_calls == [checkpoint]
    assert output.read_bytes() == b"concurrent evidence"
    assert output.is_symlink() == (rival_kind == "symlink")


def test_load_without_restore_only_keeps_implicit_file_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restore_calls: list[Path]
) -> None:
    """Ordinary load retains its existing file selection without run state."""
    checkpoints = tmp_path / "search_checkpoints"
    checkpoints.mkdir()
    checkpoint = checkpoint_path_for_generation(checkpoints, 40)
    checkpoint.write_bytes(b"fixture")
    (checkpoints / "generation_000041.sharded").mkdir()
    saved: list[object] = []
    monkeypatch.setattr(
        profile_module,
        "_profile_checkpoint_save",
        lambda **kwargs: saved.append(kwargs),
    )
    assert profile_module.main(["--mode", "load", "--work-dir", str(tmp_path)]) == 0
    assert restore_calls == [checkpoint]
    assert len(saved) == 1

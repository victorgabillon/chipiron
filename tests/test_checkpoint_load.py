"""Focused tests for checkpoint payload file I/O helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import anemone.checkpoints.io as checkpoint_io_module
from anemone.checkpoints import (
    checkpoint_path_for_generation,
    default_checkpoint_file_suffix,
    load_checkpoint_json_payload,
    parse_generation_checkpoint_name,
    resolve_latest_generation_checkpoint_path,
    write_checkpoint_json_payload,
)

from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
    prune_generation_files,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_checkpoint_json_payload_roundtrip_supports_plain_and_compressed(
    tmp_path: Path,
) -> None:
    """Checkpoint JSON payload helpers should round-trip both supported save forms."""
    payload = {"generation": 12, "nodes": [{"id": 1}, {"id": 2}]}
    plain_path = tmp_path / "tree_checkpoint.json"
    compressed_path = tmp_path / f"tree_checkpoint{default_checkpoint_file_suffix()}"

    plain_stats = write_checkpoint_json_payload(payload, plain_path)
    compressed_stats = write_checkpoint_json_payload(payload, compressed_path)

    plain_loaded, plain_read_stats = load_checkpoint_json_payload(plain_path)
    compressed_loaded, compressed_read_stats = load_checkpoint_json_payload(
        compressed_path
    )

    assert plain_loaded == payload
    assert compressed_loaded == payload
    assert plain_stats.compressed_bytes == plain_read_stats.compressed_bytes
    assert compressed_stats.compressed_bytes == compressed_read_stats.compressed_bytes
    assert compressed_stats.uncompressed_bytes == plain_stats.uncompressed_bytes
    assert compressed_stats.encoder in ("orjson", "stdlib")


def test_checkpoint_zstd_write_avoids_streaming_writer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Compressed writes should encode once and compress once, without stream writes."""

    class _FakeZstdCompressor:
        def compress(self, data: bytes) -> bytes:
            return b"zstd:" + data

        def stream_writer(self, _handle: object) -> object:
            raise AssertionError("stream_writer should not be used")

    fake_zstandard = SimpleNamespace(
        ZstdCompressor=lambda: _FakeZstdCompressor(),
        ZstdDecompressor=lambda: None,
    )

    monkeypatch.setattr(checkpoint_io_module, "_zstandard_module", fake_zstandard)
    output_path = tmp_path / "unused.json.zst"
    compressed_stats = checkpoint_io_module.write_checkpoint_json_payload(
        {"generation": 12, "nodes": [{"id": 1}]},
        output_path,
    )

    assert compressed_stats.file_format == "json_zst"
    assert compressed_stats.compress_s is not None
    assert compressed_stats.write_s >= 0.0


def test_latest_generation_checkpoint_resolution_handles_mixed_suffixes(
    tmp_path: Path,
) -> None:
    """Latest-checkpoint discovery should work across plain and compressed files."""
    older_path = tmp_path / "generation_000002.json"
    newer_path = checkpoint_path_for_generation(tmp_path, 10)
    ignored_path = tmp_path / "generation_latest.json"

    write_checkpoint_json_payload({"generation": 2}, older_path)
    write_checkpoint_json_payload({"generation": 10}, newer_path)
    ignored_path.write_text("{}", encoding="utf-8")

    assert parse_generation_checkpoint_name(older_path) == 2
    assert parse_generation_checkpoint_name(newer_path) == 10
    assert parse_generation_checkpoint_name(ignored_path) is None
    assert resolve_latest_generation_checkpoint_path(tmp_path) == newer_path


def test_prune_generation_files_keeps_latest_generations_across_mixed_formats(
    tmp_path: Path,
) -> None:
    """Retention should prune by generation, even when formats are mixed."""
    old_plain_path = tmp_path / "generation_000001.json"
    old_compressed_path = checkpoint_path_for_generation(tmp_path, 1)
    latest_path = checkpoint_path_for_generation(tmp_path, 2)

    write_checkpoint_json_payload({"generation": 1, "kind": "plain"}, old_plain_path)
    write_checkpoint_json_payload(
        {"generation": 1, "kind": "compressed"},
        old_compressed_path,
    )
    write_checkpoint_json_payload({"generation": 2}, latest_path)

    prune_generation_files(tmp_path, keep_latest=1)

    assert not old_plain_path.exists()
    assert not old_compressed_path.exists()
    assert latest_path.exists()

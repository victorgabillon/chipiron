"""Exact scientific bytes and fail-closed framing under self-consistent manifests."""

from __future__ import annotations

import hashlib
import io
import json
from typing import TYPE_CHECKING, Any

import pytest
import zstandard as zstd

from chipiron.environments.morpion.bootstrap.closed_scientific_snapshot import (
    load_closed_scientific_snapshot,
    save_closed_scientific_snapshot,
)
from chipiron.environments.morpion.bootstrap.scientific_stream import (
    ENCODING,
    ScientificRowsWriter,
    _decoded_blocks,
    open_scientific_stream,
)
from chipiron.environments.morpion.bootstrap.sharded_training_reader import (
    read_closed_manifest,
)
from tests.environments.test_morpion_closed_scientific_snapshot import (
    assert_reference,
    closed,
    view,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_exact_bytes_determinism_reference_and_cache(tmp_path: Path) -> None:
    """All JSONL bytes, full scientific fields, cursor and patch values agree."""
    raw = closed(tmp_path / "raw", created_at_unix_s=123.5, metadata={"extra": None})
    compressed = closed(
        tmp_path / "compressed",
        encoding=ENCODING,
        created_at_unix_s=123.5,
        metadata={"extra": None},
    )
    repeat = closed(
        tmp_path / "repeat",
        encoding=ENCODING,
        created_at_unix_s=123.5,
        metadata={"extra": None},
    )
    assert {p.name: p.read_bytes() for p in compressed.parent.iterdir()} == {
        p.name: p.read_bytes() for p in repeat.parent.iterdir()
    }
    assert set(p.name for p in compressed.parent.iterdir()) == {
        "generation.json",
        "records.jsonl.zst",
        "updates.jsonl.zst",
    }
    header = read_closed_manifest(compressed)
    for key, name in (
        ("record_shard", "records.jsonl"),
        ("update_shard", "updates.jsonl"),
    ):
        descriptor = header[key]
        with open_scientific_stream(
            compressed.parent / descriptor["path"], descriptor
        ) as stream:
            assert stream.read() == (raw.parent / name).read_bytes()
    assert load_closed_scientific_snapshot(raw) == load_closed_scientific_snapshot(
        compressed
    )
    assert_reference(compressed)
    with view(compressed) as cached, view(raw) as original:
        assert cached.index_reused
        assert cached.identity != original.identity
        cached.validate_source_identity()


@pytest.mark.parametrize(
    "damage",
    [
        "truncated_header",
        "truncated_checksum",
        "checksum",
        "invalid",
        "trailing",
        "concatenated",
        "wrong_encoding",
        "wrong_hash",
        "wrong_decoded_hash",
        "wrong_bytes",
        "wrong_decoded_bytes",
        "huge_declared",
        "wrong_count",
        "utf8",
        "json",
        "duplicate",
        "missing",
        "canonical",
        "contract",
        "raw_as_zstd",
        "zstd_as_raw",
        "window",
        "no_checksum",
        "reserved_block",
    ],
)
def test_corruption_rejected_by_reference_and_index(
    tmp_path: Path, damage: str
) -> None:
    """Updating the physical hash cannot conceal invalid framing or scientific data."""
    manifest = closed(tmp_path / "source", encoding=ENCODING)
    header = read_closed_manifest(manifest)
    descriptor = header["record_shard"]
    path = manifest.parent / descriptor["path"]
    with open_scientific_stream(path, descriptor) as stream:
        decoded = stream.read()
    encoded = path.read_bytes()
    if damage == "truncated_header":
        encoded = encoded[:5]
    elif damage == "truncated_checksum":
        encoded = encoded[:-1]
    elif damage == "checksum":
        encoded = encoded[:-1] + bytes([encoded[-1] ^ 1])
    elif damage == "invalid":
        encoded = encoded[:10] + b"\xff" * (len(encoded) - 10)
    elif damage == "trailing":
        encoded += b"trailing"
    elif damage == "concatenated":
        encoded += encoded
    elif damage == "wrong_encoding":
        descriptor["encoding"] = "unknown"
    elif damage == "wrong_decoded_hash":
        descriptor["decoded_sha256"] = "0" * 64
    elif damage == "wrong_decoded_bytes":
        descriptor["decoded_bytes"] -= 1
    elif damage == "huge_declared":
        descriptor["decoded_bytes"] = 10**100
    elif damage == "wrong_count":
        descriptor["count"] += 1
    elif damage == "contract":
        header["scientific_contract"] = "unknown"
    elif damage == "raw_as_zstd":
        encoded = decoded
    elif damage == "zstd_as_raw":
        for key in ("encoding", "bytes", "decoded_bytes", "decoded_sha256"):
            descriptor.pop(key)
        descriptor["path"] = "records.jsonl"
        path = path.with_suffix("")
    elif damage == "window":
        encoded = encoded[:5] + b"\xf8" + encoded[6:]
    elif damage == "no_checksum":
        encoded = zstd.ZstdCompressor().compress(decoded)
    elif damage == "reserved_block":
        encoded = encoded[:6] + b"\x07\x00\x00" + encoded[9:]
    elif damage in ("utf8", "json", "duplicate", "missing", "canonical"):
        lines = decoded.splitlines(keepends=True)
        if damage == "utf8":
            lines[1] = b"\xff\n"
        elif damage == "json":
            lines[1] = b"{\n"
        elif damage == "duplicate":
            lines[1] = lines[1].replace(b"{", b'{"node_id":"wrong",', 1)
        else:
            row = json.loads(lines[1])
            if damage == "missing":
                row.pop("node_id")
            else:
                row["state_ref_payload"] = None
            lines[1] = (json.dumps(row) + "\n").encode()
        output = io.BytesIO()
        writer = ScientificRowsWriter(output, ENCODING)
        writer.write(b"".join(lines))
        descriptor.update(writer.finish(descriptor["path"], 1, descriptor["count"]))
        encoded = output.getvalue()
    path.write_bytes(encoded)
    descriptor["sha256"] = (
        "0" * 64 if damage == "wrong_hash" else hashlib.sha256(encoded).hexdigest()
    )
    if "encoding" in descriptor:
        descriptor["bytes"] = len(encoded) + (1 if damage == "wrong_bytes" else 0)
    manifest.write_text(json.dumps(header))
    for reader in (load_closed_scientific_snapshot, view):
        with pytest.raises((ValueError, TypeError, KeyError, UnicodeError)):
            reader(manifest)


def test_bounded_high_ratio_decode(tmp_path: Path) -> None:
    """Large actual/declared contents never dictate decoder allocation size."""
    output = io.BytesIO()
    writer = ScientificRowsWriter(output, ENCODING)
    for _ in range(128):
        writer.write(b"x" * (128 * 1024) + b"\n")
    descriptor = writer.finish("x", 1, 127)
    chunks = _decoded_blocks(io.BytesIO(output.getvalue()), descriptor)
    total = 0
    for chunk in chunks:
        assert len(chunk) <= 128 * 1024
        total += len(chunk)
    assert total == descriptor["decoded_bytes"]


def test_repeated_committed_compressed_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inherited encoding reuses exact canonical records without replaying old states."""
    from atomheart.games.morpion.checkpoints import MorpionStateCheckpointCodec

    from tests.environments.morpion_durable_closed_helpers import (
        setup_closed_baseline,
    )
    from tests.environments.morpion_durable_v2_helpers import (
        scientific_nodes,
        scientific_producer,
    )
    from tests.environments.morpion_patch_receipt_helpers import BASELINE

    args, paths, state = setup_closed_baseline(tmp_path / "run", encoding=ENCODING)
    source = paths.work_dir / state.latest_tree_snapshot_path

    def forbidden(*a: Any, **k: Any) -> Any:
        pytest.fail("unchanged canonical state decoded")

    monkeypatch.setattr(MorpionStateCheckpointCodec, "load_state_ref", forbidden)
    result, stats = save_closed_scientific_snapshot(
        nodes=scientific_nodes(BASELINE),
        root_node_id="root",
        destination=tmp_path / "next",
        generation=2,
        producer=scientific_producer(args),
        committed_source=source,
    )
    assert stats["new_state_encodings"] == 0
    assert stats["copied_records"] == len(BASELINE)
    assert read_closed_manifest(result)["record_shard"]["encoding"] == ENCODING
    with pytest.raises(ValueError, match="encoding"):
        closed(tmp_path / "bad", encoding="unsupported")


def test_interrupted_compressed_index_is_disposable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted transaction is never reused as a complete scientific view."""
    from chipiron.environments.morpion.bootstrap import (
        sharded_training_reader as reader,
    )

    manifest = closed(tmp_path / "source", encoding=ENCODING)
    original = reader.iter_shard_rows

    def interrupted(*args: Any, **kwargs: Any) -> Any:
        for row in original(*args, **kwargs):
            yield row
            message = "interrupted index"
            raise RuntimeError(message)

    with monkeypatch.context() as patched:
        patched.setattr(reader, "iter_shard_rows", interrupted)
        with pytest.raises(RuntimeError, match="interrupted index"):
            view(manifest)
    assert not list((tmp_path / "cache").glob("v*.sqlite3"))
    with view(manifest) as rebuilt:
        assert not rebuilt.index_reused
    with view(manifest) as cached:
        assert cached.index_reused


def test_refreshed_accepted_model_has_exact_raw_compressed_science(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Encode the same genuinely activated M2 graph and producer in fresh fixtures."""
    from chipiron.environments.morpion.bootstrap.model_activation import (
        _runner,
        activate_model,
        prepare_model_activation_request,
    )
    from tests.environments.morpion_activation_helpers import setup_activation
    from tests.environments.morpion_fenced_dataset_helpers import local_leaderboard

    local_leaderboard(monkeypatch, tmp_path)
    _, paths, _ = setup_activation(tmp_path / "run", encoding=ENCODING)
    state = activate_model(
        paths, prepare_model_activation_request(paths, training_generation=1)
    )
    head = paths.work_dir / state.latest_tree_snapshot_path
    header = read_closed_manifest(head)
    runner, effective = _runner(paths, state)
    runner.load_or_create(
        paths.work_dir / state.latest_runtime_checkpoint_path, None, effective
    )
    runner.validate_model_refresh_certificates()
    exports = []
    for name, encoding in (("raw", None), ("compressed", ENCODING)):
        exports.append(
            runner.export_closed_scientific_snapshot(
                tmp_path / name,
                generation=1,
                producer=header["producer"]["identity"],
                encoding=encoding,
            )
        )
    assert load_closed_scientific_snapshot(
        exports[0]
    ) == load_closed_scientific_snapshot(exports[1])
    for key in ("record_shard", "update_shard"):
        decoded = []
        for path in exports:
            descriptor = read_closed_manifest(path)[key]
            with open_scientific_stream(
                path.parent / descriptor["path"], descriptor
            ) as stream:
                decoded.append(stream.read())
        assert decoded[0] == decoded[1]
    assert_reference(exports[1])
    runner.discard_uncommitted_runtime()

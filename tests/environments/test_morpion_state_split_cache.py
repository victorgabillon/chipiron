"""Exact split artifacts, real process exclusion, interruption and publication faults."""

import json
import multiprocessing
import os
import shutil
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chipiron.environments.morpion.players.evaluators.neural_networks.training import (
    MorpionTrainingArgs,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training import (
    state_split as split,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training import (
    state_split_cache as cache,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training.state_split_cache_identity import (
    ENVIRONMENT,
)
from tests.environments.test_morpion_continual_training import rows_file


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Small protocol tests use an explicit semantic fence; installed checks use real hashes."""
    monkeypatch.delenv(ENVIRONMENT, raising=False)
    monkeypatch.setattr(cache, "semantic_fingerprint", lambda: "a" * 64)
    cache._cached.cache_clear()
    split._split.cache_clear()


def invoke(
    root: Path,
    rows: Path,
    *,
    fraction: float = 0.2,
    seed: int = 0,
    limit: int | None = None,
) -> cache.SplitValue:
    """Exercise real cache I/O and the unchanged native implementation."""
    return cache.load_or_build(
        root, rows, fraction, seed, limit, split.POLICY, split._uncached_value
    )


def entry(root: Path) -> Path:
    """Locate the one completed test artifact."""
    return next(root.glob("*/slot-*/entry"))


def observation(root: Path) -> dict[str, Any]:
    """Read bounded operational evidence rather than scientific metadata."""
    return json.loads(next(root.glob("*/slot-*/operation.json")).read_text())


@pytest.mark.fast
@pytest.mark.parametrize("limit", [None, 0, 1, 9, 16, 25])
@pytest.mark.parametrize("fraction,seed", [(0.2, 0), (0.3, 17), (0.0, -1), (1.0, 42)])
def test_exact_native_cold_memory_and_relocated_hit(
    tmp_path: Path, limit: int | None, fraction: float, seed: int
) -> None:
    """Every supported limit and policy input retains exact population and membership."""
    rows = rows_file(tmp_path / "input")
    root = tmp_path / "cache"
    expected = split._uncached_value(
        str(rows), cache.digest(rows), fraction, seed, limit
    )
    assert invoke(root, rows, fraction=fraction, seed=seed, limit=limit) == expected
    assert observation(root)["outcome"] == "miss"
    assert invoke(root, rows, fraction=fraction, seed=seed, limit=limit) == expected
    assert observation(root)["outcome"] == "memory-hit"
    relocated = tmp_path / "relocated" / "rows.jsonl"
    relocated.parent.mkdir()
    shutil.copyfile(rows, relocated)
    assert (
        invoke(root, relocated, fraction=fraction, seed=seed, limit=limit) == expected
    )
    assert observation(root)["outcome"] == "hit"


@pytest.mark.fast
def test_opt_in_preserves_default_args_metadata_and_algorithm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Neither operational configuration nor evidence enters historical scientific identities."""
    rows = rows_file(tmp_path / "input")
    args = MorpionTrainingArgs(rows, tmp_path / "model", stable_validation_split=True)
    original = asdict(args)
    native = split.prepare_state_split(args)
    assert split._split.cache_info().misses == 1
    assert split.prepare_state_split(args) == native
    assert split._split.cache_info().hits == 1
    monkeypatch.setenv(ENVIRONMENT, str(tmp_path / "cache"))
    assert split.prepare_state_split(args) == native
    assert split.prepare_state_split(args) == native
    assert asdict(args) == original
    disabled = replace(args, stable_validation_split=False)
    monkeypatch.setenv(ENVIRONMENT, "")
    assert split.prepare_state_split(disabled) == (disabled, {})
    with pytest.raises(ValueError, match="must name"):
        split.prepare_state_split(args)


@pytest.mark.parametrize(
    "change", ["target", "source", "order", "duplicate", "implementation"]
)
def test_changed_identity_never_reuses_old_population(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    """Changed labels/order/codec evidence miss; duplicates remain separate rows."""
    monkeypatch.setattr(cache, "SLOTS", 1)
    rows = rows_file(tmp_path / "input")
    root = tmp_path / "cache"
    before = invoke(root, rows, fraction=1.0)
    records = [json.loads(line) for line in rows.read_text().splitlines()]
    if change == "implementation":
        monkeypatch.setattr(cache, "semantic_fingerprint", lambda: "b" * 64)
    elif change == "target":
        records[1]["row"]["target_value"] += 7
    elif change == "source":
        records[1]["row"]["metadata"]["target_source"] = "new-source"
    elif change == "order":
        records[1:] = reversed(records[1:])
    else:
        records.append(records[1])
    rows.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    after = invoke(root, rows, fraction=1.0)
    assert observation(root)["outcome"] == "miss"
    assert after == split._uncached_value(str(rows), cache.digest(rows), 1.0, 0, None)
    if change in {"target", "source", "duplicate"}:
        assert after[2] != before[2]
    assert len(list(root.glob("*/slot-*/entry"))) == 1


@pytest.mark.parametrize(
    "damage",
    [
        "checksum",
        "missing",
        "identity",
        "indices",
        "population",
        "count",
        "duplicate-key",
        "large",
    ],
)
def test_completed_corruption_is_rejected(tmp_path: Path, damage: str) -> None:
    """Malformed completed artifacts fail; they are not silently rebuilt or consumed."""
    rows = rows_file(tmp_path / "input")
    root = tmp_path / "cache"
    invoke(root, rows)
    cache._cached.cache_clear()
    target = entry(root)
    payload = target / "split.json"
    manifest_path = target / "complete.json"
    manifest = json.loads(manifest_path.read_text())
    value = json.loads(payload.read_text())
    if damage == "missing":
        payload.unlink()
    elif damage == "identity":
        manifest["identity"]["semantics"] = "b" * 64
        manifest_path.write_text(json.dumps(manifest))
    else:
        if damage == "indices":
            value["indices"] = [value["rows"]]
        if damage == "population":
            value["population"] = "not-a-digest"
        if damage == "count":
            value["rows"] = True
        raw = cache.encoded(value)
        if damage == "checksum":
            raw += b" "
        if damage == "duplicate-key":
            raw = b'{"rows":1,' + raw[1:]
        if damage == "large":
            raw = b" " * (cache.MAX_PAYLOAD_BYTES + 1)
        payload.write_bytes(raw)
        if damage != "checksum":
            manifest["payload_sha256"] = cache.digest(payload)
            manifest_path.write_text(json.dumps(manifest))
    with pytest.raises((ValueError, FileNotFoundError)):
        invoke(root, rows)
    assert target.exists()


@pytest.mark.parametrize(
    "failure", ["write", "split.json", "complete.json", "attempt", "publish", "slot"]
)
def test_write_fsync_and_publication_failures_propagate_and_recover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Actual file/directory fsync and rename faults never return an unverified result."""
    rows = rows_file(tmp_path / "input")
    root = tmp_path / "cache"
    cache._root(root, rows)
    real_sync, real_rename, real_write = os.fsync, os.rename, cache._write

    def fail_sync(fd: int) -> None:
        path = Path(os.readlink(f"/proc/self/fd/{fd}"))
        if path.name == failure or (
            failure in {"attempt", "slot"} and path.name.startswith(failure + "-")
        ):
            message = "injected fsync failure"
            raise OSError(message)
        real_sync(fd)

    def fail_rename(source: Path, destination: Path) -> None:
        if failure == "publish" and source.name.startswith("attempt-"):
            message = "injected publication failure"
            raise OSError(message)
        real_rename(source, destination)

    def fail_write(path: Path, data: bytes) -> None:
        if failure == "write" and path.name == "split.json":
            path.write_bytes(data[:8])
            message = "injected partial write"
            raise OSError(message)
        real_write(path, data)

    with monkeypatch.context() as patch:
        patch.setattr(cache, "_write", fail_write)
        patch.setattr(cache.os, "fsync", fail_sync)
        patch.setattr(cache.os, "rename", fail_rename)
        with pytest.raises(OSError, match="injected"):
            invoke(root, rows)
    assert invoke(root, rows) == split._uncached_value(
        str(rows), cache.digest(rows), 0.2, 0, None
    )
    assert not list(root.glob("*/slot-*/attempt-*"))


def process_lookup(
    root: Path, rows: Path, queue: Any, ready: Path | None = None
) -> None:
    """Spawn target, with observable native-build entry and no inherited LRU."""
    cache.semantic_fingerprint = lambda: "a" * 64
    write = cache._write

    def interrupted_write(path: Path, data: bytes) -> None:
        write(path, data)
        if ready is not None and path.name == "split.json":
            ready.touch()
            time.sleep(60)

    cache._write = interrupted_write

    def build(
        path: str, digest: str, fraction: float, seed: int, limit: int | None
    ) -> cache.SplitValue:
        with (root.parent / "builds.txt").open("a") as stream:
            stream.write(f"{os.getpid()}\n")
        return split._uncached_value(path, digest, fraction, seed, limit)

    queue.put(cache.load_or_build(root, rows, 0.2, 0, None, split.POLICY, build))


def test_spawned_builders_and_readers_share_one_complete_artifact(
    tmp_path: Path,
) -> None:
    """Two cold builders and subsequent fresh readers agree without duplicate writes."""
    rows = rows_file(tmp_path / "input")
    root = tmp_path / "cache"
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    expected = split._uncached_value(str(rows), cache.digest(rows), 0.2, 0, None)
    for _ in range(2):
        children = [
            ctx.Process(target=process_lookup, args=(root, rows, queue))
            for _ in range(2)
        ]
        try:
            for child in children:
                child.start()
            assert [queue.get(timeout=30) for _ in children] == [expected, expected]
            for child in children:
                child.join(30)
                assert child.exitcode == 0
        finally:
            for child in children:
                if child.is_alive():
                    child.kill()
                child.join()
    queue.close()
    queue.join_thread()
    assert len((tmp_path / "builds.txt").read_text().splitlines()) == 1


def test_killed_builder_is_reaped_before_abandoned_attempt_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SIGKILL releases the kernel lock; retry removes only the abandoned private build."""
    rows = rows_file(tmp_path / "input")
    root, ready = tmp_path / "cache", tmp_path / "ready"
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    child = ctx.Process(target=process_lookup, args=(root, rows, queue, ready))
    child.start()
    try:
        deadline = time.monotonic() + 30
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert ready.exists() and child.is_alive()
        assert len(list(root.glob("*/slot-*/attempt-*"))) == 1
        assert len(list(root.glob("*/slot-*/attempt-*/split.json"))) == 1
        with monkeypatch.context() as patch:
            patch.setattr(cache, "LOCK_TIMEOUT_SECONDS", 0.05)
            with pytest.raises(TimeoutError):
                invoke(root, rows)
        assert len(list(root.glob("*/slot-*/attempt-*/split.json"))) == 1
    finally:
        if child.is_alive():
            child.kill()
        child.join(30)
        queue.close()
        queue.join_thread()
    assert child.exitcode is not None
    assert invoke(root, rows) == split._uncached_value(
        str(rows), cache.digest(rows), 0.2, 0, None
    )
    assert not list(root.glob("*/slot-*/attempt-*"))


def test_source_change_and_oversize_never_change_scientific_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation rejects publication; storage bounds skip caching, not training rows."""
    rows = rows_file(tmp_path / "input")
    root = tmp_path / "cache"

    def changing(
        path: str, digest: str, fraction: float, seed: int, limit: int | None
    ) -> cache.SplitValue:
        result = split._uncached_value(path, digest, fraction, seed, limit)
        rows.write_text(rows.read_text() + "\n")
        return result

    with pytest.raises(ValueError, match="source changed"):
        cache.load_or_build(root, rows, 0.2, 0, None, split.POLICY, changing)
    assert not list(root.glob("*/slot-*/entry"))
    monkeypatch.setattr(cache, "MAX_INDICES", 0)
    expected = split._uncached_value(str(rows), cache.digest(rows), 1.0, 0, None)
    assert invoke(root, rows, fraction=1.0) == expected
    assert observation(root)["outcome"] == "uncached-size-limit"
    assert not list(root.glob("*/slot-*/entry"))


def test_cache_cannot_claim_dataset_or_foreign_namespace(tmp_path: Path) -> None:
    """Explicit opt-in is not permission to alter accepted data or unknown files."""
    rows = rows_file(tmp_path / "input")
    with pytest.raises(ValueError, match="outside dataset"):
        invoke(rows.parent / "cache", rows)
    with pytest.raises(ValueError, match="immutable publication"):
        invoke(tmp_path / "dataset_publications_v1" / "cache", rows)
    root = tmp_path / "cache"
    namespace = root / cache.FORMAT
    namespace.mkdir(parents=True)
    foreign = namespace / "history.txt"
    foreign.write_text("preserve")
    with pytest.raises(ValueError, match="nonempty unmarked"):
        invoke(root, rows)
    assert foreign.read_text() == "preserve"


def test_actual_semantic_fence_tracks_code_and_backend_without_runtime_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Actual installed fingerprints change for code/backend evidence, not Torch flags."""
    import torch

    from chipiron.environments.morpion.players.evaluators.neural_networks.training import (
        state_split_cache_identity as identity,
    )

    flags = (
        torch.get_num_threads(),
        torch.get_num_interop_threads(),
        torch.cuda.is_initialized(),
    )
    identity.semantic_fingerprint.cache_clear()
    original = identity.semantic_fingerprint()
    real_digest = identity.digest

    def changed_code(path: Path) -> str:
        return "b" * 64 if path.name == "target_confidence.py" else real_digest(path)

    with monkeypatch.context() as patch:
        patch.setattr(identity, "digest", changed_code)
        identity.semantic_fingerprint.cache_clear()
        assert identity.semantic_fingerprint() != original
    distribution = identity.importlib.metadata.distribution

    def changed_backend(name: str) -> Any:
        value = distribution(name)
        return (
            SimpleNamespace(
                version=value.version + "+different", locate_file=value.locate_file
            )
            if name == "torch"
            else value
        )

    with monkeypatch.context() as patch:
        patch.setattr(identity.importlib.metadata, "distribution", changed_backend)
        identity.semantic_fingerprint.cache_clear()
        assert identity.semantic_fingerprint() != original
    identity.semantic_fingerprint.cache_clear()
    assert (
        torch.get_num_threads(),
        torch.get_num_interop_threads(),
        torch.cuda.is_initialized(),
    ) == flags


def test_memory_hit_still_verifies_source_after_consumption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The process LRU cannot bypass the post-consumption source-stability fence."""
    rows = rows_file(tmp_path / "input")
    root = tmp_path / "cache"
    invoke(root, rows)
    original = cache.digest(rows)
    calls = 0

    def changed_source(path: Path) -> str:
        nonlocal calls
        assert path == rows
        calls += 1
        return original if calls == 1 else "b" * 64

    monkeypatch.setattr(cache, "digest", changed_source)
    with pytest.raises(ValueError, match="source changed during consumption"):
        invoke(root, rows)


def test_retry_repairs_namespace_parent_durability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An OWNER left by failed initialization cannot bypass parent fsync on retry."""
    rows = rows_file(tmp_path / "input")
    root = tmp_path / "cache"
    synchronize = cache._sync
    fail = True
    synced: list[Path] = []

    def interrupted_sync(path: Path) -> None:
        nonlocal fail
        synced.append(path)
        if path == root and fail:
            fail = False
            message = "injected namespace parent fsync failure"
            raise OSError(message)
        synchronize(path)

    monkeypatch.setattr(cache, "_sync", interrupted_sync)
    with pytest.raises(OSError, match="namespace parent"):
        invoke(root, rows)
    assert (root / cache.FORMAT / "OWNER").exists()
    synced.clear()

    def build(
        path: str, digest: str, fraction: float, seed: int, limit: int | None
    ) -> cache.SplitValue:
        assert root in synced and root.parent in synced
        return split._uncached_value(path, digest, fraction, seed, limit)

    cache.load_or_build(root, rows, 0.2, 0, None, split.POLICY, build)

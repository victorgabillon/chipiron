"""Collect matched scientific-index evidence; all inputs are generated fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import zstandard


def read(path: Path) -> Any:
    """Read a small measurement document, never a scientific shard."""
    return json.loads(path.read_text())


def same(left: Path, right: Path) -> str:
    """Require byte equality in bounded chunks and retain its exact fingerprint."""
    digest = hashlib.sha256()
    with left.open("rb") as a, right.open("rb") as b:
        while True:
            first, second = a.read(65536), b.read(65536)
            assert first == second, (left, right)
            if not first:
                break
            digest.update(first)
    return digest.hexdigest()


def collect(root: Path) -> dict[str, Any]:
    """Verify both implementations' complete rows and inspect existing SQLite files."""
    result: dict[str, Any] = {}
    for label in ("baseline", "compact"):
        source = root / f"final-{label}"
        result[label] = {
            "cold": read(source / "cold-g5.json"),
            "reuse": read(source / "reuse-g5.json"),
            "dataset": [
                read(source / f"dataset-{i}.json")["policies"] for i in range(3)
            ],
            "parity": read(source / "parity.json"),
            "reads": read(root / f"final-{label}-reads.json"),
            "scope": read(root / f"final-{label}-scope.json"),
        }
    before, after = result["baseline"], result["compact"]
    for key in ("scientific_sha256", "source_fingerprints"):
        assert before["cold"][key] == after["cold"][key]
    for mode in ("cold", "reuse"):
        assert before[mode]["hashing"] == after[mode]["hashing"]
    for mode in ("reads", "scope"):
        assert before[mode]["hashes"] == after[mode]["hashes"]
    result["exact_jsonl"] = {
        path.name: same(path, root / "final-compact" / path.name)
        for path in sorted((root / "final-baseline").glob("g5-*.jsonl"))
    }
    result["exact_patch_streams"] = {
        path.name: same(path, path.with_name(path.name.replace("baseline", "compact")))
        for path in sorted(root.glob("final-baseline-patch-*.jsonl"))
    }
    indexes = []
    pair = {}
    allocated: dict[tuple[int, int], int] = {}
    for path in sorted(root.rglob("*.sqlite3")):
        with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as db:
            identity = json.loads(
                db.execute("SELECT identity FROM completed").fetchone()[0]
            )
            assert db.execute("PRAGMA integrity_check").fetchone() == ("ok",)
            stat = path.stat()
            allocated[(stat.st_dev, stat.st_ino)] = stat.st_blocks * 512
            indexes.append({
                "path": str(path),
                "allocated_bytes": stat.st_blocks * 512,
                "schema": identity["schema"],
                "generation": identity["generation"],
                "identity": identity,
                "freelist_pages": db.execute("PRAGMA freelist_count").fetchone()[0],
            })
            if identity["generation"] == 5 and path.parent.name == "reader-cache":
                pair[identity["schema"]] = path
    decoder = zstandard.ZstdDecompressor()
    counts = {}
    with (
        sqlite3.connect(f"{pair[2].resolve().as_uri()}?mode=ro", uri=True) as old,
        sqlite3.connect(f"{pair[3].resolve().as_uri()}?mode=ro", uri=True) as new,
    ):
        for table, fields in (
            ("records", "node_id,payload"),
            ("updates", "node_id,order_index,sequence,payload"),
        ):
            count = 0
            query = f"SELECT {fields} FROM {table} ORDER BY node_id"
            for left, right in zip(old.execute(query), new.execute(query), strict=True):
                assert left[:-1] == right[:-1]
                assert left[-1].encode() == decoder.decompress(
                    right[-1], allow_extra_data=False
                )
                count += 1
            counts[table] = count
    result["exact_cache_bodies"] = counts
    result["retained_indexes"] = indexes
    result["retained_sqlite_allocation"] = sum(allocated.values())
    result["repeated_publications"] = {
        str(g): read(root / f"repeat-g{g}.json") for g in range(2, 8)
    }
    return result


def main() -> None:
    """Write curated results while preserving every raw input and output."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("bench", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = {size: collect(args.bench / size) for size in ("1k", "5k")}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({
            size: {
                "paired_dataset_rows": value["compact"]["parity"]["paired_jsonl_rows"],
                "cache_bodies": value["exact_cache_bodies"],
                "retained_indexes": len(value["retained_indexes"]),
                "retained_sqlite_allocation": value["retained_sqlite_allocation"],
            }
            for size, value in result.items()
        })
    )


if __name__ == "__main__":
    main()

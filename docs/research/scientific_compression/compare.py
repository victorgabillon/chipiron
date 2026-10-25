"""Offline exact comparison of newly generated matched raw/compressed boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from chipiron.environments.morpion.bootstrap.scientific_stream import (
    open_scientific_stream,
)
from chipiron.environments.morpion.bootstrap.sharded_training_reader import (
    read_closed_manifest,
)
from docs.research.rolling_snapshot.measure import checkpoint_digest


def compare(root: Path) -> dict[str, Any]:
    """Compare entire decoded bytes and all independent checkpoint components."""
    results: dict[str, Any] = {}
    for size in ("1k", "5k"):
        raw, compressed = (
            root / f"{size}-{encoding}" for encoding in ("raw", "jsonl-zstd-1")
        )
        generations = []
        for generation in (1, 2, 3):
            name = f"{'initial' if generation == 1 else 'repeat'}-g{generation}.json"
            observations = [
                json.loads((base / name).read_text()) for base in (raw, compressed)
            ]
            manifests = [
                base / "work" / observation["record"]["tree_snapshot_path"]
                for base, observation in zip(
                    (raw, compressed), observations, strict=True
                )
            ]
            headers = [read_closed_manifest(p) for p in manifests]
            assert {
                k: v
                for k, v in headers[0].items()
                if k not in ("record_shard", "update_shard")
            } == {
                k: v
                for k, v in headers[1].items()
                if k not in ("record_shard", "update_shard")
            }
            science = {}
            for key in ("record_shard", "update_shard"):
                left, right = (header[key] for header in headers)
                assert (left["generation"], left["count"]) == (
                    right["generation"],
                    right["count"],
                )
                digest = hashlib.sha256()
                length = 0
                with (
                    open_scientific_stream(
                        manifests[0].parent / left["path"], left
                    ) as a,
                    open_scientific_stream(
                        manifests[1].parent / right["path"], right
                    ) as b,
                ):
                    while True:
                        x, y = a.read(65536), b.read(65536)
                        assert x == y, (size, generation, key)
                        if not x:
                            break
                        digest.update(x)
                        length += len(x)
                assert digest.hexdigest() == left["sha256"] == right["decoded_sha256"]
                science[key] = {"bytes": length, "sha256": digest.hexdigest()}
            checkpoints = [
                base / "work" / observation["record"]["checkpoint_path"]
                for base, observation in zip(
                    (raw, compressed), observations, strict=True
                )
            ]
            sections = [checkpoint_digest(path) for path in checkpoints]
            assert sections[0] == sections[1], (
                size,
                generation,
                "checkpoint",
                sections,
            )
            generations.append({
                "generation": generation,
                "science": science,
                "checkpoint_sections": sections[0],
            })
        policies = []
        reference = json.loads((raw / "dataset-g3.json").read_text())
        for item in reference["policies"]:
            label = f"{item['confidence']}-{item['policy']}"
            expected = (raw / f"{label}-oracle.jsonl").read_bytes()
            assert (compressed / f"{label}-oracle.jsonl").read_bytes() == expected
            for base in (raw, compressed):
                assert (base / f"{label}-dataset_stream.jsonl").read_bytes() == expected
            policies.append({
                "confidence": item["confidence"],
                "policy": item["policy"],
                "rows": item["rows"],
                "jsonl_sha256": hashlib.sha256(expected).hexdigest(),
            })
        results[size] = {
            "generations": generations,
            "policies": policies,
            "paired_jsonl_rows": sum(item["rows"] for item in policies),
        }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(compare(args.root), indent=2) + "\n")

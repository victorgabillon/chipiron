"""Record exact E2E equality and explicitly classify independent-run identities."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def differences(a: Any, b: Any, path: str = "") -> list[str]:
    """Report differing fields without rewriting either source artifact."""
    if isinstance(a, dict) and isinstance(b, dict):
        return [
            p
            for key in sorted(a.keys() | b.keys())
            for p in (
                [path + "/" + key]
                if key not in a or key not in b
                else differences(a[key], b[key], path + "/" + key)
            )
        ]
    return [] if a == b else [path]


def compare(root: Path) -> dict[str, Any]:
    """Require exact weights, row payloads and all numerical model metrics."""
    result = {}
    for position in ("opening", "endgame"):
        runs = [root / f"{position}-{encoding}" for encoding in ("raw", "jsonl-zstd-1")]
        summaries = [json.loads((p / "summary.json").read_text()) for p in runs]
        assert all(
            len(json.loads((p / "processes.json").read_text())) == 40 for p in runs
        )
        assert [m["name"] for m in summaries[0]["models"]] == [
            m["name"] for m in summaries[1]["models"]
        ]
        models, datasets = [], []
        for run in runs:
            work = run / "workspace"
            models.append({
                p.relative_to(
                    next(a for a in p.parents if a.name == "models")
                ).as_posix(): p
                for p in (work / "training_publications_v1/objects").rglob("param.pt")
            })
            datasets.append({
                json.loads(p.read_text().splitlines()[0])["metadata"][
                    "bootstrap_generation"
                ]: p
                for p in (work / "dataset_publications_v1/objects").rglob("rows.jsonl")
            })
        assert models[0].keys() == models[1].keys()
        weights = {}
        for key in sorted(models[0]):
            a, b = (m[key] for m in models)
            assert a.read_bytes() == b.read_bytes()
            manifests = [
                json.loads(p.with_name("morpion_manifest.json").read_text())
                for p in (a, b)
            ]
            changed = differences(*manifests)
            # These differences remain recorded and sealed. Scientific metrics,
            # architecture, optimizer configuration and weights must be exact.
            assert all(
                p.startswith(("/metadata/timing", "/metadata/flat_tensor_cache"))
                or p in ("/metadata/dataset_file", "/metadata/output_dir")
                for p in changed
            ), changed
            weights[key] = {
                "sha256": hashlib.sha256(a.read_bytes()).hexdigest(),
                "manifest_differences": changed,
            }
        rows = []
        assert datasets[0].keys() == datasets[1].keys()
        for generation in sorted(datasets[0]):
            a, b = (
                m[generation].read_bytes().splitlines(keepends=True) for m in datasets
            )
            assert a[1:] == b[1:]
            changed = differences(json.loads(a[0]), json.loads(b[0]))
            assert set(changed) <= {
                "/metadata/scientific_source_provenance/producer/identity/evaluator",
                "/metadata/dataset_extraction_identity",
            }, changed
            rows.append({
                "generation": generation,
                "rows": len(a) - 1,
                "row_payload_sha256": hashlib.sha256(b"".join(a[1:])).hexdigest(),
                "full_jsonl_equal": a == b,
                "metadata_differences": changed,
            })
        patches = []
        for a in sorted((runs[0] / "evidence").glob("*/pending.json")):
            b = runs[1] / a.relative_to(runs[0])
            left, right = (json.loads(p.read_text()) for p in (a, b))
            row_changes = [
                differences(a, b)
                for a, b in zip(left["rows"], right["rows"], strict=True)
            ]
            assert all(
                set(changed) <= {"/metadata/model_bundle_path"}
                for changed in row_changes
            ), row_changes
            patches.append({
                "check": a.parent.name,
                "rows": len(left["rows"]),
                "row_differences": row_changes,
                "full_patch_differences": differences(left, right),
            })
        result[position] = {
            "models": weights,
            "datasets": rows,
            "patches": patches,
            "processes_per_encoding": 40,
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(compare(args.root), indent=2) + "\n")

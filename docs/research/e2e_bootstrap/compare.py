"""Classify every repeated-run difference without rewriting either experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def differences(left: Any, right: Any, path: str = "") -> list[dict]:
    """Retain both exact values for every differing JSON leaf."""
    if isinstance(left, dict) and isinstance(right, dict):
        assert left.keys() == right.keys(), (path, "schema/key mismatch")
        return [
            row
            for key in sorted(left.keys() | right.keys())
            for row in differences(left[key], right[key], f"{path}/{key}")
        ]
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return [
            row
            for i, (a, b) in enumerate(zip(left, right, strict=True))
            for row in differences(a, b, f"{path}/{i}")
        ]
    return (
        []
        if type(left) is type(right) and left == right
        else [{"path": path, "left": left, "right": right}]
    )


def compare(left: Path, right: Path) -> dict:
    """Prove numerical/configuration equality and enumerate preserved provenance differences."""
    runs = (left, right)
    summaries = [json.loads((r / "summary.json").read_text()) for r in runs]
    result: dict = {
        "roots": [str(r) for r in runs],
        "checkpoint_payloads": [],
        "generations": [],
        "models": [],
        "datasets": [],
    }
    for item in sorted((left / "evidence").glob("*.json")):
        data = json.loads(item.read_text())
        if data["stage"] == "check":
            label = data["label"]
            assert (left / "evidence" / label / "payload.json").read_bytes() == (
                right / "evidence" / label / "payload.json"
            ).read_bytes()
            result["checkpoint_payloads"].append(label)
    for a, b in zip(
        summaries[0]["generations"], summaries[1]["generations"], strict=True
    ):
        assert a["generation"] == b["generation"]
        assert (
            a["scientific_identity"]["checkpoint_sha256"]
            == b["scientific_identity"]["checkpoint_sha256"]
        )
        sources = [
            r / "workspace" / record["tree_snapshot_path"]
            for r, record in zip(runs, (a, b), strict=True)
        ]
        for name in ("records.jsonl", "updates.jsonl"):
            assert (sources[0].parent / name).read_bytes() == (
                sources[1].parent / name
            ).read_bytes()
        changed = differences(*(json.loads(p.read_text()) for p in sources))
        assert all(d["path"] == "/producer/identity/evaluator" for d in changed)
        result["generations"].append({
            "generation": a["generation"],
            "checkpoint_sha256": a["scientific_identity"]["checkpoint_sha256"],
            "complete_records_and_updates_equal": True,
            "manifest_differences": changed,
        })
    accepted = [s["final"]["accepted_results"] for s in summaries]
    for a, b in zip(*accepted, strict=True):
        generation = a["generation"]
        assert generation == b["generation"]
        assert (
            a["training"]["proposal"]["admission"]["configuration"]
            == b["training"]["proposal"]["admission"]["configuration"]
        )
        rows = [
            (r / "evidence/38-check" / f"rows-{generation}.jsonl")
            .read_bytes()
            .splitlines()
            for r in runs
        ]
        assert rows[0][1:] == rows[1][1:]
        changed = differences(*(json.loads(lines[0]) for lines in rows))
        assert {d["path"] for d in changed} <= {
            "/metadata/scientific_source_provenance/producer/identity/evaluator",
            "/metadata/dataset_extraction_identity",
        }
        result["datasets"].append({
            "generation": generation,
            "rows": len(rows[0]) - 1,
            "entire_row_bodies_equal": True,
            "complete_jsonl_equal": rows[0] == rows[1],
            "header_differences": changed,
        })
        for name in ("linear_a", "linear_b"):
            models = [entry["training"]["proposal"]["models"][name] for entry in (a, b)]
            directories = [
                r / "workspace" / model["path"]
                for r, model in zip(runs, models, strict=True)
            ]
            for filename in ("param.pt", "param.json", "morpion_regressor_args.json"):
                assert (directories[0] / filename).read_bytes() == (
                    directories[1] / filename
                ).read_bytes()
            manifests = [
                json.loads((d / "morpion_manifest.json").read_text())
                for d in directories
            ]
            changed = differences(*manifests)
            for difference in changed:
                path = difference["path"]
                if path.startswith("/metadata/timing") or path in {
                    "/metadata/flat_tensor_cache/load_s",
                    "/metadata/flat_tensor_cache/materialize_s",
                }:
                    difference["classification"] = "measured execution time"
                else:
                    assert path in {
                        "/metadata/dataset_file",
                        "/metadata/output_dir",
                        "/metadata/flat_tensor_cache_path",
                        "/metadata/flat_tensor_cache/path",
                        "/metadata/flat_tensor_cache/manifest_path",
                    }, difference
                    difference["classification"] = "per-attempt execution path"
            result["models"].append({
                "training_generation": generation,
                "candidate": name,
                "full_bundle_sha256": [m["sha256"] for m in models],
                "weights_and_configuration_bytes_equal": True,
                "manifest_differences": changed,
            })
    result["claim"] = (
        "Exact checkpoint, scientific record/update and row-body equality; full model/provenance artifacts intentionally retain all enumerated differences."
    )
    return result


def main() -> None:
    """Write the comparison outside both inspected workspaces."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("output", type=Path)
    options = parser.parse_args()
    options.output.write_text(
        json.dumps(compare(options.left.resolve(), options.right.resolve()), indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()

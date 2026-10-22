"""Summarize exact existing-code experiments, retaining raw observations separately."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from scripts.benchmark_morpion_scientific_reader import sha
from scripts.measure_morpion_generation_publication import disk_usage


def summarize(root: Path, destination: Path) -> dict:
    runs = root / "runs"
    raw = {p.parent.name: json.loads(p.read_text()) for p in runs.glob("*/result.json")}
    result = {
        "comparisons": [],
        "checkpoints": [],
        "readers": [],
        "scientific_update_row_pairs": 0,
    }
    for size in (1000, 5000, 10000):
        checkpoint = raw[f"real-{size}-checkpoint-r2"]
        result["checkpoints"].append({
            "requested_nodes": size,
            "nodes": checkpoint["nodes"],
            "branches": checkpoint["branches"],
            "restore_s": checkpoint["restore"]["elapsed_s"],
            "restore_peak_rss_mib": checkpoint["restore"]["process_peak_rss_mb"],
            "save_mean_s": statistics.mean(x["elapsed_s"] for x in checkpoint["saves"]),
            "save_peak_rss_mib": max(
                x["process_peak_rss_mb"] for x in checkpoint["saves"]
            ),
            "bytes": checkpoint["saves"][-1]["disk"]["unique_file_bytes"],
            "all_components_equal_after_first_restore": all(
                x["checkpoint_digest"] == checkpoint["saves"][0]["checkpoint_digest"]
                for x in checkpoint["saves"]
            ),
            "exact_original_checkpoint_parity": checkpoint["saves"][0][
                "exact_checkpoint_parity"
            ],
        })
        for suffix in ("real", "offset0", "offset20"):
            name = f"real-{size}" if suffix == "real" else f"exact-{size}-{suffix}"
            complete, shared = raw[name + "-complete"], raw[name + "-shared"]
            a, b = complete["observations"], shared["observations"]
            source_a = runs / (name + "-complete") / "complete-1/generation_000001"
            export_b = runs / (name + "-shared") / "shared"
            for index in (1, 2, 3):
                source = (
                    runs / (name + "-complete") / f"complete-{index}/generation_000001"
                )
                assert sha(source / "records.jsonl") == a[0]["record_sha256"][0]
                assert sha(source / "updates.jsonl") == a[0]["update_sha256"]
            assert b[0]["record_sha256"] == a[0]["record_sha256"]
            assert b[0]["update_sha256"] == a[0]["update_sha256"]
            for generation in (1, 2, 3):
                descriptor = json.loads(
                    (
                        export_b / f"generation_{generation:06d}/generation.json"
                    ).read_text()
                )
                first = json.loads((source_a / "generation.json").read_text())
                assert descriptor["generation"] == generation
                for key in ("snapshot", "producer", "creation_index"):
                    assert descriptor[key] == first[key]
                with (
                    (source_a / "updates.jsonl").open("rb") as reference,
                    (export_b / f"generation_{generation:06d}/updates.jsonl").open(
                        "rb"
                    ) as current,
                ):
                    assert json.loads(next(reference))["generation"] == 1
                    assert json.loads(next(current))["generation"] == generation
                    pairs = 0
                    for x, y in zip(reference, current, strict=True):
                        assert (
                            x == y
                        )  # Complete byte rows, including every value contract.
                        pairs += 1
                    assert pairs == shared["nodes"]
                    result["scientific_update_row_pairs"] += pairs
            result["scientific_update_row_pairs"] += 2 * shared["nodes"]
            lengths = []
            with (source_a / "records.jsonl").open() as stream:
                next(stream)
                for line in stream:
                    lengths.append(len(json.loads(line)["state_ref_payload"][1]))
            a_bytes = sum(
                disk_usage(runs / (name + "-complete") / f"complete-{i}")[
                    "unique_file_bytes"
                ]
                for i in (1, 2, 3)
            )
            b_disk = disk_usage(export_b)
            result["comparisons"].append({
                "name": name,
                "nodes": shared["nodes"],
                "replay_moves_min": min(lengths),
                "replay_moves_mean": statistics.mean(lengths),
                "replay_moves_max": max(lengths),
                "complete_first_s": a[0]["elapsed_s"],
                "complete_steady_mean_s": statistics.mean(
                    x["elapsed_s"] for x in a[1:]
                ),
                "shared_steady_mean_s": statistics.mean(x["elapsed_s"] for x in b[1:]),
                "record_copy_mean_s": statistics.mean(
                    x["components"]["copy"]["elapsed_s"] for x in b[1:]
                ),
                "record_link_mean_s": statistics.mean(
                    x["components"]["link"]["elapsed_s"] for x in b[1:]
                ),
                "immutable_bytes": b[-1]["immutable_bytes"],
                "mutable_bytes": b[-1]["mutable_bytes"],
                "descriptor_bytes": b[-1]["descriptor_bytes"],
                "complete_three_export_bytes": a_bytes,
                "shared_three_export_bytes": b_disk["unique_file_bytes"],
                "shared_savings_fraction": 1 - b_disk["unique_file_bytes"] / a_bytes,
                "complete_peak_rss_mib": max(x["process_peak_rss_mb"] for x in a),
                "shared_peak_rss_mib": max(x["process_peak_rss_mb"] for x in b),
                "exact_current_scientific_rows_and_record_bytes": True,
            })
        reader = {"requested_nodes": size}
        for mode in ("cold", "reuse"):
            item = raw[f"real-{size}-indexed-{mode}"]
            assert item["reused"] == (mode == "reuse")
            reader[mode] = item
        reader["reference"] = raw[f"real-{size}-oracle"]["reference"]
        reader["oracle"] = raw[f"real-{size}-oracle"]["cold_result"]
        reader["index_disk"] = disk_usage(runs / f"real-{size}-shared-reader-cache")
        result["readers"].append(reader)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "measurements.json").write_text(json.dumps(raw, indent=2) + "\n")
    (destination / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(summarize(Path(sys.argv[1]), Path(sys.argv[2])), indent=2))

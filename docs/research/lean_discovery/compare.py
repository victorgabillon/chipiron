"""Assert matched scientific inputs/outputs and retain the two history measurements."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


def compare(root: Path, output: Path) -> dict[str, object]:
    """Compare scientific bytes; keep timing/RSS observations separate from parity."""
    before, after = [
        json.loads((root / side / "publication.json").read_bytes())
        for side in ("before", "after_")
    ]
    assert before["producer"] == after["producer"]
    assert before["source_fingerprints"] == after["source_fingerprints"]
    count_keys = (
        "generation",
        "node_count",
        "new_node_count",
        "state_accesses",
        "bytes_written",
    )
    for a, b in zip(before["writer_stats"], after["writer_stats"], strict=True):
        assert {k: a[k] for k in count_keys} == {k: b[k] for k in count_keys}
    stages = []
    for a, b in zip(before["stages"][1:], after["stages"][1:], strict=True):
        keys = ("generation", "scientific_export_sha256", "checkpoint_sha256")
        assert {k: a[k] for k in keys} == {k: b[k] for k in keys}
        stages.append({k: a[k] for k in keys})
    summary = {
        "base": "3623bb0f60d784476294f93fa533f7ec2d0e35db",
        "parity": {
            "matched_generations": 64,
            "matched_export_and_checkpoint_hash_pairs": len(stages),
            "matched_final_source_files": len(before["source_fingerprints"]),
            "producer_equal": True,
            "writer_counts_equal": True,
        },
        "runs": [],
    }
    output.mkdir(exist_ok=True, parents=True)
    for g in (8, 16, 32, 64):
        a, b = [
            json.loads((root / side / f"discovery-{g}.json").read_bytes())
            for side in ("before", "after_")
        ]
        assert a["hashing"] == b["hashing"]
        for kind, factor in (("manifest", 1), ("status", 2)):
            assert (
                b["artifacts"][kind]["publication_entries"]
                == b["artifacts"][kind]["receipt_entries"]
                == 0
            )
            assert (
                a["artifacts"][kind]["publication_entries"] == factor * g * (g + 1) // 2
            )
            assert a["artifacts"][kind]["receipt_entries"] == factor * g * (g - 1) // 2
        summary["runs"].append({"generation": g, "before": a, "after": b})
        for side, label in (("before", "before"), ("after_", "after")):
            shutil.copy2(
                root / side / f"discovery-{g}.json", output / f"{label}-G{g}.json"
            )
    assert (
        summary["runs"][-1]["after"]["artifacts"]["manifest"]["bytes"]
        < 2.1 * summary["runs"][-2]["after"]["artifacts"]["manifest"]["bytes"]
    )
    (output / "scaling.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "scientific-fingerprints.json").write_text(
        json.dumps(
            {
                "matched_source_fingerprints": before["source_fingerprints"],
                "matched_producer": before["producer"],
                "stages": stages,
            },
            indent=2,
        )
        + "\n"
    )
    return summary


if __name__ == "__main__":
    result = compare(Path(sys.argv[1]), Path(sys.argv[2]))
    print(json.dumps(result["parity"], indent=2))
    for run in result["runs"]:
        print(
            run["generation"],
            [
                (
                    v["artifacts"]["manifest"]["bytes"],
                    v["traced_manifest_retained_bytes"],
                    round(v["elapsed_s"], 3),
                )
                for v in (run["before"], run["after"])
            ],
        )

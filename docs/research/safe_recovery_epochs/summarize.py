"""Summarize existing serial measurements; no storage or recovery implementation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def encoded_size(value: Any) -> int:
    """Count canonical compact JSON bytes, excluding a trailing newline."""
    return len(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def summarize(root: Path) -> dict[str, Any]:
    """Read only the new experiment's evidence; never touch its committed artifacts."""
    results = []
    for generation in (8, 16, 32, 64):
        observed = json.loads((root / f"reachability-g{generation}.json").read_bytes())
        report, cost = observed["report"], observed["costs"]
        oracle = json.loads((root / f"oracle-g{generation}.json").read_bytes())
        continuation = json.loads(
            (root / f"continuation-g{generation}.json").read_bytes()
        )
        publication = json.loads((root / f"repeat-g{generation}.json").read_bytes())
        record = oracle["record"]
        identity = record["scientific_identity"]
        assert report["validated"] and observed["exact_workspace_fingerprint_unchanged"]
        assert oracle["unchanged_generations_scientific_parity"]
        assert continuation["selector_rng_continuation"]
        # Serialization model only, NOT an implemented or accepted epoch format.
        epoch = {
            "id": "e" * 32,
            "forked_from": {
                "epoch_id": "a" * 32,
                "generation": generation,
                "capsule_id": record["id"],
                "descriptor_sha256": record["descriptor_sha256"],
                "checkpoint_sha256": identity["checkpoint_sha256"],
                "scientific_sha256": oracle["scientific_sha256"],
                "config_sha256": identity["producer"]["identity"]["config_sha256"],
                "model_sha256": report["generations"][-1]["models"][0]["sha256"],
            },
        }
        results.append({
            "generation": generation,
            "nodes": observed["nodes"],
            "active_root_bytes": report["active_root_bytes"],
            "receipt_count": report["receipt_count"],
            "receipt_bytes": report["receipt_metadata_bytes"],
            "publication_metadata_bytes": report["current_publication_metadata_bytes"],
            "current_epoch_metadata_bytes": 0,
            "modeled_epoch_metadata_bytes_not_implemented": encoded_size(epoch),
            "modeled_epoch": epoch,
            "retained_heavy_generations": len(report["generations"]),
            "scientific_file_blocks": report["scientific_unique_file_blocks"],
            "conditional_reclaimable_file_blocks": report[
                "conditional_reclaimable_file_blocks"
            ],
            "current_reclaimable_bytes": report["current_contract_reclaimable_bytes"],
            "workspace_allocated_bytes": cost["disk"]["allocated_bytes_unique_inodes"],
            "historical_metrics_bytes": observed["historical_metrics_bytes"],
            "historical_stage_metrics_bytes": observed[
                "historical_stage_metrics_bytes"
            ],
            "analysis_seconds": cost["elapsed_s"],
            "analysis_cpu_seconds": cost["costs"]["cpu_user_s"]
            + cost["costs"]["cpu_system_s"],
            "analysis_peak_rss_mib": cost["process_peak_rss_mb"],
            "analysis_hashing": observed["hashing"],
            "publication_seconds": publication["observation"]["elapsed_s"],
            "publication_peak_rss_mib": publication["observation"][
                "process_peak_rss_mb"
            ],
            "publication_hashing": publication["hashing"],
            "scientific_sha256": oracle["scientific_sha256"],
            "selector_rng_continuation": continuation["selector_rng_continuation"],
            "raw_checkpoint_sections_differing": continuation["raw_sections_differing"],
        })
    discovery = json.loads((root / "discover-g64.json").read_bytes())
    return {
        "implementation_base": "089fec3106bdbc690435e61a5b0ce236a5b4821e",
        "parent": "ffa5ea7d3d9624c5efc8a562a184cf3f402d43e3",
        "scope": "new fixed five-node synthetic history; no epoch fork implemented",
        "rows": results,
        "g64_selected_discovery": {
            "seconds": discovery["observation"]["elapsed_s"],
            "hashing": discovery["hashing"],
            "peak_rss_mib": discovery["observation"]["process_peak_rss_mb"],
            "costs": discovery["observation"]["costs"],
        },
        "production_line_delta": {"added": 0, "removed": 0, "changed": 0},
        "deletion_authorized": False,
    }


if __name__ == "__main__":
    print(json.dumps(summarize(Path(sys.argv[1])), indent=2))

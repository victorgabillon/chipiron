"""Dry-run only: demonstrate existing retention constraints on a new tiny fixture.

No collector, retirement protocol or deletion operation is implemented here.
Claim timestamps are deliberately irrelevant to actual lock ownership.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from chipiron.environments.morpion.bootstrap import load_bootstrap_run_state
from chipiron.environments.morpion.bootstrap.generation_publication import (
    DurableGenerationPublication,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    capsule_lease,
    capsule_path,
    committed_scientific_source,
    publication_records,
    validate_capsule,
)
from tests.environments.morpion_durable_v2_helpers import (
    ScientificPublicationRunner,
    run_scientific_publication,
    scientific_producer,
    setup_scientific_baseline,
)
from tests.environments.morpion_patch_receipt_helpers import (
    fixture_patch,
    publish_fixture_patch,
)


def demonstrate(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=False)
    args, paths, state = setup_scientific_baseline(root / "work")
    for generation in range(2, 9):
        patch = replace(
            fixture_patch(f"rolling-{generation}"), tree_generation=state.generation
        )
        patch = replace(
            patch,
            metadata={
                **patch.metadata,
                "scientific_publication": committed_scientific_source(
                    paths.work_dir / state.latest_tree_snapshot_path
                ),
            },
        )
        publish_fixture_patch(paths, patch)
        state = run_scientific_publication(args, ScientificPublicationRunner())
        assert state.generation == generation
    from docs.research.durable_scientific_v2 import measure as existing_measure

    existing_measure.options = SimpleNamespace(profile=False)
    recovered, recovery = existing_measure.measured(
        "restart",
        root,
        lambda: run_scientific_publication(
            args, ScientificPublicationRunner(), max_cycles=0
        ),
    )
    assert recovered == state
    records = publication_records(state.metadata)
    publisher = DurableGenerationPublication(
        paths, scientific_v2_producer=scientific_producer(args)
    )
    existing_roots = publisher.retention_roots(state)
    receipts = state.metadata["checkpoint_bound_reevaluation_v1"]["receipts"]
    raw_before = paths.run_state_path.read_bytes()
    inodes: dict[tuple[int, int], dict] = {}
    for record in records:
        capsule = capsule_path(paths.work_dir, record["id"], 2)
        validate_capsule(paths.work_dir, record)
        for file in capsule.rglob("*"):
            if file.is_file():
                info = file.stat()
                item = inodes.setdefault(
                    (info.st_dev, info.st_ino),
                    {"bytes": info.st_size, "generations": [], "paths": []},
                )
                item["generations"].append(record["generation"])
                item["paths"].append(str(file.relative_to(paths.work_dir)))
    # Actual flock, without a worker process or timeout-based assumption.
    pinned = records[1]
    refused = False
    with capsule_lease(paths.work_dir, pinned["id"], version=2):
        try:
            with capsule_lease(
                paths.work_dir,
                pinned["id"],
                version=2,
                exclusive=True,
                nonblocking=True,
            ):
                raise AssertionError("Exclusive lease acquired over a live reader")
        except BlockingIOError:
            refused = True
    with capsule_lease(
        paths.work_dir, pinned["id"], version=2, exclusive=True, nonblocking=True
    ):
        after_release = True
    # This is a hypothetical root selection, not permission to remove old v2 roots.
    reasons = {
        8: ["current"],
        7: ["rollback"],
        6: ["rollback"],
        2: ["observed-reader-pin"],
        4: ["example-pending-patch-source"],
        1: ["example-explicit-scientific-retention"],
    }
    live = set(reasons)
    exclusively_outside = [
        item for item in inodes.values() if live.isdisjoint(item["generations"])
    ]
    result = {
        "action": "dry-run-only",
        "existing_tiny_fixture_restart": recovery,
        "exact_recovery_state": recovered == state,
        "committed_generations": len(records),
        "current_protocol_roots": sorted(
            str(p.relative_to(paths.work_dir)) for p in existing_roots
        ),
        "committed_deletion_candidates_under_current_contract": [],
        "hypothetical_future_semantic_roots": reasons,
        "receipt_checkpoint_dependencies": [r["checkpoint_path"] for r in receipts],
        "receipt_model_dependencies": sorted({
            (r["model_bundle_path"], r["model_sha256"]) for r in receipts
        }),
        "live_pin_refused_exclusive_lock": refused,
        "exclusive_lock_after_release": after_release,
        "unique_capsule_file_bytes": sum(x["bytes"] for x in inodes.values()),
        "bytes_unique_to_hypothetically_unselected_capsules": sum(
            x["bytes"] for x in exclusively_outside
        ),
        "historical_scientific_files_still_in_head": [
            p
            for x in inodes.values()
            for p in x["paths"]
            if str(Path(records[-1]["descriptor_path"]).parent) in p
            and any(f"generation_{g:06d}" in p for g in range(1, 8))
        ],
        "blockers": [
            "All v2 committed records remain roots and ancestry must start at genesis.",
            "Receipt checkpoint/model paths remain protected by the current retention contract.",
            "Latest capsule still contains every historical update and descriptor.",
            "A momentary lock probe is not authorization for later deletion.",
            "Rollback requires a matching committed receipt/cursor/model recovery boundary, not an older checkpoint alone.",
        ],
    }
    assert raw_before == paths.run_state_path.read_bytes()
    assert load_bootstrap_run_state(paths.run_state_path) == state
    (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    import sys

    logging.disable(logging.CRITICAL)
    target = Path(sys.argv[1]).resolve()
    assert target.is_relative_to(
        Path(
            "/home/pompote/oldata/victor/morpion-scratch/rolling-snapshot-design-20261022"
        )
    )
    print(json.dumps(demonstrate(target), indent=2))

"""Quiescent research inventory using existing authority, validation and inode pins.

No deletion decisions or persistent collector state. Unknown dependencies retain all.
"""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

from chipiron.environments.morpion.bootstrap import MorpionBootstrapPaths
from chipiron.environments.morpion.bootstrap import (
    dataset_publication as dataset,
)
from chipiron.environments.morpion.bootstrap import (
    training_publication as training,
)
from chipiron.environments.morpion.bootstrap.checkpoint_patch_receipts import (
    receipt_ledger,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    capsule_lease,
    capsule_path,
    committed_evaluator,
    file_digest,
    growth_receipt_ownership,
    publication_records,
    validate_capsule,
)
from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
    load_reevaluation_patch,
)
from chipiron.environments.morpion.bootstrap.reevaluation_transaction import (
    validate_proposal,
)
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from docs.research.atomic_activation.measure import disk
from scripts.report_morpion_rolling_recovery import analyze


def audit(root: Path) -> dict:
    """Inventory all generations and fail closed on unsupported ownership."""
    paths = MorpionBootstrapPaths.from_work_dir(root)
    try:
        old = analyze(paths)
    except (ValueError, KeyError, OSError) as error:
        old = {"validated": False, "error": str(error), "type": type(error).__name__}
    with growth_receipt_ownership(paths, create=False), ExitStack() as pins:
        before = file_digest(paths.run_state_path)
        state = load_bootstrap_run_state(paths.run_state_path)
        records = publication_records(state.metadata)
        reasons = {r["generation"]: [] for r in records}
        reasons[state.generation].append("current")
        for record in records[-3:-1]:
            reasons[record["generation"]].append(
                "rollback-candidate-not-executable-in-place"
            )
        # This experiment explicitly retains its genesis for scientific reproducibility.
        reasons[records[0]["generation"]].append("explicit-research-genesis")
        receipts = receipt_ledger(state)
        for receipt in receipts.values():
            reasons[receipt.committed_generation].append("exact-receipt-checkpoint")
            reasons[receipt.tree_generation].append("exact-receipt-source-provenance")
        pending = None
        if paths.pipeline_reevaluation_patch_path.exists():
            proposal = load_reevaluation_patch(paths.pipeline_reevaluation_patch_path)
            pending = validate_proposal(state, proposal)
            reasons[proposal.tree_generation].append("pending-reevaluation-input")
        generations = []
        capsule_files = set()
        for record in records:
            generation = record["generation"]
            try:
                pins.enter_context(
                    capsule_lease(
                        root,
                        record["id"],
                        version=3,
                        exclusive=True,
                        nonblocking=True,
                        create=False,
                    )
                )
                reader = False
            except BlockingIOError:
                pins.enter_context(
                    capsule_lease(
                        root, record["id"], version=3, nonblocking=True, create=False
                    )
                )
                reader = True
                reasons[generation].append("active-reader-observed")
            descriptor = validate_capsule(root, record)
            if "activation_request_sha256" in record:
                reasons[generation].append(
                    "exact-activation-retry-requires-complete-capsule"
                )
            dependencies = {}
            if "accepted_dataset" in record:
                reasons[generation].append("accepted-dataset-source")
                dependencies["dataset"] = dataset.accepted_dataset_identity(
                    paths, state, generation
                )
            if "accepted_training" in record:
                reasons[generation].append(
                    "accepted-training-original-input-and-models"
                )
                reference, selected = training.accepted_selected_model(
                    paths, state, generation
                )
                dependencies["training"] = {
                    "reference": reference,
                    "selected": selected,
                }
            directory = capsule_path(root, record["id"], 3)
            files = [
                root / record["descriptor_path"],
                *(directory / item["path"] for item in descriptor["files"]),
            ]
            capsule_files.update(str(p.relative_to(root)) for p in files)
            generations.append({
                "generation": generation,
                "identity": record,
                "reasons": reasons[generation],
                "reader_observed": reader,
                "models": descriptor["model_pins"],
                "dependencies": dependencies,
                "files": [str(p.relative_to(root)) for p in files],
                "disk": disk(directory),
            })
        inodes = {}
        for path in sorted(root.rglob("*")):
            assert not path.is_symlink(), f"unresolved symlink: {path}"
            if not path.is_file():
                continue
            stat = path.stat()
            item = inodes.setdefault(
                (stat.st_dev, stat.st_ino),
                {
                    "logical_bytes": stat.st_size,
                    "allocated_bytes": stat.st_blocks * 512,
                    "nlink": stat.st_nlink,
                    "paths": [],
                },
            )
            item["paths"].append(str(path.relative_to(root)))
        external = [i for i in inodes.values() if len(i["paths"]) != i["nlink"]]
        staging = list((root / "generation_publications_v3/staging").rglob("*"))
        claims = [
            str(p.relative_to(root))
            for p in paths.pipeline_dir.rglob("*claim*")
            if p.is_file()
        ]
        assert before == file_digest(paths.run_state_path)
        return {
            "action": "read-only",
            "deletion_authorized": False,
            "authority_sha256": before,
            "existing_analyzer": old,
            "generations": generations,
            "active_model": committed_evaluator(root, state)[0],
            "accepted_cursor": state.metadata["atomic_reevaluation"]["accepted_cursor"],
            "exact_receipts": {k: asdict(v) for k, v in receipts.items()},
            "pending": pending,
            "inode_inventory": list(inodes.values()),
            "disk": disk(root),
            "external_hardlinks": external,
            "stage_claim_files": claims,
            "staging_entries": [str(p.relative_to(root)) for p in staging],
            "current_contract_reclaimable_bytes": 0,
            "conditional_unrooted_generations": [
                g for g, r in reasons.items() if not r
            ],
            "conditional_reclaimable_bytes": 0,
            "discovery_projection_bytes": sum(
                p.stat().st_size
                for p in paths.pipeline_dir.rglob("*.json")
                if p.is_file()
                and p.name
                in {
                    "manifest.json",
                    "dataset_status.json",
                    "training_status.json",
                    "active_model.json",
                }
            ),
            "limitations": [
                "All committed capsules remain required by protocol 3.",
                "Every observed generation in this two-cycle run has an explicit dependency in addition to ancestry.",
                "Rollback candidates are complete scientific/checkpoint boundaries, not a validated intentional rollback protocol.",
                "Active claims, unrecognized files and external hardlinks are retained, never treated as expired leases.",
                "This quiescent experiment inventory provides no future deletion fence.",
                "Full experiment/evidence directories remain explicitly retained for this validation.",
            ],
        }


def main() -> None:
    """Write only a report outside the inspected workspace."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    options = parser.parse_args()
    options.output.write_text(
        json.dumps(audit(options.root.resolve()), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

"""Read-only closed-capsule reachability; no retirement or deletion authority.

The proposed roots are a design projection, NOT an accepted run_state format.
All existing committed capsules remain required by publication protocol 3.
"""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chipiron.environments.morpion.bootstrap import (
    MorpionBootstrapPaths,
    dataset_publication,
    load_bootstrap_run_state,
    training_publication,
)
from chipiron.environments.morpion.bootstrap.checkpoint_patch_receipts import (
    RECEIPTS_METADATA_KEY,
    GrowthPatchReceiptSession,
    observe_pending_patch,
    receipt_ledger,
)
from chipiron.environments.morpion.bootstrap.derived.provenance import inside
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    PUBLICATION_KEY,
    active_model_projection,
    artifact_digest,
    capsule_lease,
    capsule_path,
    committed_evaluator,
    file_digest,
    growth_receipt_ownership,
    publication_records,
    require,
    validate_capsule,
)
from chipiron.environments.morpion.bootstrap.linoo_selection_table import (
    load_linoo_selection_table,
)
from chipiron.environments.morpion.bootstrap.model_activation import (
    activation_completion,
)
from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
    load_pipeline_active_model,
    load_pipeline_dataset_status_file,
    load_pipeline_manifest,
    load_pipeline_stage_claim,
    load_pipeline_training_cursor,
    load_pipeline_training_status_file,
    load_reevaluation_cursor,
)
from chipiron.environments.morpion.bootstrap.reevaluation_transaction import (
    complete_cursor,
    transaction_state,
    validate_proposal,
)

if TYPE_CHECKING:
    from chipiron.environments.morpion.bootstrap.run_state import (
        MorpionBootstrapRunState,
    )


def encoded_bytes(value: Any) -> int:
    """Count canonical compact JSON bytes, independently of pretty printing."""
    return len(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _controls(paths: MorpionBootstrapPaths) -> dict[str, str]:
    """Fingerprint observed control files; detect changes, not future quiescence."""
    files = [paths.run_state_path, *paths.pipeline_dir.glob("*.json")]
    files += list(paths.pipeline_dir.glob("generation_*/*"))
    return {
        str(p.relative_to(paths.work_dir)): file_digest(
            inside(paths.work_dir, str(p.relative_to(paths.work_dir)))
        )
        for p in sorted(files)
        if p.is_file() or p.is_symlink()
    }


def _work_roots(
    paths: MorpionBootstrapPaths,
    state: MorpionBootstrapRunState,
    records: dict[int, dict[str, Any]],
    reasons: dict[int, list[str]],
    models: dict[str, set[str]],
    ledger: dict[str, Any],
) -> None:
    """Conservatively retain observed work, including expired claims and cursors."""

    def retain(generation: int | None, reason: str) -> None:
        require(generation in records, f"Unknown {reason} generation: {generation}")
        assert generation is not None
        reasons[generation].append(reason)

    def model(origin: str, digest: str | None = None) -> None:
        require(origin in models, f"Unpinned model: {origin}")
        observed = artifact_digest(inside(paths.work_dir, origin))
        require(observed in models[origin], f"Changed model: {origin}")
        require(
            digest is None or observed == digest, f"Receipt model changed: {origin}"
        )

    explicit_evaluator = state.metadata[PUBLICATION_KEY].get("evaluator_contract") == 1
    transaction = transaction_state(state)
    if explicit_evaluator:
        active_identity, _ = committed_evaluator(paths.work_dir, state)
        assert active_identity is not None
        model(active_identity["path"], active_identity["sha256"])
    if paths.pipeline_active_model_path.exists():
        active = load_pipeline_active_model(paths.pipeline_active_model_path)
        require(
            active == active_model_projection(paths, state)
            if explicit_evaluator
            else active.source == "local_training" and not active.metadata,
            "Unknown model ownership.",
        )
        model(active.model_bundle_path)
    for receipt in ledger.values():
        if explicit_evaluator:
            retain(
                receipt.committed_generation, "exact-receipt-checkpoint-conservative"
            )
            retain(receipt.tree_generation, "exact-receipt-source-conservative")
        model(receipt.model_bundle_path, receipt.model_sha256)
        record = records.get(receipt.committed_generation)
        require(record is not None, "Missing receipt generation.")
        assert record is not None
        require(
            (receipt.checkpoint_path, receipt.checkpoint_sha256)
            == (
                record["checkpoint_path"],
                record["scientific_identity"]["checkpoint_sha256"],
            ),
            "Receipt checkpoint identity differs from its committed capsule.",
        )

    cursor = None
    if transaction is not None and transaction["accepted_cursor"] is not None:
        cursor = complete_cursor(transaction["accepted_cursor"])
        retain(cursor.tree_generation, "authoritative-accepted-cursor")
        model(cursor.model_bundle_path)
    if paths.pipeline_reevaluation_cursor_path.exists():
        projection = load_reevaluation_cursor(paths.pipeline_reevaluation_cursor_path)
        require(transaction is None or projection == cursor, "Stale cursor projection.")
        cursor = projection
        require(not cursor.metadata, "Unknown producer cursor dependencies.")
        retain(cursor.tree_generation, "producer-cursor")
        model(cursor.model_bundle_path)
    pending = observe_pending_patch(paths.pipeline_reevaluation_patch_path)
    if pending is not None:
        patch = pending.patch
        retain(patch.tree_generation, "pending-patch")
        record = records[patch.tree_generation]
        if transaction is not None:
            receipt = ledger.get(patch.patch_id)
            envelope = validate_proposal(
                state,
                patch,
                duplicate_cursor_digest=None
                if receipt is None
                else receipt.accepted_cursor_sha256,
            )
            model(envelope["model"]["path"], envelope["model"]["sha256"])
        else:
            require(
                patch.metadata.get("scientific_publication")
                == {
                    "capsule_id": record["id"],
                    "scientific_identity": record["scientific_identity"],
                },
                "Pending patch scientific source mismatch.",
            )
        model(patch.model_bundle_path)
        require(
            transaction is not None
            or (
                cursor is not None
                and (
                    cursor.last_patch_id,
                    cursor.tree_generation,
                    cursor.evaluator_generation,
                    cursor.evaluator_name,
                    cursor.model_bundle_path,
                )
                == (
                    patch.patch_id,
                    patch.tree_generation,
                    patch.evaluator_generation,
                    patch.evaluator_name,
                    patch.model_bundle_path,
                )
            ),
            "Pending patch lacks matching producer cursor.",
        )
        if patch.patch_id in ledger:
            receipt = ledger[patch.patch_id]
            require(
                (
                    receipt.content_sha256,
                    receipt.tree_generation,
                    receipt.evaluator_generation,
                    receipt.evaluator_name,
                    receipt.model_bundle_path,
                    receipt.rows_requested,
                )
                == (
                    pending.content_sha256,
                    patch.tree_generation,
                    patch.evaluator_generation,
                    patch.evaluator_name,
                    patch.model_bundle_path,
                    len(patch.rows),
                ),
                "Conflicting pending receipt identity.",
            )

    for generation, record in records.items():
        if "accepted_dataset" in record:
            dataset_publication.accepted_dataset_identity(paths, state, generation)
            retain(generation, "accepted-dataset-source-and-results")
        if "accepted_training" in record:
            training_publication.accepted_selected_model(paths, state, generation)
            retain(generation, "accepted-training-source-and-all-candidates")

    if paths.pipeline_training_cursor_path.exists():
        training = load_pipeline_training_cursor(paths.pipeline_training_cursor_path)
        for generation in (
            training.latest_started_generation,
            training.latest_completed_generation,
        ):
            if generation is not None:
                retain(generation, "training-cursor")
    readers = {
        "dataset_claim.json": load_pipeline_stage_claim,
        "training_claim.json": load_pipeline_stage_claim,
        "dataset_status.json": load_pipeline_dataset_status_file,
        "training_status.json": load_pipeline_training_status_file,
        "manifest.json": load_pipeline_manifest,
    }
    for directory in sorted(paths.pipeline_dir.glob("generation_*")):
        require(
            directory.is_dir() and not directory.is_symlink(),
            "Unknown pipeline directory.",
        )
        for path in sorted(directory.iterdir()):
            require(
                path.name in readers and path.is_file(),
                f"Unknown stage dependency: {path}",
            )
            item = readers[path.name](path)
            require(
                directory.name == f"generation_{item.generation:06d}",
                "Stage generation mismatch.",
            )
            require(item.generation in records, "Stage refers to missing generation.")
            if "claim" in path.name:
                require(
                    path.name == f"{item.stage}_claim.json" and not item.metadata,
                    "Unknown claim dependencies.",
                )
                retain(item.generation, f"{item.stage}-claim-even-if-expired")
            elif path.name != "manifest.json":
                # A finished/failed status is not a worker lifetime fence either.
                retain(item.generation, f"{path.stem}-conservative")
            else:
                record = records[item.generation]
                require(
                    (item.runtime_checkpoint_path, item.tree_snapshot_path)
                    == (record["checkpoint_path"], record["tree_snapshot_path"]),
                    "Discovery source differs from committed capsule.",
                )
                if (
                    item.dataset_status != "not_started"
                    or item.training_status != "not_started"
                ):
                    retain(item.generation, "stage-status-conservative")
                # Outputs outside capsules are never candidates. A capsule-local
                # output reference needs a separate ownership contract.
                for value in [item.rows_path, *item.model_bundle_paths.values()]:
                    if value is not None:
                        target = inside(paths.work_dir, value)
                        require(
                            not target.is_relative_to(
                                paths.work_dir / "generation_publications_v3"
                            ),
                            "Unknown capsule-local stage output dependency.",
                        )
    known = {
        "active_model.json",
        "reevaluation_cursor.json",
        "reevaluation_patch.json",
        "training_cursor.json",
        paths.latest_linoo_selection_table_path.name,
    }
    if paths.latest_linoo_selection_table_path.exists():
        require(
            load_linoo_selection_table(paths.latest_linoo_selection_table_path)
            is not None,
            "Malformed diagnostic selection table.",
        )
    require(
        all(p.name in known for p in paths.pipeline_dir.glob("*.json")),
        "Unknown pipeline control.",
    )


def analyze(
    paths: MorpionBootstrapPaths, *, retain: tuple[int, ...] = ()
) -> dict[str, Any]:
    """Report a point-in-time projection; any validation error suppresses candidates."""
    result: dict[str, Any] = {
        "report_version": 1,
        "action": "read-only",
        "deletion_authorized": False,
        "current_contract_reclaimable_bytes": 0,
        "conditional_reclaimable_file_blocks": 0,
    }
    try:
        with growth_receipt_ownership(paths, create=False), ExitStack() as pins:
            result.update(_inspect(paths, pins, retain))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result.update(validated=False, blockers=[f"{type(exc).__name__}: {exc}"])
    return result


def _inspect(
    paths: MorpionBootstrapPaths, pins: ExitStack, retain: tuple[int, ...]
) -> dict[str, Any]:
    """Use existing capsule validation, receipt parsing, Growth lock and inode pins."""
    before = _controls(paths)
    state = load_bootstrap_run_state(paths.run_state_path)
    ordered = publication_records(state.metadata)
    require(
        state.metadata[PUBLICATION_KEY]["schema_version"] == 3,
        "Only closed publication v3 is analyzed.",
    )
    head = ordered[-1]
    require(
        ordered[0]["generation"] == 1,
        "Closed publication ancestry must retain genesis 1.",
    )
    require(
        (
            state.generation,
            state.latest_runtime_checkpoint_path,
            state.latest_tree_snapshot_path,
        )
        == (head["generation"], head["checkpoint_path"], head["tree_snapshot_path"]),
        "Head mismatch; crash recovery cannot select an older checkpoint.",
    )
    ledger = receipt_ledger(state)
    GrowthPatchReceiptSession(paths, object()).validate_state(state)
    receipt_generations = {r.committed_generation for r in ledger.values()}
    records = {r["generation"]: r for r in ordered}
    reasons: dict[int, list[str]] = {g: [] for g in records}
    reasons[state.generation].append("current-crash-recovery")
    for record in ordered[-3:-1]:
        reasons[record["generation"]].append("rollback-candidate-not-executable")
    for generation in retain:
        require(
            type(generation) is int and generation in records,
            "Unknown explicit retention root.",
        )
        reasons[generation].append("explicit-scientific-retention")
    namespace = paths.work_dir / "generation_publications_v3"
    require(
        {p.name for p in (namespace / "objects").iterdir()}
        == {r["id"] for r in ordered},
        "Unknown/uncommitted capsule ownership; recover separately.",
    )
    for name in ("staging", "trash"):
        require(
            not (namespace / name).exists() or not any((namespace / name).iterdir()),
            f"Unresolved {name} ownership; recover separately.",
        )
    inodes: dict[tuple[int, int], dict[str, Any]] = {}
    generations = []
    models: dict[str, set[str]] = {}
    model_holders: dict[tuple[str, str], list[int]] = {}
    blockers = []
    completion_evidence = []
    # Publication still reads the genesis descriptor; it need not keep its tree
    # merely for that metadata. This analysis never retires either artifact.
    retained_commitments = {ordered[0]["descriptor_path"]}
    for record in ordered:
        generation, identifier = record["generation"], record["id"]
        pin = namespace / "pins" / f"{identifier}.lock"
        reader = False
        if not pin.exists():
            reasons[generation].append("missing-stable-pin")
            blockers.append(
                f"G{generation}: missing stable pin; analyzer never creates it"
            )
        else:
            try:
                pins.enter_context(
                    capsule_lease(
                        paths.work_dir,
                        identifier,
                        version=3,
                        exclusive=True,
                        nonblocking=True,
                        create=False,
                    )
                )
            except BlockingIOError:
                pins.enter_context(
                    capsule_lease(
                        paths.work_dir,
                        identifier,
                        version=3,
                        nonblocking=True,
                        create=False,
                    )
                )
                reader = True
                reasons[generation].append("observed-reader-pin")
        descriptor = validate_capsule(paths.work_dir, record)
        require(
            ("model_activation" in descriptor)
            == ("activation_request_sha256" in record),
            "Missing activation completion binding.",
        )
        if "activation_request_sha256" in record:
            completion = activation_completion(paths, state, record)
            retained_commitments.add(record["descriptor_path"])
            completion_evidence.append({
                "generation": generation,
                "request_sha256": completion["request_sha256"],
                "descriptor_path": record["descriptor_path"],
                "descriptor_sha256": record["descriptor_sha256"],
                "historical_payload_required_for_acknowledgement": False,
            })
        root = capsule_path(paths.work_dir, identifier, 3)
        model_pins = descriptor["model_pins"]
        for model in model_pins:
            require(
                artifact_digest(inside(root, model["path"])) == model["sha256"],
                "Frozen model identity mismatch.",
            )
            models.setdefault(model["origin"], set()).add(model["sha256"])
            model_holders.setdefault((model["origin"], model["sha256"]), []).append(
                generation
            )
        files = [
            paths.work_dir / record["descriptor_path"],
            *(root / entry["path"] for entry in descriptor["files"]),
        ]
        for file in files:
            stat = file.stat()
            entry = inodes.setdefault(
                (stat.st_dev, stat.st_ino),
                {
                    "size": stat.st_size,
                    "blocks": stat.st_blocks * 512,
                    "nlink": stat.st_nlink,
                    "paths": [],
                    "generations": [],
                },
            )
            require(
                (entry["size"], entry["blocks"], entry["nlink"])
                == (stat.st_size, stat.st_blocks * 512, stat.st_nlink),
                "Inode changed during analysis.",
            )
            entry["paths"].append(str(file.relative_to(paths.work_dir)))
            entry["generations"].append(generation)
        generations.append({
            "generation": generation,
            "id": identifier,
            "current_contract_required": True,
            "current_contract_reasons": ["committed-publication-ancestry"]
            + (
                ["exact-receipt-checkpoint"]
                if generation in receipt_generations
                else []
            ),
            "observed_reader": reader,
            "rolling_reasons": reasons[generation],
            "rollback_executable": False,
            "files": [str(p.relative_to(paths.work_dir)) for p in files],
            "models": model_pins,
        })
    _work_roots(paths, state, records, reasons, models, ledger)
    # Every unique receipt model must survive in a selected capsule, even if its
    # original logical path is kept outside the candidate namespace.
    live = {g for g, why in reasons.items() if why}
    for holders in model_holders.values():
        if live.isdisjoint(holders):
            chosen = holders[-1]
            reasons[chosen].append("frozen-model-dependency")
            live.add(chosen)
    candidate_inodes = [
        i
        for i in inodes.values()
        if live.isdisjoint(i["generations"])
        and len(i["paths"]) == i["nlink"]
        and retained_commitments.isdisjoint(i["paths"])
    ]
    require(
        all(len(i["paths"]) == i["nlink"] for i in inodes.values()),
        "Unresolved external hard-link ownership; no conditional removal set.",
    )
    require(
        before == _controls(paths),
        "Control artifacts changed during analysis; retry under quiescence.",
    )

    def reference(record: dict[str, Any]) -> dict[str, Any]:
        return {key: record[key] for key in ("generation", "id", "descriptor_sha256")}

    active = {
        "contract": "morpion-rolling-recovery-roots-draft",
        "schema_version": 1,
        "capsule_publication_version": 3,
        "current": reference(head),
        "rollback_candidates": [reference(r) for r in ordered[-3:-1]],
        "additional": [reference(r) for r in ordered[:-3] if r["generation"] in live],
    }
    for item in generations:
        item["rolling_required"] = bool(item["rolling_reasons"])
    return {
        "validated": not blockers,
        "blockers": blockers,
        "source_run_state_sha256": before["run_state.json"],
        "proposed_active_roots": active,
        "active_root_bytes": encoded_bytes(active),
        "receipt_count": len(ledger),
        "receipt_metadata_bytes": encoded_bytes(state.metadata[RECEIPTS_METADATA_KEY]),
        "current_publication_metadata_bytes": encoded_bytes(
            state.metadata[PUBLICATION_KEY]
        ),
        "generations": generations,
        "activation_completion_evidence": completion_evidence,
        "retained_commitment_files": sorted(retained_commitments),
        "conditional_candidates": [] if blockers else sorted(set(records) - live),
        "conditional_reclaimable_files": [] if blockers else candidate_inodes,
        "conditional_reclaimable_file_blocks": 0
        if blockers
        else sum(i["blocks"] for i in candidate_inodes),
        "conditional_reclaimable_file_bytes": 0
        if blockers
        else sum(i["size"] for i in candidate_inodes),
        "capsule_unique_file_blocks": sum(i["blocks"] for i in inodes.values()),
        "scientific_unique_file_blocks": sum(
            i["blocks"]
            for i in inodes.values()
            if any("/scientific_snapshot/" in p for p in i["paths"])
        ),
        "retained_control_files": before,
        "outside_capsules": "all retained; not inventoried for reclamation",
        "limitations": [
            "Protocol 3 and receipt v1 retain every committed generation; current reclaimability is zero.",
            "Conditional candidates mean payload retirement only; listed commitment descriptors remain required.",
            "Accepted results and exact receipt sources/checkpoints are conservatively retained for explicit evaluator workspaces.",
            "Rollback candidates lack sealed producer cursor/receipt recovery context; no in-place rollback.",
            "Control fingerprints detect observed changes, not ABA or future readers; no deletion fence.",
            "Conditional blocks exclude directory overhead, external links, open-FD delays and filesystem sharing.",
        ],
    }


def main() -> None:
    """Print a report to stdout; never write into the inspected workspace."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("--retain", type=int, action="append", default=[])
    args = parser.parse_args()
    result = analyze(
        MorpionBootstrapPaths.from_work_dir(args.work_dir), retain=tuple(args.retain)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result.get("validated"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

"""Seal observed current origins inside the existing Growth capsule transaction."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

from .derived.provenance import atomic_json, inside
from .generation_publication_io import (
    artifact_digest,
    capsule_path,
    file_digest,
    object_digest,
    read_object,
    require,
)
from .prediction_lineage import (
    PatchDisposition,
    PredictionLineage,
    PredictionOrigin,
    read_prediction_lineage,
)
from .prediction_origin_tracking import GrowthPredictionOrigins
from .sharded_training_v2 import _producer
from .sharded_training_v2_contract import LINEAGE_CONTRACT

if TYPE_CHECKING:
    from pathlib import Path

    from .checkpoint_patch_receipts import CheckpointPatchReceipt, PendingPatch
    from .generation_publication import DurableGenerationPublication

MAP = "prediction_lineage.json"
EVIDENCE = "prediction_evidence.json"


@dataclass(frozen=True)
class CommittedPredictionOrigins:
    """Immutable C5 input authenticated by the existing committed capsule."""

    lineage: PredictionLineage
    creation_generations: Mapping[str, int]
    current_origins: Mapping[str, PredictionOrigin]


def _snapshot_inventory(path: Path) -> tuple[dict[str, int], frozenset[str]]:
    """Reuse the bounded scientific parser, without a recursive capsule/index reader."""
    from .sharded_training_reader import iter_shard_rows

    header = read_object(path)
    creation = {}
    estimates = set()
    for role, key in (("records", "record_shard"), ("updates", "update_shard")):
        shard = header[key]
        for parsed in iter_shard_rows(
            path.parent / shard["path"],
            role,
            scientific_generation=header["generation"],
            closed=True,
            descriptor=shard,
        ):
            row = cast("dict[str, Any]", parsed)
            if role == "records":
                creation[row["node_id"]] = row["creation_generation"]
            elif row["value_contract"]["certainties"]["direct"] == "ESTIMATE":
                estimates.add(row["node_id"])
    return creation, frozenset(estimates)


def _origin(event: dict[str, Any]) -> PredictionOrigin:
    """Derive identities from sealed evidence, never from a caller's qualified set."""
    digest = (
        object_digest({
            "contract": LINEAGE_CONTRACT,
            "receipt": event["receipt"],
            "disposition": event["disposition"],
        })
        if event["kind"] == "accepted_patch"
        else object_digest(event)
    )
    return PredictionOrigin(
        event["kind"],
        event["model_sha256"],
        object_digest(event["producer"]),
        event["producer"]["identity"]["config_sha256"],
        event["numeric_recipe_sha256"],
        digest,
    )


def resolve(
    lineage: PredictionLineage, creation: dict[str, int], estimates: frozenset[str]
) -> dict[str, PredictionOrigin]:
    """Resolve C2's compact map after its structural validator has succeeded."""
    births, overrides = dict(lineage.births), dict(lineage.overrides)
    result = {}
    for node in estimates:
        index = overrides.get(node, births.get(creation[node]))
        if index is None:
            require(lineage.default is not None, "Missing default origin.")
            assert lineage.default is not None
            index = lineage.default[1]
        result[node] = lineage.origins[index]
    return result


def _compact(tracker: GrowthPredictionOrigins, checkpoint: str) -> PredictionLineage:
    origins = tuple(
        sorted(
            set(tracker.current.values()),
            key=lambda origin: (
                origin.kind,
                origin.evidence_sha256,
                origin.evaluator_sha256,
            ),
        )
    )
    indices = {origin: index for index, origin in enumerate(origins)}
    default = None
    births = {}
    for index, origin in enumerate(origins):
        if origin.kind == "genesis":
            require(default is None, "Conflicting genesis evidence.")
            default = (1, index)
        elif origin.kind == "full_refresh" and len(origins) == 1:
            default = (max(tracker.creation.values()), index)
        elif origin.kind == "growth_birth":
            generation = tracker.evidence[origin.evidence_sha256]["generation"]
            require(generation not in births, "Conflicting birth events.")
            births[generation] = index
    overrides = []
    for node, origin in sorted(tracker.current.items()):
        generation = tracker.creation[node]
        inherited = births.get(generation)
        if inherited is None and default is not None and generation <= default[0]:
            inherited = default[1]
        if inherited != indices[origin]:
            overrides.append((node, indices[origin]))
    return PredictionLineage(
        checkpoint, origins, default, tuple(sorted(births.items())), tuple(overrides)
    )


def seal_prediction_origins(
    tracker: GrowthPredictionOrigins,
    publication: DurableGenerationPublication,
    checkpoint: Path,
    *,
    pending: PendingPatch | None = None,
    receipt: CheckpointPatchReceipt | None = None,
) -> dict[str, Any]:
    """Prepare bytes before capsule sealing; this function never commits run state."""
    require(tracker.valid, "Invalidated prediction transaction.")
    digest = artifact_digest(checkpoint)
    require(
        tracker.saved == (digest, tracker.revision),
        "Prediction observations differ from the saved checkpoint.",
    )
    require(
        publication.version == 3 and publication.producer is not None,
        "Prediction origins require the existing closed publication.",
    )
    assert publication.producer is not None and publication.staging is not None
    require(
        tracker.generation
        == (1 if publication.state is None else publication.state.generation + 1),
        "Prediction origin generation mismatch.",
    )
    producer = _producer(publication.producer)
    grouped: dict[tuple[str, str], list[str]] = {}
    for node, write in tracker.writes.items():
        grouped.setdefault(write, []).append(node)
    for (kind, model_sha), nodes in grouped.items():
        if kind == "accepted_patch":
            require(receipt is not None, "Applied predictions lack a model receipt.")
            assert receipt is not None
            model_sha = receipt.model_sha256
        require(
            any(pin["sha256"] == model_sha for pin in publication.model_pins),
            "Observed model has no frozen publication pin.",
        )
        event: dict[str, Any] = {
            "kind": kind,
            "model_sha256": model_sha,
            "producer": {
                **producer,
                "identity": {**producer["identity"], "evaluator": model_sha},
            },
            "numeric_recipe_sha256": tracker.native_recipe(),
            "generation": tracker.generation,
            "parent_checkpoint": tracker.parent_checkpoint,
            "written": sorted(nodes),
        }
        if kind == "accepted_patch":
            require(
                pending is not None
                and receipt is not None
                and tracker.disposition is not None,
                "Applied predictions have no accepted receipt evidence.",
            )
            assert (
                pending is not None
                and receipt is not None
                and tracker.disposition is not None
            )
            require(
                tracker.patch == pending.patch,
                "Observed patch differs from pending bytes.",
            )
            event.update(
                receipt=asdict(receipt), disposition=asdict(tracker.disposition)
            )
        origin = _origin(event)
        if kind == "accepted_patch":
            assert (
                tracker.disposition is not None
                and pending is not None
                and receipt is not None
            )
            tracker.disposition.validate_receipt(pending, receipt, origin)
        tracker.evidence[origin.evidence_sha256] = event
        for node in nodes:
            tracker.current[node] = origin
    for node in tracker.expected_values:
        tracker.creation.setdefault(node, tracker.generation)
    # Exact/null writes never inherit an old neural origin.
    from valanga.evaluations import Certainty

    estimates = frozenset(
        node
        for node, value in tracker.expected_values.items()
        if value is not None and getattr(value, "certainty", None) == Certainty.ESTIMATE
    )
    require(
        frozenset(tracker.current) == estimates,
        "Sealed direct origin coverage differs.",
    )
    assert publication.sharded_path is not None
    actual_creation, actual_estimates = _snapshot_inventory(publication.sharded_path)
    require(
        actual_creation == tracker.creation and actual_estimates == estimates,
        "Observed origins differ from the sealed scientific inventory.",
    )
    for node, origin in tracker.current.items():
        require(
            node in tracker.evidence[origin.evidence_sha256]["written"],
            "Current origin differs from actual observed write IDs.",
        )
    lineage = _compact(tracker, digest)
    lineage.validate(
        checkpoint_sha256=digest,
        creation_generations=tracker.creation,
        estimate_node_ids=estimates,
        qualified_origins=frozenset(tracker.current.values()),
    )
    events = {
        origin.evidence_sha256: tracker.evidence[origin.evidence_sha256]
        for origin in lineage.origins
    }
    stage = publication.staging
    (stage / MAP).write_bytes(lineage.to_bytes())
    patches = {}
    if receipt is not None:
        require(
            pending is not None and tracker.disposition is not None,
            "Receipt has no actual patch disposition.",
        )
        assert pending is not None and tracker.disposition is not None
        origin = PredictionOrigin(
            "accepted_patch",
            receipt.model_sha256,
            object_digest(producer),
            producer["identity"]["config_sha256"],
            tracker.native_recipe(),
            tracker.disposition.evidence_sha256(receipt),
        )
        tracker.disposition.validate_receipt(pending, receipt, origin)
        sha = receipt.content_sha256
        relative = f"prediction_patches/{sha}.json"
        (stage / relative).parent.mkdir(exist_ok=True)
        (stage / relative).write_bytes(tracker.patch_bytes[sha])
        require(
            file_digest(stage / relative) == sha,
            "Pending patch changed before sealing.",
        )
        patches[sha] = relative
    for event in events.values():
        if event["kind"] == "accepted_patch":
            sha = event["receipt"]["content_sha256"]
            data = tracker.patch_bytes[sha]
            relative = f"prediction_patches/{sha}.json"
            path = stage / relative
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(data)
            require(file_digest(path) == sha, "Preserved patch bytes changed.")
            patches[sha] = relative
    atomic_json(
        stage / EVIDENCE,
        {
            "contract": LINEAGE_CONTRACT,
            "creation": tracker.creation,
            "estimates": sorted(estimates),
            "events": events,
            "patches": patches,
            # Even a proof-only/zero-applied patch keeps its actual disposition.
            "accepted_patch": None
            if receipt is None
            else {
                "receipt": asdict(receipt),
                "disposition": asdict(tracker.disposition)
                if tracker.disposition is not None
                else None,
            },
        },
    )
    return {
        "contract": LINEAGE_CONTRACT,
        "path": MAP,
        "sha256": file_digest(stage / MAP),
        "evidence_path": EVIDENCE,
        "evidence_sha256": file_digest(stage / EVIDENCE),
    }


def read_committed_prediction_origins(
    work_dir: Path,
    record: dict[str, Any],
) -> CommittedPredictionOrigins | None:
    """Authenticate existing capsule bytes and qualify each node from actual write evidence.

    The returned state is private input to the next Growth transaction, not a new
    scientific admission. C5 still owns the mixed-source Dataset boundary.
    """
    from .bootstrap_paths import MorpionBootstrapPaths
    from .generation_publication_io import publication_records, validate_capsule
    from .run_state import load_bootstrap_run_state

    state = load_bootstrap_run_state(
        MorpionBootstrapPaths.from_work_dir(work_dir).run_state_path
    )
    matches = [
        candidate
        for candidate in publication_records(state.metadata)
        if candidate["id"] == record["id"]
    ]
    require(
        len(matches) == 1
        and all(
            matches[0].get(key) == record.get(key)
            for key in (
                "id",
                "descriptor_path",
                "descriptor_sha256",
                "checkpoint_path",
                "tree_snapshot_path",
                "parent_id",
                "generation",
                "publication_version",
                "scientific_identity",
                "prediction_lineage",
            )
        ),
        "Prediction source is not a committed publication.",
    )
    record = matches[0]
    descriptor = validate_capsule(work_dir, record)
    tracker = read_prediction_evidence(work_dir, record, descriptor)
    if tracker is None:
        return None
    root = capsule_path(work_dir, record["id"], record["publication_version"])
    ref = descriptor["prediction_lineage"]
    lineage = read_prediction_lineage(
        (root / MAP).read_bytes(),
        sidecar_sha256=ref["sha256"],
        capsule_file_sha256=ref["sha256"],
    )
    return CommittedPredictionOrigins(
        lineage, MappingProxyType(tracker.creation), MappingProxyType(tracker.current)
    )


def read_prediction_evidence(
    work_dir: Path,
    record: dict[str, Any],
    descriptor: dict[str, Any],
) -> GrowthPredictionOrigins | None:
    """Validate the focused extension after the existing inventory/seal checks."""
    ref = descriptor.get("prediction_lineage")
    require(
        record.get("prediction_lineage") == ref,
        "Prediction reference differs from committed root.",
    )
    if ref is None:
        return None
    require(
        set(ref) == {"contract", "path", "sha256", "evidence_path", "evidence_sha256"}
        and ref["contract"] == LINEAGE_CONTRACT
        and ref["path"] == MAP
        and ref["evidence_path"] == EVIDENCE,
        "Malformed prediction reference.",
    )
    root = capsule_path(work_dir, record["id"], record["publication_version"])
    inventory = {entry["path"]: entry["sha256"] for entry in descriptor["files"]}
    require(
        inventory.get(EVIDENCE)
        == ref["evidence_sha256"]
        == file_digest(root / EVIDENCE),
        "Prediction evidence binding differs.",
    )
    lineage = read_prediction_lineage(
        (root / MAP).read_bytes(),
        sidecar_sha256=ref["sha256"],
        capsule_file_sha256=inventory[MAP],
    )
    evidence = read_object(root / EVIDENCE)
    require(
        set(evidence)
        == {"contract", "creation", "estimates", "events", "patches", "accepted_patch"}
        and evidence["contract"] == LINEAGE_CONTRACT,
        "Malformed prediction evidence.",
    )
    tracker = GrowthPredictionOrigins(
        generation=record["generation"] + 1,
        parent_checkpoint=lineage.checkpoint_sha256,
        creation=evidence["creation"],
        evidence=evidence["events"],
    )
    for sha, path in evidence["patches"].items():
        require(
            path == f"prediction_patches/{sha}.json" and inventory.get(path) == sha,
            "Unsealed prediction patch.",
        )
        data = inside(root, path).read_bytes()
        require(
            file_digest(inside(root, path)) == sha, "Prediction patch bytes changed."
        )
        tracker.patch_bytes[sha] = data
    from .bootstrap_paths import MorpionBootstrapPaths
    from .generation_publication_io import (
        publication_records,
        read_committed_descriptor,
    )
    from .run_state import load_bootstrap_run_state

    paths = MorpionBootstrapPaths.from_work_dir(work_dir)
    committed = (
        load_bootstrap_run_state(paths.run_state_path)
        if paths.run_state_path.exists()
        else None
    )
    records = [] if committed is None else publication_records(committed.metadata)
    sources = [(record, descriptor)]
    sources.extend(
        (previous, read_committed_descriptor(work_dir, previous))
        for previous in records
        if previous["generation"] < record["generation"]
    )
    model_pins: dict[str, Path] = {}
    for source_record, source_descriptor in sources:
        source_root = capsule_path(
            work_dir, source_record["id"], source_record["publication_version"]
        )
        for pin in source_descriptor["model_pins"]:
            model_pins.setdefault(pin["sha256"], inside(source_root, pin["path"]))
    qualified = set()
    for sha, event in tracker.evidence.items():
        origin = _origin(event)
        require(origin.evidence_sha256 == sha, "Fabricated prediction event digest.")
        require(
            event["producer"]["identity"]["evaluator"] == origin.evaluator_sha256,
            "Prediction event model differs from producer.",
        )
        require(
            origin.evaluator_sha256 in model_pins
            and artifact_digest(model_pins[origin.evaluator_sha256])
            == origin.evaluator_sha256,
            "Prediction origin has no authentic retained model pin.",
        )
        require(
            isinstance(event["generation"], int)
            and not isinstance(event["generation"], bool)
            and 1 <= event["generation"] <= record["generation"],
            "Invalid write-event generation.",
        )
        parent_sources = [
            source_record
            for source_record, _ in sources
            if source_record["generation"] == event["generation"] - 1
        ]
        parent_digest = (
            None
            if event["generation"] == 1
            else (
                parent_sources[0]["scientific_identity"]["checkpoint_sha256"]
                if len(parent_sources) == 1
                else "missing"
            )
        )
        require(
            event["parent_checkpoint"] == parent_digest,
            "Write event has a different parent checkpoint.",
        )
        event_sources = [
            source
            for source_record, source in sources
            if source_record["generation"] == event["generation"]
        ]
        require(
            len(event_sources) == 1
            and event["producer"]
            == event_sources[0]["scientific_identity"]["producer"],
            "Prediction event producer differs from its committed write boundary.",
        )
        written = event["written"]
        require(
            isinstance(written, list)
            and len(written) == len(set(written))
            and all(
                isinstance(node, str) and node in tracker.creation for node in written
            ),
            "Invalid observed write IDs.",
        )
        if origin.kind == "accepted_patch":
            from .checkpoint_patch_receipts import CheckpointPatchReceipt, PendingPatch
            from .pipeline_artifacts import load_reevaluation_patch

            receipt = CheckpointPatchReceipt(**event["receipt"])
            disposition = PatchDisposition(**{
                key: tuple(value) for key, value in event["disposition"].items()
            })
            path = inside(root, evidence["patches"][receipt.content_sha256])
            pending = PendingPatch(
                load_reevaluation_patch(path), receipt.content_sha256
            )
            disposition.validate_receipt(pending, receipt, origin)
            require(
                set(written) == set(disposition.applied),
                "Proposed IDs differ from actual applied IDs.",
            )
        else:
            require(
                origin.kind in {"genesis", "growth_birth", "full_refresh"},
                "Unknown native write event.",
            )
            if origin.kind != "full_refresh":
                require(
                    all(
                        tracker.creation[node] == event["generation"]
                        for node in written
                    ),
                    "Birth evidence describes an existing node.",
                )
        qualified.add(origin)
    accepted = evidence["accepted_patch"]
    if accepted is not None:
        from .checkpoint_patch_receipts import (
            CheckpointPatchReceipt,
            PendingPatch,
            receipt_ledger,
        )
        from .pipeline_artifacts import load_reevaluation_patch

        receipt = CheckpointPatchReceipt(**accepted["receipt"])
        disposition = PatchDisposition(**{
            key: tuple(value) for key, value in accepted["disposition"].items()
        })
        pending = PendingPatch(
            load_reevaluation_patch(
                inside(root, evidence["patches"][receipt.content_sha256])
            ),
            receipt.content_sha256,
        )
        producer = descriptor["scientific_identity"]["producer"]
        origin = PredictionOrigin(
            "accepted_patch",
            receipt.model_sha256,
            object_digest(producer),
            producer["identity"]["config_sha256"],
            tracker.native_recipe(),
            disposition.evidence_sha256(receipt),
        )
        disposition.validate_receipt(pending, receipt, origin)
        require(
            receipt.checkpoint_path == record["checkpoint_path"]
            and receipt.checkpoint_sha256 == lineage.checkpoint_sha256
            and receipt.committed_generation == record["generation"],
            "Prediction disposition is not bound to this checkpoint commit.",
        )
        if committed is not None and any(r["id"] == record["id"] for r in records):
            require(
                receipt_ledger(committed).get(receipt.patch_id) == receipt,
                "Sealed disposition differs from the accepted receipt ledger.",
            )
    # The sealed current map is not its own origin qualification: every assignment
    # must occur in its separately bound actual-write disposition/event.
    require(
        isinstance(evidence["estimates"], list)
        and all(isinstance(node, str) for node in evidence["estimates"])
        and len(evidence["estimates"]) == len(set(evidence["estimates"])),
        "Malformed current estimate inventory.",
    )
    estimates = frozenset(evidence["estimates"])
    from .generation_publication_io import RETIRED_PAYLOAD_KEY

    if RETIRED_PAYLOAD_KEY not in record:
        creation, actual_estimates = _snapshot_inventory(
            inside(work_dir, record["tree_snapshot_path"])
        )
        require(
            creation == tracker.creation and actual_estimates == estimates,
            "Prediction evidence differs from the actual scientific inventory.",
        )
    lineage.validate(
        checkpoint_sha256=record["scientific_identity"]["checkpoint_sha256"],
        creation_generations=tracker.creation,
        estimate_node_ids=estimates,
        qualified_origins=frozenset(qualified),
    )
    tracker.current = resolve(lineage, tracker.creation, estimates)
    for node, origin in tracker.current.items():
        require(
            node in tracker.evidence[origin.evidence_sha256]["written"],
            "Node origin was fabricated from a proposed or unrelated write.",
        )
    return tracker


def restore_prediction_origins(checkpoint: Path) -> GrowthPredictionOrigins | None:
    """Restore only from an existing committed capsule, never a loose sidecar."""
    from .bootstrap_paths import MorpionBootstrapPaths
    from .generation_publication_io import locate_capsule, publication_records
    from .run_state import load_bootstrap_run_state

    location = locate_capsule(checkpoint)
    if location is None or location[2] != 3:
        return None
    work_dir, identifier, _ = location
    paths = MorpionBootstrapPaths.from_work_dir(work_dir)
    state = load_bootstrap_run_state(paths.run_state_path)
    records = publication_records(state.metadata)
    require(
        identifier == records[-1]["id"],
        "Prediction restore requires the committed head.",
    )
    require(
        inside(work_dir, records[-1]["checkpoint_path"]) == checkpoint,
        "Prediction restore checkpoint differs.",
    )
    from .generation_publication_io import validate_capsule

    descriptor = validate_capsule(work_dir, records[-1])
    return read_prediction_evidence(work_dir, records[-1], descriptor)

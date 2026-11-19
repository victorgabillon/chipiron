"""C2 structural evidence contracts and refusal fences; no mixed runtime admission."""

import hashlib
import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest

from chipiron.environments.morpion.bootstrap.prediction_lineage import (
    PatchDisposition,
    PredictionLineage,
    PredictionLineageError,
    PredictionOrigin,
    read_prediction_lineage,
    validate_homogeneous_direct_claim,
)
from chipiron.environments.morpion.bootstrap.sharded_training_v2_contract import (
    CONTRACT,
    KIND,
    MIXED_SOURCE_CONTRACT,
    closed_envelope,
)
from chipiron.environments.morpion.learning.target_confidence import (
    CONFIDENCE_CONTRACT,
    TARGET_CONTRACT,
    DatasetConfidenceContext,
    dataset_format_version,
)
from chipiron.environments.morpion.learning.target_policy import (
    training_target_metadata,
    validate_model_target_provenance,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.fast


def digest(label: str) -> str:
    """Synthetic identities, never claims about an installed native producer."""
    return hashlib.sha256(label.encode()).hexdigest()


A = PredictionOrigin(
    "genesis",
    digest("A"),
    digest("producer"),
    digest("config"),
    digest("numeric"),
    digest("verified-genesis-event"),
)
B = replace(
    A,
    kind="growth_birth",
    evaluator_sha256=digest("B"),
    evidence_sha256=digest("birth-event"),
)
PATCH = replace(
    B, kind="accepted_patch", evidence_sha256=digest("patch-disposition-event")
)
CHECKPOINT = digest("checkpoint")
INVENTORY = {"old": 1, "updated": 1, "new": 2, "certified": 1, "null": 1}
ESTIMATES = frozenset({"old", "updated", "new"})
QUALIFIED = frozenset({A, B, PATCH})
LINEAGE = PredictionLineage(
    CHECKPOINT, (A, B, PATCH), (1, 0), ((2, 1),), (("updated", 2),)
)


def validate(lineage: PredictionLineage) -> frozenset[PredictionOrigin]:
    """Supply synthetic already-authenticated facts; C3 will own their production."""
    return lineage.validate(
        checkpoint_sha256=CHECKPOINT,
        creation_generations=INVENTORY,
        estimate_node_ids=ESTIMATES,
        qualified_origins=QUALIFIED,
    )


def read(data: bytes) -> PredictionLineage:
    """Simulate matching exact-byte hashes from the two authenticated bindings."""
    sha = hashlib.sha256(data).hexdigest()
    return read_prediction_lineage(data, sidecar_sha256=sha, capsule_file_sha256=sha)


def provenance() -> dict[str, Any]:
    """Keep the frozen homogeneous source envelope, including its old producer."""
    return {
        "source_format_kind": KIND,
        "source_format_version": 2,
        "scientific_contract": CONTRACT,
        "source_generation": 1,
        "producer": {"identity": {"evaluator": digest("historical-model")}},
    }


def metadata(source: dict[str, Any]) -> dict[str, Any]:
    """Build complete existing explicit target provenance without runtime work."""
    return {
        "dataset_version": 2,
        "dataset_target_source": "effective",
        "dataset_target_contract": TARGET_CONTRACT,
        "dataset_family_target_policy": "none",
        "dataset_family_prediction_blend": 0.5,
        "dataset_confidence_policy": "selected_value_v2",
        "dataset_confidence_contract": CONFIDENCE_CONTRACT,
        "dataset_extraction_identity": digest("extraction"),
        "scientific_source_provenance": source,
    }


def test_mixed_origins_pass_structural_contract_and_compact_roundtrip() -> None:
    """The current map stands alone; certificates/null values are not predictions."""
    data = LINEAGE.to_bytes()
    restored = read(data)
    assert restored == LINEAGE
    assert restored.to_bytes() == data
    assert validate(restored) == QUALIFIED
    assert data.count(A.evaluator_sha256.encode()) == 1
    assert b"score" not in data
    assert dict(restored.overrides) == {"updated": 2}


def test_active_model_cannot_relabel_untouched_estimates() -> None:
    """Partial B writes leave A present; asking for B never fills missing origins."""
    actual = validate(LINEAGE)
    with pytest.raises(PredictionLineageError, match="Homogeneous claim"):
        validate_homogeneous_direct_claim(actual, B)
    with pytest.raises(PredictionLineageError, match="Missing direct ESTIMATE"):
        validate(replace(LINEAGE, default=None))
    with pytest.raises(PredictionLineageError, match="No direct origins"):
        validate_homogeneous_direct_claim(frozenset(), B)
    # A homogeneous claim is testable only after every estimate resolves to B.
    refreshed = replace(LINEAGE, overrides=(("old", 2), ("updated", 2)))
    validate_homogeneous_direct_claim(validate(refreshed), B)
    # The same model under a different numeric recipe is not homogeneous either.
    with pytest.raises(PredictionLineageError, match="Homogeneous claim"):
        validate_homogeneous_direct_claim(
            validate(refreshed),
            replace(B, numeric_recipe_sha256=digest("other-recipe")),
        )


@pytest.mark.parametrize(
    ("lineage", "message"),
    [
        (replace(LINEAGE, checkpoint_sha256=digest("other")), "checkpoint binding"),
        (replace(LINEAGE, origins=(A, B, PATCH, A)), "Duplicate origin"),
        (
            replace(
                LINEAGE,
                origins=(A, B, replace(PATCH, producer_sha256=digest("unqualified"))),
            ),
            "Unqualified",
        ),
        (replace(LINEAGE, default=(1, 1)), "proven genesis"),
        (replace(LINEAGE, default=(9, 0)), "exceeds checkpoint"),
        (replace(LINEAGE, default=(True, 0)), "integer"),
        (replace(LINEAGE, births=()), "Missing direct ESTIMATE"),
        (replace(LINEAGE, births=((2, 1), (2, 1))), "Duplicate birth"),
        (replace(LINEAGE, births=((1, 1),)), "overlaps default"),
        (replace(LINEAGE, births=((3, 1),)), "absent from inventory"),
        (replace(LINEAGE, births=((2, 0),)), "Invalid birth origin"),
        (replace(LINEAGE, overrides=(("updated", 99),)), "Unknown prediction origin"),
        (replace(LINEAGE, overrides=(("updated", -1),)), "integer"),
        (replace(LINEAGE, overrides=(("updated", True),)), "integer"),
        (
            replace(LINEAGE, overrides=(("updated", 2), ("updated", 1))),
            "Duplicate override",
        ),
        (replace(LINEAGE, overrides=(("certified", 2),)), "not a direct ESTIMATE"),
        (replace(LINEAGE, overrides=(("null", 2),)), "not a direct ESTIMATE"),
        (replace(LINEAGE, overrides=(("missing", 2),)), "not a direct ESTIMATE"),
    ],
)
def test_refuse_ambiguous_or_unqualified_map(
    lineage: PredictionLineage, message: str
) -> None:
    """Resealing a structurally contradictory map cannot make it qualified."""
    with pytest.raises(PredictionLineageError, match=message):
        validate(lineage)


def test_estimate_inventory_and_write_evidence_are_required() -> None:
    """A model pin alone cannot supply the write evidence or missing inventory."""
    for inventory, qualified, message in (
        ({"updated": 1, "new": 2}, QUALIFIED, "absent from checkpoint"),
        (INVENTORY, frozenset(), "owned write evidence"),
    ):
        with pytest.raises(PredictionLineageError, match=message):
            LINEAGE.validate(
                checkpoint_sha256=CHECKPOINT,
                creation_generations=inventory,
                estimate_node_ids=ESTIMATES,
                qualified_origins=qualified,
            )


def test_certified_or_absent_direct_values_do_not_need_neural_origins() -> None:
    """A proof-only inventory is distinct from a neural full-refresh claim."""
    lineage = PredictionLineage(CHECKPOINT, (), None, (), ())
    resolved = lineage.validate(
        checkpoint_sha256=CHECKPOINT,
        creation_generations={"certified": 1, "null": 1},
        estimate_node_ids=frozenset(),
        qualified_origins=frozenset(),
    )
    assert resolved == frozenset()
    with pytest.raises(PredictionLineageError, match="No direct origins"):
        validate_homogeneous_direct_claim(resolved, A)


@pytest.mark.parametrize("binding", ["source", "capsule", "bytes"])
def test_exact_sidecar_binding(binding: str) -> None:
    """Both authenticated references must match the exact bytes, including whitespace."""
    data = LINEAGE.to_bytes()
    sha = hashlib.sha256(data).hexdigest()
    with pytest.raises(PredictionLineageError, match="sidecar binding"):
        read_prediction_lineage(
            data + b"\n" if binding == "bytes" else data,
            sidecar_sha256=digest("wrong") if binding == "source" else sha,
            capsule_file_sha256=digest("wrong") if binding == "capsule" else sha,
        )


@pytest.mark.parametrize(
    "damage", ["missing", "extra", "version", "hash", "integer", "pairs", "kind"]
)
def test_invalid_sidecar_schema_even_with_valid_hash(damage: str) -> None:
    """Hash integrity is separate from schema and origin qualification."""
    payload = json.loads(LINEAGE.to_bytes())
    if damage == "missing":
        payload.pop("origins")
    elif damage == "extra":
        payload["requested_evaluator"] = digest("B")
    elif damage == "version":
        payload["contract"] = "future-contract"
    elif damage == "hash":
        payload["origins"][0]["evidence_sha256"] = ""
    elif damage == "integer":
        payload["births"][0][1] = True
    elif damage == "pairs":
        payload["overrides"] = {"updated": 2}
    else:
        payload["origins"][0]["kind"] = "training_accepted"
    with pytest.raises(PredictionLineageError):
        read(json.dumps(payload).encode())


def test_duplicate_json_keys_refused() -> None:
    """A decoder must not silently choose a second conflicting source identity."""
    data = LINEAGE.to_bytes()
    with pytest.raises(PredictionLineageError, match="Duplicate lineage JSON key"):
        read(b'{"checkpoint_sha256":"' + CHECKPOINT.encode() + b'",' + data[1:])


def test_patch_disposition_preserves_equal_score_writes_and_certificate_skips() -> None:
    """No scalar comparison is involved in assigning actual write dispositions."""
    disposition = PatchDisposition(
        ("equal-score",), ("protected",), ("absent",), ("proof",)
    )
    disposition.validate(("equal-score", "protected", "absent", "proof"))
    assert disposition.applied == ("equal-score",)
    for bad in (
        replace(disposition, applied=()),
        replace(disposition, applied=("equal-score", "equal-score")),
        replace(disposition, protected=("protected", "equal-score")),
        replace(disposition, missing=("other",)),
    ):
        with pytest.raises(PredictionLineageError):
            bad.validate(("equal-score", "protected", "absent", "proof"))
    with pytest.raises(PredictionLineageError, match="Duplicate"):
        disposition.validate(("equal-score", "equal-score"))
    PatchDisposition((), (), (), ()).validate(())


def test_disposition_reuses_exact_existing_patch_and_receipt_contracts() -> None:
    """No parallel patch schema; counts alone or saved-value fallback are insufficient."""
    from chipiron.environments.morpion.bootstrap.checkpoint_patch_receipts import (
        CheckpointPatchReceipt,
        PendingPatch,
    )
    from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
        MorpionReevaluationPatch,
        MorpionReevaluationPatchRow,
    )

    row = MorpionReevaluationPatchRow(
        "equal-score",
        3.0,
        is_exact=False,
        is_terminal=False,
        metadata={"source": "active_model_reevaluation"},
    )
    patch = MorpionReevaluationPatch(
        "patch-1",
        "now",
        1,
        "B",
        "models/B",
        (row,),
        tree_generation=1,
    )
    pending = PendingPatch(patch, digest("exact-patch-bytes"))
    receipt = CheckpointPatchReceipt(
        "patch-1",
        pending.content_sha256,
        1,
        1,
        "B",
        "models/B",
        B.evaluator_sha256,
        "checkpoint",
        CHECKPOINT,
        2,
        1,
        1,
        1.0,
    )
    disposition = PatchDisposition(("equal-score",), (), (), ())
    origin = replace(PATCH, evidence_sha256=disposition.evidence_sha256(receipt))
    disposition.validate_receipt(pending, receipt, origin)
    terminal = replace(
        row,
        node_id="proof",
        is_exact=True,
        is_terminal=True,
        metadata={"source": "terminal_existing_value"},
    )
    with_reuse = replace(disposition, certified_reuse=("proof",))
    reuse_pending = replace(pending, patch=replace(patch, rows=(row, terminal)))
    reuse_receipt = replace(receipt, rows_requested=2, rows_applied=2)
    reuse_origin = replace(
        origin, evidence_sha256=with_reuse.evidence_sha256(reuse_receipt)
    )
    with_reuse.validate_receipt(reuse_pending, reuse_receipt, reuse_origin)
    with pytest.raises(PredictionLineageError, match="differs from accepted receipt"):
        with_reuse.validate_receipt(
            reuse_pending, replace(reuse_receipt, rows_applied=1), reuse_origin
        )
    with pytest.raises(PredictionLineageError, match="existing terminal value"):
        with_reuse.validate_receipt(
            replace(
                reuse_pending,
                patch=replace(patch, rows=(row, replace(terminal, is_exact=False))),
            ),
            reuse_receipt,
            reuse_origin,
        )
    for invalid in (
        replace(receipt, patch_id="other"),
        replace(receipt, content_sha256=digest("other-bytes")),
        replace(receipt, evaluator_generation=2),
        replace(receipt, evaluator_name="C"),
        replace(receipt, model_bundle_path="models/C"),
        replace(receipt, rows_requested=2),
        replace(receipt, rows_applied=0),
        replace(receipt, blend_alpha=0.5),
    ):
        with pytest.raises(PredictionLineageError):
            disposition.validate_receipt(pending, invalid, origin)
    for wrong_origin in (
        replace(origin, evaluator_sha256=A.evaluator_sha256),
        replace(origin, evidence_sha256=digest("requested-model-only")),
        replace(origin, kind="genesis"),
    ):
        with pytest.raises(PredictionLineageError, match="accepted patch evidence"):
            disposition.validate_receipt(pending, receipt, wrong_origin)
    for invalid_row in (
        replace(row, metadata={"source": "snapshot_existing_values"}),
        replace(row, metadata={"source": "custom_evaluator"}),
        replace(row, is_exact=True),
        replace(row, is_terminal=True),
        replace(row, backed_up_value=3.0),
    ):
        with pytest.raises(PredictionLineageError, match="built-in direct model"):
            disposition.validate_receipt(
                replace(pending, patch=replace(patch, rows=(invalid_row,))),
                receipt,
                origin,
            )


@pytest.mark.parametrize(
    "declaration", ["mixed", "sidecar", "policy", "null-policy", "missing-sidecar"]
)
def test_dataset_and_training_refuse_b4_even_with_legacy_target_fallback(
    declaration: str,
) -> None:
    """Well-formed C2 bytes are not admission; stripping target policy cannot opt in."""
    source = provenance()
    if declaration in {"mixed", "missing-sidecar"}:
        source["scientific_contract"] = MIXED_SOURCE_CONTRACT
    if declaration in {"mixed", "sidecar"}:
        source["prediction_lineage"] = {
            "path": "prediction_lineage.json",
            "sha256": hashlib.sha256(LINEAGE.to_bytes()).hexdigest(),
        }
    if declaration in {"policy", "null-policy"}:
        source["source_policy"] = (
            "mixed_search_targets_v1" if declaration == "policy" else None
        )
    with pytest.raises(ValueError, match="B4 source provenance is not admitted"):
        DatasetConfidenceContext(lambda _node: {}, source)
    for data in (metadata(source), {"scientific_source_provenance": source}):
        with pytest.raises(ValueError, match="B4 source provenance is not admitted"):
            training_target_metadata(data)
        with pytest.raises(ValueError, match="B4 source provenance is not admitted"):
            dataset_format_version(data)
    with pytest.raises(ValueError, match="B4 source provenance is not admitted"):
        validate_model_target_provenance(
            "backup_value", {"scientific_source_provenance": source}
        )
    model = training_target_metadata(metadata(provenance()))
    model["scientific_source_provenance"] = source
    with pytest.raises(ValueError, match="B4 source provenance is not admitted"):
        validate_model_target_provenance("effective_value", model)


def test_frozen_historical_bytes_and_target_meanings() -> None:
    """No inferred lineage or recipe upgrade enters historical metadata."""
    source = provenance()
    before = json.dumps(source, sort_keys=True)
    context = DatasetConfidenceContext(lambda _node: {}, source)
    assert (
        json.dumps(context.metadata()["scientific_source_provenance"], sort_keys=True)
        == before
    )
    assert training_target_metadata({"scientific_source_provenance": source}) == {}
    validate_model_target_provenance("backup_value", {})
    assert dataset_format_version({}) == 1
    model = training_target_metadata(metadata(source))
    validate_model_target_provenance("effective_value", model)
    assert model["scientific_source_provenance"] == source
    assert json.dumps(closed_envelope("descriptor"), separators=(",", ":")) == (
        '{"format_kind":"morpion_closed_scientific_snapshot","format_version":2,'
        '"scientific_contract":"anemone-values-canonical-morpion-1",'
        '"storage_layout_version":1,"role":"descriptor"}'
    )


@pytest.mark.parametrize("closed", [False, True])
def test_existing_readers_reject_new_discriminator(
    tmp_path: Path, closed: bool
) -> None:
    """Both eager and indexed frozen readers reject B4 before reading any values."""
    from chipiron.environments.morpion.bootstrap.closed_scientific_snapshot import (
        load_closed_scientific_snapshot,
    )
    from chipiron.environments.morpion.bootstrap.sharded_training_reader import (
        ShardedTrainingGenerationView,
    )
    from chipiron.environments.morpion.bootstrap.sharded_training_v2 import (
        load_scientific_training_generation,
    )

    path = tmp_path / "generations" / "generation.json"
    path.parent.mkdir()
    payload = (
        closed_envelope("generation")
        if closed
        else {
            "format_kind": KIND,
            "format_version": 2,
            "role": "generation",
        }
    )
    payload["scientific_contract"] = MIXED_SOURCE_CONTRACT
    path.write_text(json.dumps(payload))
    (tmp_path / "manifest.json").write_text(json.dumps({**payload, "role": "manifest"}))
    with pytest.raises(ValueError, match=r"[Uu]nsupported"):
        (
            load_closed_scientific_snapshot
            if closed
            else load_scientific_training_generation
        )(path)
    with (
        pytest.raises(ValueError, match=r"[Uu]nsupported.*contract"),
        ShardedTrainingGenerationView(
            path, allow_scientific_v2=True, allow_closed_snapshot=closed
        ),
    ):
        pytest.fail("Mixed-source reader admission is not implemented")

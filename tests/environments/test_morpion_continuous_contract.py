"""In-memory foundation checks; no claim, tensor or recovery qualification."""

import copy
import json
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from typing import Any

import pytest

from chipiron.environments.morpion.bootstrap.continuous_contract import (
    Accepted,
    ContractError,
    Debt,
    Job,
    JobRecord,
    Policy,
    Ready,
    ReadyRecord,
    Ref,
    Source,
    encode_record,
    record_ref,
)
from chipiron.environments.morpion.bootstrap.continuous_transitions import (
    Boundary,
    Change,
    Frame,
    NoWork,
    admit_cohort,
    admit_dataset,
    bind_dataset,
    growth_wait,
    initialize,
    offer_source,
    publish_ready,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    object_digest,
)


def ref(path: str = "proof.json", digest: str = "a") -> Ref:
    """Make explicitly illustrative, unauthenticated descriptor evidence."""
    return Ref(path, digest * 64, 128)


def source(generation: int = 14, cycle: int = 11, epoch: int = 0) -> Source:
    """Use the contract's example source generation and cycle."""
    return Source(f"publication-{generation}", generation, "b" * 64, cycle, epoch)


def boundary(frame: Frame, cycle: int = 11) -> Boundary:
    """Supply an explicit caller boundary; never read a clock."""
    return Boundary(frame.context, frame.state.revision, cycle)


def changed(result: Change | NoWork) -> Change:
    """Require an actual proposed transition, not a silent no-work result."""
    assert isinstance(result, Change)
    return result


def extraction(frame: Frame, selected: Source) -> JobRecord:
    """Bind claim/session/budget refs without pretending they are authenticated."""
    return JobRecord(
        frame.context,
        Job(
            "dataset-d14",
            "EXTRACT",
            selected,
            ref("claim"),
            ref("launch"),
            ref("budget"),
            "ADMITTED",
            None,
            None,
            None,
            None,
            None,
            None,
        ),
    )


def prepared(frame: Frame, job: JobRecord) -> ReadyRecord:
    """Mirror CONTRACT's complete five-file prepared-ready example."""
    accepted = job.payload.accepted_dataset
    assert accepted is not None
    root = f"pipeline/prepared_inputs_v1/{accepted.descriptor_sha256}"
    entry = f"{root}/.rows.jsonl.shared-preparation-v1/entry"
    files = tuple(
        (name, ref(f"{entry}/{name}"))
        for name in (
            "flat.manifest.json",
            "flat.pt",
            "relational.manifest.json",
            "relational.pt",
            "split.json",
        )
    )
    return ReadyRecord(
        frame.context,
        Ready(
            accepted,
            job.payload.source,
            ref("dataset_publications_v1/objects/accepted/rows.jsonl"),
            ref(f"{root}/rows.jsonl"),
            "c" * 64,
            ref(f"{entry}/complete.json"),
            dict(files)["split.json"],
            files,
            job.payload.id,
        ),
    )


def train_job(
    frame: Frame, ready: ReadyRecord, parents: Ref | None = None
) -> JobRecord:
    """Capture parents at admission, independently from the producer."""
    return JobRecord(
        frame.context,
        Job(
            "train-1",
            "TRAIN",
            ready.payload.source,
            ref("training-claim"),
            ref("launch"),
            ref("training-budget"),
            "ADMITTED",
            ready.payload.accepted_dataset,
            record_ref(ready),
            parents or ref("newest-parents"),
            "cohort-1",
            frame.state.last_accepted_round + 1,
            None,
        ),
    )


def ready_frame() -> tuple[Frame, ReadyRecord]:
    """Run the real pure path through accepted Dataset and completed preparation."""
    frame = initialize("d" * 64, "e" * 64)
    selected = source()
    frame = changed(
        offer_source(frame, boundary(frame), selected, dataset=None, ready=None)
    ).frame
    job = extraction(frame, selected)
    frame = changed(admit_dataset(frame, boundary(frame), job)).frame
    result = changed(
        bind_dataset(frame, boundary(frame), job, Accepted("accepted", "f" * 64))
    )
    frame = result.frame
    preparing = result.records[0]
    assert isinstance(preparing, JobRecord)
    ready = prepared(frame, preparing)
    frame = changed(
        publish_ready(frame, boundary(frame), preparing, ready, previous_ready=None)
    ).frame
    return frame, ready


@pytest.mark.parametrize(
    "value", [True, False, -1, 1.5, float("nan"), float("inf"), "1", None]
)
def test_strict_numbers(value: Any) -> None:
    """No bool coercion, missing evidence or nonfinite numeric values."""
    with pytest.raises(ContractError):
        Ref("data", "a" * 64, value)
    with pytest.raises(ContractError):
        Source("s", value, "a" * 64, 0, 0)


@pytest.mark.parametrize(
    "path",
    [
        "",
        "/root/a",
        "../a",
        "a/../b",
        "a/./b",
        "a//b",
        "a/",
        "C:/a",
        "a\\b",
        "a\x00b",
        "a\nb",
    ],
)
def test_lexical_containment(path: str) -> None:
    """Reject lexical escapes without claiming symlink or file authentication."""
    with pytest.raises(ContractError):
        ref(path)


@pytest.mark.parametrize("digest", ["A" * 64, "x" * 64, "a" * 63, "a" * 65, "", None])
def test_digest_refusal(digest: Any) -> None:
    """References require exact lowercase SHA256 shape."""
    with pytest.raises(ContractError):
        Ref("a", digest, 0)


def test_exact_shapes_and_round_trips() -> None:
    """Every implemented wire shape refuses unknown/missing fields."""
    frame, ready = ready_frame()
    job = train_job(frame, ready)
    records = [
        Policy(),
        frame.context,
        ref(),
        Accepted("a", "a" * 64),
        source(),
        Debt("d", 1),
        frame.state,
        job.payload,
        ready.payload,
        job,
        ready,
    ]
    for record in records:
        payload = record.to_metadata()
        original = copy.deepcopy(payload)
        assert type(record).from_metadata(json.loads(json.dumps(payload))) == record
        assert payload == original
        for key in payload:
            missing = {k: v for k, v in payload.items() if k != key}
            with pytest.raises(ContractError):
                type(record).from_metadata(missing)
        with pytest.raises(ContractError):
            type(record).from_metadata({**payload, "unreviewed": 1})
    assert record_ref(ready).sha256 == object_digest(json.loads(encode_record(ready)))
    assert record_ref(ready).bytes == len(encode_record(ready))


def test_nested_detachment_and_frozen_construction() -> None:
    """Neither decoded nor encoded mutable dictionaries can alter a record."""
    _, ready = ready_frame()
    data = ready.to_metadata()
    decoded = ReadyRecord.from_metadata(data)
    data["payload"]["files"]["flat.pt"]["bytes"] = 999
    data["context"]["config_sha256"] = "0" * 64
    assert decoded == ready
    output = decoded.to_metadata()
    output["payload"]["files"].clear()
    assert decoded == ready
    with pytest.raises(FrozenInstanceError):
        decoded.payload.producer_job_id = "other"  # type: ignore[misc]
    with pytest.raises(ContractError):
        replace(decoded.payload, files=list(decoded.payload.files))  # type: ignore[arg-type]
    policy = Policy.from_metadata(Policy().to_metadata())
    assert type(policy.slots) is tuple


def test_realistic_sequence_binds_new_parents_and_preserves_other_slots() -> None:
    """Preparation has no parents; consumption binds the admitted latest map."""
    frame, ready = ready_frame()
    assert frame.state.revision == 4
    assert frame.state.dataset_job is None
    reserved, pending = ref("reserved"), ref("pending")
    frame = replace(
        frame,
        state=replace(
            frame.state,
            reserved_promotion=reserved,
            pending_promotion=pending,
            promotion_debt=Debt("promotion-oldest", 10),
        ),
    )
    before = frame.state.to_metadata()
    parents = ref("parents-after-preparation")
    job = train_job(frame, ready, parents)
    result = changed(
        admit_cohort(frame, boundary(frame), ready, job, validation_count=2)
    )
    assert frame.state.to_metadata() == before
    assert result.frame.state.ready is None
    assert result.frame.state.cohort == record_ref(job)
    assert result.frame.state.last_consumed_generation == 14
    assert result.frame.state.last_accepted_round == 0
    assert result.frame.state.reserved_promotion == reserved
    assert result.frame.state.pending_promotion == pending
    assert result.frame.state.promotion_debt == frame.state.promotion_debt
    assert result.frame.state.learning_debt == frame.state.learning_debt
    assert result.records == (job,)
    assert job.payload.parents == parents
    assert admit_cohort(
        result.frame, boundary(result.frame), ready, job, validation_count=2
    ) == NoWork("training_slot_full")


def test_coalescing_preserves_admitted_job_and_original_debt() -> None:
    """New pending sources cannot replace an admitted source or slide the gate."""
    frame = initialize("d" * 64, "e" * 64)
    first = source(10, 5)
    frame = changed(
        offer_source(frame, boundary(frame, 5), first, dataset=None, ready=None)
    ).frame
    job = extraction(frame, first)
    frame = changed(admit_dataset(frame, boundary(frame, 5), job)).frame
    frame = changed(
        offer_source(frame, boundary(frame, 6), source(11, 6), dataset=job, ready=None)
    ).frame
    result = changed(
        offer_source(frame, boundary(frame, 7), source(12, 7), dataset=job, ready=None)
    )
    assert result.displacement is not None
    assert result.displacement.reason == "coalesced_before_admission"
    assert result.displacement.subject == source(11, 6)
    assert result.frame.state.dataset_job == record_ref(job)
    assert result.frame.state.learning_debt == Debt(first.publication_id, 5)
    assert job.payload.source == first
    assert admit_dataset(
        result.frame, boundary(result.frame, 7), extraction(frame, source(12, 7))
    ) == NoWork("dataset_slot_full")
    assert offer_source(
        result.frame, boundary(result.frame, 7), source(11, 6), dataset=job, ready=None
    ) == NoWork("generation_not_new")


def test_context_revision_and_resolved_record_fences() -> None:
    """Equal policy hashes do not permit cross-workspace or stale-state proposals."""
    frame, ready = ready_frame()
    job = train_job(frame, ready)
    for bad in [
        replace(boundary(frame), revision=0),
        replace(
            boundary(frame), context=replace(frame.context, config_sha256="0" * 64)
        ),
    ]:
        with pytest.raises(ContractError):
            admit_cohort(frame, bad, ready, job, validation_count=2)
    with pytest.raises(ContractError):
        admit_cohort(
            frame,
            boundary(frame),
            replace(
                ready, context=replace(frame.context, workspace_epoch_sha256="0" * 64)
            ),
            job,
            validation_count=2,
        )
    with pytest.raises(ContractError):
        growth_wait(frame, boundary(frame), dataset=None, ready=None)


@pytest.mark.parametrize(
    ("age", "epoch", "reason"),
    [(8, 1, None), (9, 1, "source_age"), (8, 2, "teacher_lag")],
)
def test_age_and_epoch_boundaries(age: int, epoch: int, reason: str | None) -> None:
    """Check freshness at admission, preserving accepted data on refusal."""
    frame, ready = ready_frame()
    frame = replace(frame, state=replace(frame.state, incumbent_epoch=epoch))
    result = admit_cohort(
        frame,
        boundary(frame, 11 + age),
        ready,
        train_job(frame, ready),
        validation_count=2,
    )
    if reason is None:
        assert isinstance(result, Change)
    else:
        assert isinstance(result, NoWork) and result.reason == reason
        assert frame.state.ready == record_ref(ready)


def test_gate_and_frozen_head_repair_progress() -> None:
    """Age/debt and increasing generation allow finite repair, not a moving gate."""
    frame = initialize("d" * 64, "e" * 64)
    first = source(14, 2)
    frame = changed(
        offer_source(frame, boundary(frame, 2), first, dataset=None, ready=None)
    ).frame
    assert growth_wait(frame, boundary(frame, 9), dataset=None, ready=None) is None
    assert growth_wait(frame, boundary(frame, 10), dataset=None, ready=None) == NoWork(
        "learning_debt"
    )
    fresh = source(15, 11)
    assert offer_source(
        frame, boundary(frame, 11), fresh, dataset=None, ready=None
    ) == NoWork("freshness_disposition_required")
    fixed = changed(
        offer_source(
            frame,
            boundary(frame, 11),
            fresh,
            dataset=None,
            ready=None,
            sealed_head=fresh,
        )
    ).frame
    assert fixed.state.learning_debt == frame.state.learning_debt
    assert isinstance(
        admit_dataset(fixed, boundary(fixed, 11), extraction(fixed, fresh)), Change
    )
    # No newer unconsumed committed generation: explicit finite refusal.
    consumed = replace(
        frame,
        state=replace(frame.state, pending_source=None, last_consumed_generation=15),
    )
    assert offer_source(
        consumed,
        boundary(consumed, 11),
        fresh,
        dataset=None,
        ready=None,
        sealed_head=fresh,
    ) == NoWork("generation_not_new")


def test_no_validation_and_suspended_are_no_work() -> None:
    """Tiny validation populations never admit scientifically unusable Training."""
    frame, ready = ready_frame()
    job = train_job(frame, ready)
    assert admit_cohort(
        frame, boundary(frame), ready, job, validation_count=0
    ) == NoWork("empty_validation_population")
    with pytest.raises(ContractError):
        admit_cohort(frame, boundary(frame), ready, job, validation_count=True)
    stopped = replace(frame, state=replace(frame.state, admission_open=False))
    assert admit_cohort(
        stopped, boundary(stopped), ready, job, validation_count=2
    ) == NoWork("suspended")


def test_ready_binding_and_stable_path_refusals() -> None:
    """Structural proof does not silently accept a changed inventory or rows."""
    frame, ready = ready_frame()
    for field, value in [
        ("rows", ref("wrong")),
        ("split", ref("wrong")),
        ("recipe_sha256", "bad"),
        ("files", ready.payload.files[:-1]),
    ]:
        with pytest.raises(ContractError):
            replace(ready.payload, **{field: value})
    data = ready.to_metadata()
    data["payload"]["files"]["wrong.pt"] = data["payload"]["files"].pop("flat.pt")
    with pytest.raises(ContractError):
        ReadyRecord.from_metadata(data)
    job = train_job(frame, ready)
    for bad in [
        replace(job.payload, source=source(15)),
        replace(job.payload, accepted_dataset=Accepted("other", "f" * 64)),
        replace(job.payload, ready=ref("other")),
        replace(job.payload, round_number=2),
    ]:
        with pytest.raises(ContractError):
            admit_cohort(
                frame,
                boundary(frame),
                ready,
                JobRecord(frame.context, bad),
                validation_count=1,
            )


def test_publication_checks_job_and_acceptance_and_preserves_pending() -> None:
    """No incomplete or unrelated producer releases the preparation slot."""
    frame = initialize("d" * 64, "e" * 64)
    frame = changed(
        offer_source(frame, boundary(frame), source(), dataset=None, ready=None)
    ).frame
    extract = extraction(frame, source())
    frame = changed(admit_dataset(frame, boundary(frame), extract)).frame
    accepted = changed(
        bind_dataset(frame, boundary(frame), extract, Accepted("accepted", "f" * 64))
    )
    frame = accepted.frame
    job = accepted.records[0]
    assert isinstance(job, JobRecord)
    ready = prepared(frame, job)
    frame = changed(
        offer_source(
            frame, boundary(frame, 12), source(16, 12), dataset=job, ready=None
        )
    ).frame
    for bad in [
        replace(ready.payload, producer_job_id="other"),
        replace(ready.payload, source=source(15)),
        replace(ready.payload, accepted_dataset=Accepted("other", "f" * 64)),
    ]:
        with pytest.raises(ContractError):
            publish_ready(
                frame,
                boundary(frame, 12),
                job,
                ReadyRecord(frame.context, bad),
                previous_ready=None,
            )
    with pytest.raises(ContractError):
        bind_dataset(
            frame, boundary(frame, 12), extract, Accepted("accepted", "f" * 64)
        )
    result = changed(
        publish_ready(frame, boundary(frame, 12), job, ready, previous_ready=None)
    )
    assert result.frame.state.pending_source == source(16, 12)
    assert result.frame.state.dataset_job is None


def test_unsupported_jobs_policies_and_existing_dispatch() -> None:
    """Foundation types do not expose a runtime policy or partial future stage."""
    from chipiron.environments.morpion.bootstrap.learning_schedule import (
        BoundedGrowthWindow,
        ReadyBoundedGrowthWindow,
        context_marker,
        schedule_from_metadata,
    )

    with pytest.raises(ValueError):
        schedule_from_metadata({"learning_schedule": Policy().to_metadata()})
    assert context_marker(None) == {"schema_version": 1}
    assert context_marker(BoundedGrowthWindow(4))["schema_version"] == 2
    assert context_marker(ReadyBoundedGrowthWindow(8))["schema_version"] == 3
    frame = initialize("d" * 64, "e" * 64)
    with pytest.raises(ContractError):
        replace(extraction(frame, source()).payload, stage="COMPARE")
    with pytest.raises(ContractError):
        replace(Policy(), max_source_age_cycles=9)
    with pytest.raises(ContractError):
        replace(Policy(), slots=(True, 1, 1, 1, 1, 1))


def test_leaf_import_does_not_load_torch() -> None:
    """A clean interpreter imports the foundation without any neural runtime."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from chipiron.environments.morpion.bootstrap.continuous_transitions import initialize; initialize('a'*64, 'b'*64); assert 'torch' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_new_ready_replaces_only_unconsumed_pointer_with_receipt() -> None:
    """Old ready proof remains immutable while its pointer is explicitly replaced."""
    frame, first = ready_frame()
    frame = changed(
        offer_source(
            frame, boundary(frame, 12), source(16, 12), dataset=None, ready=first
        )
    ).frame
    extract = extraction(frame, source(16, 12))
    frame = changed(admit_dataset(frame, boundary(frame, 12), extract)).frame
    result = changed(
        bind_dataset(
            frame, boundary(frame, 12), extract, Accepted("accepted-next", "1" * 64)
        )
    )
    frame = result.frame
    job = result.records[0]
    assert isinstance(job, JobRecord)
    second = prepared(frame, job)
    result = changed(
        publish_ready(frame, boundary(frame, 12), job, second, previous_ready=first)
    )
    assert result.displacement is not None
    assert result.displacement.reason == "ready_superseded_unconsumed"
    assert result.displacement.subject == record_ref(first)
    assert result.frame.state.ready == record_ref(second)
    assert result.frame.state.learning_debt == frame.state.learning_debt


def test_builder_can_finish_old_epoch_but_no_takeover_or_consumption() -> None:
    """Acceptance/preparation finish pinned work; freshness gates new consumption."""
    frame = initialize("d" * 64, "e" * 64)
    frame = changed(
        offer_source(frame, boundary(frame), source(), dataset=None, ready=None)
    ).frame
    job = extraction(frame, source())
    frame = changed(admit_dataset(frame, boundary(frame), job)).frame
    # A later, separately implemented activation has changed the state epoch.
    frame = replace(frame, state=replace(frame.state, incumbent_epoch=2))
    result = changed(
        bind_dataset(frame, boundary(frame, 19), job, Accepted("accepted", "f" * 64))
    )
    frame = result.frame
    preparing = result.records[0]
    assert isinstance(preparing, JobRecord)
    ready = prepared(frame, preparing)
    result = changed(
        publish_ready(frame, boundary(frame, 19), preparing, ready, previous_ready=None)
    )
    frame = result.frame
    assert admit_cohort(
        frame, boundary(frame, 19), ready, train_job(frame, ready), validation_count=2
    ) == NoWork("teacher_lag")
    assert offer_source(
        frame,
        boundary(frame, 19),
        source(16, 19, 2),
        dataset=None,
        ready=ready,
        sealed_head=source(16, 19, 2),
    ) == NoWork("freshness_disposition_required")


def test_promotion_gate_future_clocks_and_consumed_generation() -> None:
    """Opaque promotion bodies cannot lose their debt or evade the cycle gate."""
    frame, ready = ready_frame()
    frame = replace(
        frame,
        state=replace(
            frame.state,
            promotion_debt=Debt("old", 10),
            reserved_promotion=ref("reserved"),
        ),
    )
    assert growth_wait(frame, boundary(frame, 11), dataset=None, ready=ready) is None
    assert growth_wait(frame, boundary(frame, 12), dataset=None, ready=ready) == NoWork(
        "promotion_debt"
    )
    with pytest.raises(ContractError):
        growth_wait(frame, boundary(frame, 9), dataset=None, ready=ready)
    with pytest.raises(ContractError):
        offer_source(frame, boundary(frame), source(20, 12), dataset=None, ready=ready)
    frame = replace(frame, state=replace(frame.state, last_consumed_generation=14))
    assert admit_cohort(
        frame, boundary(frame), ready, train_job(frame, ready), validation_count=2
    ) == NoWork("generation_not_new")

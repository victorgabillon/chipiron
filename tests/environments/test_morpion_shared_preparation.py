"""Exact shared inputs, private ownership, failure and recovery boundaries."""

import json
import multiprocessing
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
import torch

from chipiron.environments.morpion.bootstrap.candidate_job import CandidateJob
from chipiron.environments.morpion.bootstrap.candidate_preparation import (
    prepare_candidate_inputs,
)
from chipiron.environments.morpion.learning import tree_to_dataset as reader
from chipiron.environments.morpion.players.evaluators.neural_networks.training import (
    MorpionStreamingTrainingArgs,
    MorpionTrainingArgs,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training import (
    flat_tensor_cache as flat,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training import (
    relational_entity_token_cache as relational,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training import (
    shared_preparation as shared,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training import (
    shared_preparation_builder as builder,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training import (
    state_split as split,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training import (
    state_split_cache as split_cache,
)
from chipiron.environments.morpion.types import MorpionDynamics
from tests.environments.test_morpion_continual_training import rows_file


@pytest.fixture(autouse=True)
def isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fence protocol fixtures; installed validation separately hashes real semantics."""
    monkeypatch.delenv("CHIPIRON_STATE_SPLIT_CACHE_DIR", raising=False)
    monkeypatch.delenv(shared.ENVIRONMENT, raising=False)
    monkeypatch.setattr(shared, "semantic_fingerprint", lambda: "a" * 64)
    monkeypatch.setattr(split_cache, "semantic_fingerprint", lambda: "a" * 64)
    split._split.cache_clear()
    split_cache._cached.cache_clear()


def prepare(rows: Path, limit: int | None = None, **kwargs: Any) -> Any:
    """Open the real producer with small bounded staging buffers."""
    return shared.prepare_shared_inputs(
        rows, fraction=0.2, seed=17, max_rows=limit, row_chunk_size=3, **kwargs
    )


def root(rows: Path) -> Path:
    """Locate private attempt-owned preparation storage."""
    return rows.with_name(f".{rows.name}.shared-preparation-v1")


def native(
    rows: Path, limit: int | None
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], split.StateSplit]:
    """Run the original independent materializers as an exact oracle."""
    f = flat.load_or_materialize_flat_tensor_cache(
        rows_path=rows, row_chunk_size=3, max_rows=limit
    )
    r = relational.load_or_materialize_relational_entity_token_cache(
        rows_path=rows, row_chunk_size=3, max_rows=limit, entity_max_tokens=1536
    )
    tensors = (
        torch.load(f.paths.tensor_path, weights_only=True),
        torch.load(r.paths.tensor_path, weights_only=True),
    )
    value = split._split.__wrapped__(
        str(rows), shared.file_digest(rows), 0.2, 17, limit
    )
    return *tensors, value


@pytest.mark.fast
@pytest.mark.parametrize("limit", [None, 0, 1, 9, 16, 25])
def test_exact_outputs_one_decode_and_warm_hit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: int | None
) -> None:
    """Every tensor, dtype, order and population matches native; warm hit replays nothing."""
    baseline = rows_file(tmp_path / "native")
    rows = rows_file(tmp_path / "shared")
    # Include a duplicate to preserve positional membership and duplicate targets.
    for path in (baseline, rows):
        lines = path.read_text().splitlines()
        path.write_text("\n".join([*lines, lines[1]]) + "\n")
    expected_flat, expected_relational, expected_split = native(baseline, limit)
    original = reader._decode_validated_payload
    calls = 0

    def decode(value: object) -> Any:
        nonlocal calls
        calls += 1
        return original(value)

    monkeypatch.setattr(reader, "_decode_validated_payload", decode)
    with prepare(rows, limit) as prepared:
        assert prepared is not None
        assert prepared.split == expected_split
        for path, expected in (
            (prepared.flat_paths.tensor_path, expected_flat),
            (prepared.relational_paths.tensor_path, expected_relational),
        ):
            actual = torch.load(path, weights_only=True)
            assert actual.keys() == expected.keys()
            for name, tensor in expected.items():
                assert actual[name].dtype == tensor.dtype
                assert torch.equal(actual[name], tensor), name
    assert calls == (17 if limit is None else min(limit + 1, 17))
    with prepare(rows, limit) as hit:
        assert hit is not None and hit.split == expected_split
    assert calls == (17 if limit is None else min(limit + 1, 17))
    assert json.loads((root(rows) / "operation.json").read_text())["outcome"] == "hit"


@pytest.mark.fast
@pytest.mark.parametrize("persistent", [False, True])
def test_split_handoff_uses_original_api_without_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, persistent: bool
) -> None:
    """A miss seeds the existing split API; scope and scientific args remain unchanged."""
    rows = rows_file(tmp_path / "input")
    args = MorpionTrainingArgs(
        rows, tmp_path / "model", stable_validation_split=True, validation_seed=17
    )
    expected = split.prepare_state_split(args)
    original = asdict(args)
    split._split.cache_clear()
    if persistent:
        monkeypatch.setenv("CHIPIRON_STATE_SPLIT_CACHE_DIR", str(tmp_path / "splits"))

    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("native split replayed after shared decoding")

    monkeypatch.setattr(split, "_native_identified_rows", refuse)
    with prepare(rows) as prepared:
        assert prepared is not None
        assert prepared.prepare_split(args, None) == expected
        assert split.prepare_state_split(args) == expected
        assert split._SUPPLIED.get() is None
        with pytest.raises(ValueError, match="recipe"):
            prepared.prepare_split(replace(args, validation_seed=18), None)
    with pytest.raises(ValueError, match="lease"):
        prepared.prepare_split(args, None)
    assert asdict(args) == original


@pytest.mark.fast
def test_split_handoff_resets_after_publication_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A persistent-cache failure cannot leak a supplied result into later work."""
    rows = rows_file(tmp_path / "input")
    args = MorpionTrainingArgs(
        rows, tmp_path / "model", stable_validation_split=True, validation_seed=17
    )
    with prepare(rows) as prepared:
        assert prepared is not None
        monkeypatch.setenv("CHIPIRON_STATE_SPLIT_CACHE_DIR", "")
        with pytest.raises(ValueError, match="must name"):
            prepared.prepare_split(args, None)
        assert split._SUPPLIED.get() is None
        monkeypatch.delenv("CHIPIRON_STATE_SPLIT_CACHE_DIR")
        prepared.prepare_split(args, None)


@pytest.mark.fast
@pytest.mark.parametrize("available", ["flat", "relational", "both"])
def test_existing_native_partial_or_warm_cache_remains_hit(
    tmp_path: Path, available: str
) -> None:
    """Legacy partial availability falls back without deleting or rebuilding valid input."""
    rows = rows_file(tmp_path / "input")
    if available in {"flat", "both"}:
        flat.load_or_materialize_flat_tensor_cache(
            rows_path=rows, row_chunk_size=3, max_rows=None
        )
    if available in {"relational", "both"}:
        relational.load_or_materialize_relational_entity_token_cache(
            rows_path=rows, row_chunk_size=3, max_rows=None, entity_max_tokens=1536
        )
    before = {
        p: (p.stat().st_mtime_ns, shared.file_digest(p))
        for p in rows.parent.glob("tensor_cache/*")
    }
    with prepare(rows) as prepared:
        assert prepared is None
    assert before == {p: (p.stat().st_mtime_ns, shared.file_digest(p)) for p in before}
    assert not (root(rows) / "entry").exists()
    assert (
        json.loads((root(rows) / "operation.json").read_text())["outcome"]
        == "native-warm-fallback"
    )


@pytest.mark.fast
@pytest.mark.parametrize(
    "damage", ["truncate", "symlink", "extra", "split", "completion"]
)
def test_corrupt_completed_bundle_refuses(tmp_path: Path, damage: str) -> None:
    """A completed bundle is never partially accepted or silently repaired."""
    rows = rows_file(tmp_path / "input")
    with prepare(rows):
        pass
    entry = root(rows) / "entry"
    if damage == "truncate":
        (entry / "flat.pt").write_bytes(b"broken")
    elif damage == "symlink":
        (entry / "flat.pt").unlink()
        (entry / "flat.pt").symlink_to(rows)
    elif damage == "extra":
        (entry / "extra").touch()
    elif damage == "split":
        (entry / "split.json").write_text("{}")
    else:
        (entry / "complete.json").write_text('{"format":"x","format":"y"}')
    with pytest.raises(ValueError), prepare(rows):
        pytest.fail("corrupt completed bundle accepted")


@pytest.mark.fast
@pytest.mark.parametrize(
    "budget",
    [
        shared._Budget(packed_bytes=1),
        shared._Budget(scratch_bytes=1),
        shared._Budget(chunk_bytes=1),
    ],
)
def test_resource_stop_never_truncates_or_completes(
    tmp_path: Path, budget: shared._Budget
) -> None:
    """A byte limit stops the attempt, never changes the scientific row limit."""
    rows = rows_file(tmp_path / "input")
    with pytest.raises(ValueError, match="budget"), prepare(rows, budget=budget):
        pytest.fail("insufficient budget accepted")
    assert not (root(rows) / "entry").exists()
    with prepare(rows) as prepared:
        assert prepared is not None and prepared.split.row_count == 16
    assert not list(root(rows).glob("attempt-*"))


@pytest.mark.fast
@pytest.mark.parametrize("damage", ["target", "state", "metadata", "envelope"])
def test_validation_error_in_first_excluded_row_matches_native(
    tmp_path: Path, damage: str
) -> None:
    """Finite limits retain the native split's validation of the first excluded row."""
    rows = rows_file(tmp_path / "input")
    records = [json.loads(line) for line in rows.read_text().splitlines()]
    if damage == "target":
        records[2]["row"].pop("target_value")
    elif damage == "state":
        records[2]["row"]["state_ref_payload"] = {}
    elif damage == "metadata":
        records[2]["row"]["node_id"] = 7
    else:
        records[2]["kind"] = "unknown"
    rows.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    with pytest.raises((TypeError, ValueError)) as native_error:
        split._split.__wrapped__(str(rows), shared.file_digest(rows), 0.2, 17, 1)
    with pytest.raises(type(native_error.value)) as shared_error, prepare(rows, 1):
        pytest.fail("invalid excluded row accepted")
    assert str(shared_error.value) == str(native_error.value)
    assert not (root(rows) / "entry").exists()


@pytest.mark.fast
def test_converters_do_not_mutate_shared_state(tmp_path: Path) -> None:
    """The immutable wrapper and underlying certified state retain exact values."""
    rows = rows_file(tmp_path / "input")
    dynamics = MorpionDynamics()
    flat_converter = builder.MorpionFeatureTensorConverter(
        dynamics, builder.full_morpion_feature_subset()
    )
    relational_converter = builder.MorpionRelationalEntityTokenConverter(
        dynamics=dynamics, max_tokens=1536
    )
    for validated in reader._iter_validated_morpion_jsonl_rows(rows):
        state = dynamics.wrap_atomheart_state(validated.state)
        before = asdict(state)
        atom_before = repr(validated.state)
        flat_converter.state_to_tensor(state)
        relational_converter.state_to_tensors(state)
        assert asdict(state) == before
        assert repr(validated.state) == atom_before


def process_prepare(
    rows: str, ready: str, release: str, result: str, kill_during_build: bool
) -> None:
    """Use a fresh process with real filesystem locking and crashable private staging."""
    shared.semantic_fingerprint = lambda: "a" * 64
    if kill_during_build:
        save = builder._save

        def staged(path: Path, payload: object) -> None:
            save(path, payload)
            Path(ready).touch()
            while not Path(release).exists():
                time.sleep(0.01)

        builder._save = staged
    with prepare(Path(rows)) as prepared:
        assert prepared is not None
        Path(ready).touch()
        while not Path(release).exists():
            time.sleep(0.01)
        Path(result).write_text(str(prepared.split.row_count))


def await_file(path: Path, process: Any) -> None:
    """Wait boundedly for a real child observation, diagnosing early child failures."""
    deadline = time.monotonic() + 45
    while not path.exists():
        assert process.is_alive(), process.exitcode
        assert time.monotonic() < deadline, str(path)
        time.sleep(0.02)


@pytest.mark.parametrize("crash", [False, True])
def test_owned_process_death_and_concurrent_consumer_exclusion(
    tmp_path: Path, crash: bool
) -> None:
    """A second process waits; killing a builder reaps it before retry cleans staging."""
    rows = rows_file(tmp_path / "input")
    context = multiprocessing.get_context("spawn")
    children = []
    try:
        for index in range(2):
            child = context.Process(
                target=process_prepare,
                args=(
                    str(rows),
                    str(tmp_path / f"ready{index}"),
                    str(tmp_path / f"release{index}"),
                    str(tmp_path / f"result{index}"),
                    crash and index == 0,
                ),
            )
            child.start()
            children.append(child)
            if index == 0:
                await_file(tmp_path / "ready0", child)
        time.sleep(0.2)
        assert not (tmp_path / "ready1").exists()
        if crash:
            children[0].kill()
        else:
            (tmp_path / "release0").touch()
        children[0].join(15)
        assert not children[0].is_alive()
        await_file(tmp_path / "ready1", children[1])
        assert not list(root(rows).glob("attempt-*"))
        (tmp_path / "release1").touch()
        children[1].join(15)
        assert children[1].exitcode == 0
        assert (tmp_path / "result1").read_text() == "16"
    finally:
        for child in children:
            if child.is_alive():
                child.kill()
            child.join(15)
            assert not child.is_alive()


@pytest.mark.fast
def test_operational_cohort_gate_and_unchanged_scientific_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default and unsupported cohorts retain the original trainer; qualified opt-in shares."""
    rows = rows_file(tmp_path / "input")
    jobs = [
        CandidateJob(
            kind,
            MorpionStreamingTrainingArgs(
                MorpionTrainingArgs(
                    rows,
                    tmp_path / kind,
                    model_kind=kind,
                    stable_validation_split=True,
                    entity_relation_schema="morpion_entity_relations_v1"
                    if kind != "linear"
                    else None,
                    entity_relation_type_count=16 if kind != "linear" else None,
                ),
                row_chunk_size=3,
                max_rows=None,
            ),
        )
        for kind in ("linear", "relation_biased_entity_token_transformer_value_net")
    ]
    original = asdict(jobs[0])
    with prepare_candidate_inputs(jobs, serial=True) as disabled:
        assert disabled is None
    assert not root(rows).exists()
    monkeypatch.setenv(shared.ENVIRONMENT, "1")
    with prepare_candidate_inputs(jobs, serial=False) as parallel:
        assert parallel is None
    with prepare_candidate_inputs(jobs[:1], serial=True) as flat_only:
        assert flat_only is None
    with prepare_candidate_inputs(jobs, serial=True) as prepared:
        assert prepared is not None
    assert asdict(jobs[0]) == original


@pytest.mark.fast
@pytest.mark.parametrize(
    "stage", ["chunk_write", "fsync", "publish", "native_validation"]
)
def test_failed_build_never_exposes_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    """Write, durability, publication and consumer validation errors all leave a retryable attempt."""
    rows = rows_file(tmp_path / "input")

    def fail(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("injected preparation failure")  # noqa: TRY003

    with monkeypatch.context() as patch:
        if stage == "chunk_write":
            patch.setattr(builder, "_save", fail)
        elif stage == "fsync":
            patch.setattr(builder.os, "fsync", fail)
        elif stage == "publish":
            patch.setattr(shared, "publish_directory_if_absent", fail)
        else:
            patch.setattr(shared, "_validate_native", fail)
        with pytest.raises(OSError, match="injected"), prepare(rows):
            pytest.fail("failed preparation exposed a consumer")
    assert not (root(rows) / "entry").exists()
    with prepare(rows) as prepared:
        assert prepared is not None and prepared.split.row_count == 16


@pytest.mark.fast
@pytest.mark.parametrize("change", ["target", "order", "source", "seed", "semantics"])
def test_content_order_settings_and_semantics_fence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    """Current labels and source order cannot be hidden by a stale prepared bundle."""
    rows = rows_file(tmp_path / "input")
    with prepare(rows):
        pass
    before = json.loads((root(rows) / "operation.json").read_text())["key"]
    records = [json.loads(line) for line in rows.read_text().splitlines()]
    seed = 17
    if change == "target":
        records[1]["row"]["target_value"] += 7
    elif change == "source":
        records[1]["row"]["metadata"]["target_source"] = "new-source"
    elif change == "order":
        records[1:] = reversed(records[1:])
    elif change == "seed":
        seed = 19
    else:
        monkeypatch.setattr(shared, "semantic_fingerprint", lambda: "b" * 64)
    if change in {"target", "source", "order"}:
        rows.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    expected = split._split.__wrapped__(
        str(rows), shared.file_digest(rows), 0.2, seed, None
    )
    with shared.prepare_shared_inputs(
        rows, fraction=0.2, seed=seed, max_rows=None, row_chunk_size=3
    ) as prepared:
        assert prepared is not None and prepared.split == expected
    operation = json.loads((root(rows) / "operation.json").read_text())
    assert operation["key"] != before and operation["outcome"] == "miss"


@pytest.mark.fast
def test_source_change_during_consumption_refuses_and_releases(tmp_path: Path) -> None:
    """An input mutation cannot return a successful preparation context or live descriptor."""
    rows = rows_file(tmp_path / "input")
    with pytest.raises(ValueError, match="source changed"), prepare(rows) as prepared:
        assert prepared is not None
        with rows.open("a") as stream:
            stream.write("\n")
    assert prepared is not None and not prepared.lease.active
    with prepare(rows) as retry:
        assert retry is not None and retry.split.row_count == 16


@pytest.fixture(scope="module")
def scientific_rows(tmp_path_factory: pytest.TempPathFactory) -> dict[bool, Any]:
    """Use real terminal/backup proof exported by the scientific confidence audit."""
    from chipiron.environments.morpion.bootstrap import MorpionBootstrapArgs
    from chipiron.environments.morpion.bootstrap.sharded_training_v2 import (
        load_scientific_training_generation,
    )
    from docs.research.dataset_confidence_v2.audit import generated_cases
    from tests.environments.test_morpion_dataset_target_confidence import extract

    work = tmp_path_factory.mktemp("shared-confidence")
    cases, _ = generated_cases(work)
    case = next(c for c in cases if c.name == "actual-endgame" and c.generation == 2)
    oracle = load_scientific_training_generation(case.path)
    return {
        backup: extract(
            MorpionBootstrapArgs(
                work_dir=work,
                dataset_confidence_policy="selected_value_v2",
                use_backed_up_value=backup,
            ),
            oracle,
        )
        for backup in (False, True)
    }


@pytest.mark.parametrize("backup", [False, True])
def test_exact_certified_terminal_and_backup_outputs(
    tmp_path: Path, scientific_rows: dict[bool, Any], backup: bool
) -> None:
    """Certified targets and all compatible encodings remain exact for terminal/backup rows."""
    from chipiron.environments.morpion.learning import (
        save_morpion_supervised_rows_streaming,
    )

    paths = [tmp_path / name / "rows.jsonl" for name in ("native", "shared")]
    rows = scientific_rows[backup]
    for path in paths:
        save_morpion_supervised_rows_streaming(
            rows=iter(rows.rows), metadata=rows.metadata, path=path
        )
    f, r, membership = native(paths[0], None)
    with prepare(paths[1]) as prepared:
        assert prepared is not None and prepared.split == membership
        for path, expected in (
            (prepared.flat_paths.tensor_path, f),
            (prepared.relational_paths.tensor_path, r),
        ):
            actual = torch.load(path, weights_only=True)
            assert all(torch.equal(actual[k], v) for k, v in expected.items())


@pytest.mark.parametrize(
    "damage",
    ["promote", "source", "missing", "contract", "version", "envelope", "replay"],
)
def test_corrupt_scientific_certificate_retains_native_error(
    tmp_path: Path, scientific_rows: dict[bool, Any], damage: str
) -> None:
    """A validated-state handoff never bypasses confidence or legal checkpoint replay."""
    from chipiron.environments.morpion.learning import (
        save_morpion_supervised_rows_streaming,
    )

    path = tmp_path / "rows.jsonl"
    rows = scientific_rows[False]
    save_morpion_supervised_rows_streaming(
        rows=iter(rows.rows), metadata=rows.metadata, path=path
    )
    records = [json.loads(line) for line in path.read_text().splitlines()]
    row = records[1]["row"]
    confidence = row["metadata"]["target_confidence"]
    if damage == "promote":
        confidence["training_target_certainty"] = "FORCED"
        row["is_exact"] = True
    elif damage == "source":
        confidence["raw_target_source"] = "tree_value"
    elif damage == "missing":
        del confidence["raw_target_certainty"]
    elif damage == "contract":
        records[0]["metadata"]["dataset_confidence_contract"] = "future"
    elif damage == "version":
        records[0]["metadata"]["dataset_version"] = 3
    elif damage == "envelope":
        records[0]["format_version"] = 1
    else:
        row["state_ref_payload"][1].append(row["state_ref_payload"][1][-1])
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    with pytest.raises((TypeError, ValueError)) as expected:
        split._split.__wrapped__(str(path), shared.file_digest(path), 0.2, 17, None)
    with pytest.raises(type(expected.value)) as actual, prepare(path):
        pytest.fail("invalid scientific row accepted")
    assert str(actual.value) == str(expected.value)


@pytest.mark.fast
def test_complete_rename_then_failure_is_retryable_hit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure after atomic rename requires revalidation and durability on retry."""
    rows = rows_file(tmp_path / "input")
    original = shared.publish_directory_if_absent

    def published(staging: Path, final: Path, work: Path) -> None:
        original(staging, final, work)
        raise OSError("post-rename failure")  # noqa: TRY003

    with monkeypatch.context() as patch:
        patch.setattr(shared, "publish_directory_if_absent", published)
        with pytest.raises(OSError, match="post-rename"), prepare(rows):
            pytest.fail("failed publication exposed a consumer")
    assert (root(rows) / "entry").is_dir()
    with prepare(rows) as prepared:
        assert prepared is not None and prepared.split.row_count == 16
    assert json.loads((root(rows) / "operation.json").read_text())["outcome"] == "hit"


@pytest.mark.fast
def test_immutable_dataset_input_cannot_gain_preparation_files(tmp_path: Path) -> None:
    """Accepted Dataset ownership is preserved before any cache directory is created."""
    source = rows_file(tmp_path / "source")
    rows = tmp_path / "dataset_publications_v1" / "objects" / "accepted" / "rows.jsonl"
    rows.parent.mkdir(parents=True)
    rows.write_bytes(source.read_bytes())
    before = {p: shared.file_digest(p) for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="immutable"), prepare(rows):
        pytest.fail("immutable input accepted as writable attempt storage")
    assert before == {
        p: shared.file_digest(p) for p in tmp_path.rglob("*") if p.is_file()
    }
    assert not root(rows).exists()


@pytest.mark.fast
@pytest.mark.parametrize("persistent", [False, True])
def test_relative_source_spelling_uses_exact_split_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, persistent: bool
) -> None:
    """Native LRU spelling and persistent-cache resolved spelling receive the same verified split."""
    rows = rows_file(tmp_path / "input")
    monkeypatch.chdir(tmp_path)
    if persistent:
        monkeypatch.setenv("CHIPIRON_STATE_SPLIT_CACHE_DIR", str(tmp_path / "splits"))
    args = MorpionTrainingArgs(
        Path("input/rows.jsonl"),
        tmp_path / "model",
        stable_validation_split=True,
        validation_seed=17,
    )
    with prepare(rows) as prepared:
        assert prepared is not None
        bound, metadata = prepared.prepare_split(args, None)
        assert bound.validation_indices == prepared.split.validation_indices
        assert (
            metadata["validation_population_sha256"]
            == prepared.split.validation_population_sha256
        )


@pytest.mark.fast
def test_host_memory_stop_before_decoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reserve actual host/container capacity before allocating chunk buffers."""
    rows = rows_file(tmp_path / "input")
    monkeypatch.setattr(shared, "available_memory_mb", lambda: 0)
    with pytest.raises(ValueError, match="memory headroom"), prepare(rows):
        pytest.fail("insufficient host capacity accepted")
    assert not (root(rows) / "entry").exists()


@pytest.mark.fast
@pytest.mark.parametrize(
    "setting", ["split", "seed", "fraction", "limit", "chunk", "tokens", "format"]
)
def test_other_valid_cohort_settings_use_existing_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, setting: str
) -> None:
    """Only explicitly qualified common settings reach the shared producer."""
    rows = rows_file(tmp_path / "input")
    flat_args = MorpionTrainingArgs(
        rows, tmp_path / "flat", stable_validation_split=True
    )
    relational_args = replace(
        flat_args,
        output_dir=tmp_path / "relational",
        model_kind="relation_biased_entity_token_transformer_value_net",
        entity_relation_schema="morpion_entity_relations_v1",
        entity_relation_type_count=16,
    )
    if setting == "split":
        relational_args = replace(relational_args, stable_validation_split=False)
    elif setting == "seed":
        relational_args = replace(relational_args, validation_seed=1)
    elif setting == "fraction":
        relational_args = replace(relational_args, validation_fraction=0.3)
    elif setting == "tokens":
        relational_args = replace(relational_args, entity_max_tokens=1024)
    elif setting == "format":
        flat_args = replace(flat_args, dataset_file=rows.with_suffix(".json"))
        relational_args = replace(
            relational_args, dataset_file=rows.with_suffix(".json")
        )
    jobs = [
        CandidateJob("flat", MorpionStreamingTrainingArgs(flat_args, row_chunk_size=3)),
        CandidateJob(
            "relational",
            MorpionStreamingTrainingArgs(
                relational_args,
                row_chunk_size=2 if setting == "chunk" else 3,
                max_rows=1 if setting == "limit" else None,
            ),
        ),
    ]
    monkeypatch.setenv(shared.ENVIRONMENT, "1")
    with prepare_candidate_inputs(jobs, serial=True) as prepared:
        assert prepared is None
    assert not root(rows).exists()


@pytest.mark.fast
@pytest.mark.parametrize("threads", [1, 2])
def test_shared_inputs_preserve_same_setting_i3_continuation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, threads: int
) -> None:
    """Preserve I3 thread ownership and exact same-setting tensors/Adam/RNG over two rounds."""
    from functools import partial

    from chipiron.environments.morpion.bootstrap import candidate_resources
    from chipiron.environments.morpion.bootstrap.candidate_job import run_candidate
    from chipiron.environments.morpion.players.evaluators.neural_networks.training import (
        train_morpion_regressor_streaming,
    )
    from chipiron.learning.torch_runtime import torch_execution_profile
    from tests.environments.test_morpion_candidate_execution import assert_state_equal
    from tests.environments.test_morpion_combined_execution import _job

    before = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setenv("CHIPIRON_CANDIDATE_WORKERS", "1")
    monkeypatch.setenv("CHIPIRON_CANDIDATE_CPU_BUDGET", "4")
    monkeypatch.setenv("CHIPIRON_CANDIDATE_RESERVED_CPUS", "2")
    monkeypatch.setattr(candidate_resources, "cpu_capacity", lambda: 8.0)
    try:
        with torch_execution_profile("existing"):
            for mode in ("native", "shared"):
                rows = rows_file(tmp_path / mode)
                for number in (1, 2):
                    output = tmp_path / mode / f"round-{number}"
                    parent = None if number == 1 else tmp_path / mode / "round-1"
                    job = _job(rows, output, threads, parent)
                    args = replace(
                        job.training.training_args,
                        stable_validation_split=True,
                        validation_seed=17,
                    )
                    job = replace(
                        job, training=replace(job.training, training_args=args)
                    )
                    if mode == "native":
                        run_candidate(job)
                    else:
                        with prepare(rows) as prepared:
                            assert prepared is not None and torch.get_num_threads() == 1
                            run_candidate(
                                job,
                                trainer=partial(
                                    train_morpion_regressor_streaming,
                                    _prepared=prepared,
                                ),
                            )
                            assert torch.get_num_threads() == 1
                    assert torch.get_num_threads() == 1
            for number in (1, 2):
                left, right = (
                    tmp_path / mode / f"round-{number}" for mode in ("native", "shared")
                )
                for name in ("param.pt", "optimizer.pt"):
                    assert_state_equal(
                        torch.load(left / name, weights_only=True),
                        torch.load(right / name, weights_only=True),
                    )
                a, b = (
                    json.loads((p / "morpion_manifest.json").read_text())["metadata"]
                    for p in (left, right)
                )
                for key in (
                    "state_split",
                    "train_loss",
                    "train_mae",
                    "validation_loss",
                    "validation_mae",
                ):
                    assert a[key] == b[key], key
                # Switching durations are observations, not thread-policy differences.
                timing_keys = {"switch_in_s", "restore_s"}
                assert a["cpu_threading"].keys() == b["cpu_threading"].keys()
                for key in a["cpu_threading"]:
                    if key in timing_keys:
                        assert (
                            a["cpu_threading"][key] >= 0
                            and b["cpu_threading"][key] >= 0
                        )
                    else:
                        assert a["cpu_threading"][key] == b["cpu_threading"][key], key
                for key in (
                    "rng_before_initialization",
                    "rng_before_training",
                    "rng_after_training",
                    "optimizer_steps",
                    "cumulative_optimizer_steps",
                    "parent_optimizer_steps",
                    "starting_weights_sha256",
                    "final_weights_sha256",
                ):
                    assert a["continuation"][key] == b["continuation"][key], key
    finally:
        torch.set_num_threads(before)

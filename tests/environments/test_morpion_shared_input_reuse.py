"""Exact input-only reuse with CURRENT targets, split, population and row order."""

import json
import shutil
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import TYPE_CHECKING, Any

import pytest
import torch

from chipiron.environments.morpion.players.evaluators.neural_networks.training import (
    shared_preparation as shared,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training import (
    shared_preparation_reuse as reuse,
)
from tests.environments.test_morpion_continual_training import rows_file
from tests.environments.test_morpion_shared_preparation import (
    native,
    prepare,
)
from tests.environments.test_morpion_shared_preparation import (
    scientific_rows as scientific_rows,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def isolate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep all derived storage private and separate from attempt inputs."""
    monkeypatch.setattr(shared, "semantic_fingerprint", lambda: "a" * 64)
    monkeypatch.setenv(reuse.ENVIRONMENT, "1")
    monkeypatch.setenv(reuse.DIRECTORY, str(tmp_path / "operational"))
    monkeypatch.delenv("CHIPIRON_STATE_SPLIT_CACHE_DIR", raising=False)


def contents(prepared: Any) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Read the unchanged native payload, including fresh targets and lengths."""
    return (
        torch.load(prepared.flat_paths.tensor_path, weights_only=True),
        torch.load(prepared.relational_paths.tensor_path, weights_only=True),
    )


def equal(actual: Any, expected: Any) -> None:
    """Require every native tensor/dtype exactly, not approximate predictions."""
    for a, b in zip(actual, expected, strict=True):
        assert a.keys() == b.keys()
        for name in b:
            assert a[name].dtype == b[name].dtype
            assert torch.equal(a[name], b[name]), name


def change(path: Path, mode: str) -> None:
    """Change current population without changing the reuse input recipe."""
    records = [json.loads(s) for s in path.read_text().splitlines()]
    header, rows = records[0], records[1:]
    if mode == "all-miss":
        rows = rows[8:]
    else:
        rows.reverse()
        if mode == "partial":
            del rows[5]
            rows.append(json.loads(json.dumps(rows[3])))
        for i, item in enumerate(rows):
            item["row"]["target_value"] += 17.0 + i
            item["row"]["metadata"]["target_source"] = "new-current-target"
    path.write_text("\n".join(json.dumps(x) for x in (header, *rows)) + "\n")


@pytest.mark.parametrize("mode", ["all-hit", "partial", "all-miss"])
@pytest.mark.parametrize("limit", [None, 0, 9])
def test_current_native_outputs_and_independent_split(
    tmp_path: Path, mode: str, limit: int | None
) -> None:
    """Reorder/change labels/duplicates/removals/limits while reusing only inputs."""
    old = rows_file(tmp_path / "old")
    if mode == "all-miss":
        old.write_text("\n".join(old.read_text().splitlines()[:9]) + "\n")
    with prepare(old):
        pass
    current = rows_file(tmp_path / "current", extra=2)
    if mode == "all-hit":
        lines = current.read_text().splitlines()
        current.write_text("\n".join(lines[:17]) + "\n")
    change(current, mode)
    oracle = tmp_path / "oracle/rows.jsonl"
    oracle.parent.mkdir()
    shutil.copyfile(current, oracle)
    expected = native(oracle, limit)
    with prepare(current, limit) as actual:
        assert actual is not None
        equal(contents(actual), expected[:2])
        assert actual.split == expected[2]
    operation = json.loads((tmp_path / "operational/operation.json").read_text())
    assert operation["rows"] == expected[2].row_count
    if limit != 0:
        assert (operation["hits"] == 0) == (mode == "all-miss")
    cache = torch.load(tmp_path / "operational/entry/inputs.pt", weights_only=True)
    assert set(cache) == {
        "flat",
        "tokens",
        "relations",
        "token_lengths",
        "relation_lengths",
    }
    assert not any("target" in key or "split" in key for key in cache)


@pytest.mark.parametrize(
    "damage", ["checksum", "binding", "index", "incomplete", "codec", "schema", "dtype"]
)
def test_corrupt_prior_is_rebuilt_from_current_source(
    tmp_path: Path, damage: str
) -> None:
    """A corrupted derived entry cannot silently supply trusted features."""
    old = rows_file(tmp_path / "old")
    with prepare(old):
        pass
    entry = tmp_path / "operational/entry"
    complete = json.loads((entry / "complete.json").read_text())
    if damage == "incomplete":
        (entry / "complete.json").unlink()
    elif damage in {"codec", "schema", "dtype"}:
        complete["identity"]["inputs"][damage] = "future"
        (entry / "complete.json").write_text(json.dumps(complete))
    elif damage == "checksum":
        with (entry / "inputs.pt").open("r+b") as f:
            f.seek(100)
            f.write(b"broken")
    else:
        if damage == "binding":
            payload = torch.load(entry / "inputs.pt", weights_only=True)
            payload["flat"][0, 0] += 1
            torch.save(payload, entry / "inputs.pt")
            name = "inputs.pt"
        else:
            with sqlite3.connect(entry / "index.sqlite") as c:
                c.execute("UPDATE inputs SET key=? WHERE ordinal=0", ("0" * 64,))
            name = "index.sqlite"
        complete["files"][name] = {
            "bytes": (entry / name).stat().st_size,
            "sha256": shared.file_digest(entry / name),
        }
        (entry / "complete.json").write_text(json.dumps(complete))
    current = rows_file(tmp_path / "current", reverse=True)
    oracle = rows_file(tmp_path / "oracle", reverse=True)
    expected = native(oracle, None)
    with prepare(current) as actual:
        equal(contents(actual), expected[:2])
        assert actual.split == expected[2]
    result = json.loads((tmp_path / "operational/operation.json").read_text())
    assert result["outcome"] == (
        "incompatible" if damage in {"codec", "schema", "dtype"} else "corrupt-rebuild"
    )
    assert result["hits"] == 0


def test_attempt_deletion_and_relocation(tmp_path: Path) -> None:
    """Input storage outlives disposable source attempts without symlink dependencies."""
    old = rows_file(tmp_path / "old")
    with prepare(old):
        pass
    shutil.rmtree(old.parent)
    current = rows_file(tmp_path / "relocated", reverse=True)
    with prepare(current) as actual:
        equal(
            contents(actual),
            native(rows_file(tmp_path / "oracle", reverse=True), None)[:2],
        )
    result = json.loads((tmp_path / "operational/operation.json").read_text())
    assert result["hits"] == 16 and result["reencoded_rows"] == 0


@pytest.mark.parametrize("setting", ["code", "dtype", "budget", "flag", "directory"])
def test_incompatible_and_invalid_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, setting: str
) -> None:
    """Fence code/dtype and reject invalid operational configuration."""
    with prepare(rows_file(tmp_path / "old")):
        pass
    current = rows_file(tmp_path / "current")
    if setting == "code":
        monkeypatch.setattr(shared, "semantic_fingerprint", lambda: "b" * 64)
    elif setting == "dtype":
        torch.set_default_dtype(torch.float64)
    elif setting == "budget":
        monkeypatch.setenv(reuse.ENVIRONMENT + "_MAX_MB", "0")
    elif setting == "flag":
        monkeypatch.setenv(reuse.ENVIRONMENT, "2")
    else:
        monkeypatch.setenv(reuse.DIRECTORY, str(current.parent / "cache"))
    try:
        if setting in {"budget", "flag", "directory"}:
            with pytest.raises(ValueError), prepare(current):
                pytest.fail("invalid input reuse configuration accepted")
        else:
            with prepare(current) as actual:
                equal(
                    contents(actual), native(rows_file(tmp_path / "oracle"), None)[:2]
                )
            result = json.loads((tmp_path / "operational/operation.json").read_text())
            assert result["hits"] == 0
            assert result["outcome"] == (
                "incompatible" if setting == "code" else "absent"
            )
    finally:
        torch.set_default_dtype(torch.float32)


def test_current_split_settings_are_independent(tmp_path: Path) -> None:
    """A changed split recipe and finite population still reuse exact inputs only."""
    from chipiron.environments.morpion.players.evaluators.neural_networks.training import (
        state_split,
    )

    with prepare(rows_file(tmp_path / "old")):
        pass
    current = rows_file(tmp_path / "current", reverse=True)
    expected = state_split._split.__wrapped__(
        str(current), shared.file_digest(current), 0.4, 29, 11
    )
    with shared.prepare_shared_inputs(
        current, fraction=0.4, seed=29, max_rows=11, row_chunk_size=3
    ) as actual:
        assert actual is not None and actual.split == expected
    result = json.loads((tmp_path / "operational/operation.json").read_text())
    assert result["hits"] == 11 and result["rows"] == 11


@pytest.mark.parametrize("stage", ["before_publish", "replacement_gap"])
def test_killed_builder_reaps_before_restart(tmp_path: Path, stage: str) -> None:
    """A dead writer exposes no partial entry; restart cleans only private debris."""
    with prepare(rows_file(tmp_path / "old")):
        pass
    current = rows_file(tmp_path / "current", extra=1)
    script = """
import os,sys
from pathlib import Path
from chipiron.environments.morpion.players.evaluators.neural_networks.training import shared_preparation as s, shared_preparation_reuse as r
s.semantic_fingerprint=lambda:'a'*64
def die(*args,**kwargs):os._exit(73)
if sys.argv[2]=='before_publish':r._Reuse.publish=die
else:r.publish_directory_if_absent=die
with s.prepare_shared_inputs(Path(sys.argv[1]),fraction=.2,seed=17,max_rows=None,row_chunk_size=3):pass
"""
    child = subprocess.run(
        [sys.executable, "-c", script, str(current), stage], check=False, timeout=45
    )
    assert child.returncode == 73
    with prepare(current) as actual:
        equal(
            contents(actual), native(rows_file(tmp_path / "oracle", extra=1), None)[:2]
        )
    result = json.loads((tmp_path / "operational/operation.json").read_text())
    assert result["hits"] == (16 if stage == "before_publish" else 0)
    assert not list((tmp_path / "operational").glob("attempt-*"))


def test_lock_lifetime_and_pruning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A prior consumer excludes pruning; native consumers own independent files."""
    import fcntl

    with prepare(rows_file(tmp_path / "old")):
        pass
    entered, release = Event(), Event()
    original = reuse._Reuse.lookup
    paused = False

    def lookup(self: Any, payload: object) -> Any:
        nonlocal paused
        if not paused:
            paused = True
            entered.set()
            assert release.wait(15)
        return original(self, payload)

    monkeypatch.setattr(reuse._Reuse, "lookup", lookup)
    a, b = rows_file(tmp_path / "a"), rows_file(tmp_path / "b", reverse=True)

    def build(path: Path) -> None:
        with prepare(path) as actual:
            assert actual is not None and actual.split.row_count == 16

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(build, a)
        try:
            assert entered.wait(15)
            with (
                (tmp_path / "operational/.lock").open("r") as lock,
                pytest.raises(BlockingIOError),
            ):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            second = pool.submit(build, b)
        finally:
            release.set()
        first.result(timeout=30)
        second.result(timeout=30)
    with prepare(a) as active:
        before = contents(active)
        with prepare(rows_file(tmp_path / "c", extra=1)):
            pass
        equal(contents(active), before)


def test_coexistence_budget_preserves_prior(tmp_path: Path) -> None:
    """An insufficient current+prior bound stops without pruning a live prior."""
    with prepare(rows_file(tmp_path / "old")):
        pass
    entry = tmp_path / "operational/entry"
    before = {p.name: shared.file_digest(p) for p in entry.iterdir()}
    current = rows_file(tmp_path / "current")
    with (
        pytest.raises(ValueError, match="budget"),
        prepare(current, budget=shared._Budget(scratch_bytes=1)),
    ):
        pytest.fail("insufficient aggregate capacity accepted")
    assert before == {p.name: shared.file_digest(p) for p in entry.iterdir()}
    with prepare(current) as actual:
        assert actual is not None and actual.split.row_count == 16


def test_current_confidence_and_exactness(
    tmp_path: Path, scientific_rows: dict[bool, Any]
) -> None:
    """Valid new backup/exactness certificates and targets cannot be retained from cache."""
    from chipiron.environments.morpion.learning import (
        save_morpion_supervised_rows_streaming,
    )

    for label, backup in (("old", False), ("current", True), ("oracle", True)):
        data = scientific_rows[backup]
        save_morpion_supervised_rows_streaming(
            rows=iter(data.rows),
            metadata=data.metadata,
            path=tmp_path / label / "rows.jsonl",
        )
    with prepare(tmp_path / "old/rows.jsonl"):
        pass
    expected = native(tmp_path / "oracle/rows.jsonl", None)
    with prepare(tmp_path / "current/rows.jsonl") as actual:
        equal(contents(actual), expected[:2])
        assert actual.split == expected[2]
    result = json.loads((tmp_path / "operational/operation.json").read_text())
    assert result["hits"] == len(scientific_rows[True].rows)
    invalid = tmp_path / "invalid/rows.jsonl"
    invalid.parent.mkdir()
    records = [
        json.loads(s)
        for s in (tmp_path / "current/rows.jsonl").read_text().splitlines()
    ]
    del records[1]["row"]["metadata"]["target_confidence"]["raw_target_certainty"]
    invalid.write_text("\n".join(json.dumps(x) for x in records) + "\n")
    with pytest.raises(ValueError), prepare(invalid):
        pytest.fail("cached input bypassed current certificate validation")


def test_orientation_is_not_split_equivalence(tmp_path: Path) -> None:
    """D4-equivalent split keys cannot authorize oriented input hits."""
    from dataclasses import replace

    from atomheart.games.morpion.checkpoints import MorpionStateCheckpointCodec

    from chipiron.environments.morpion.learning import (
        decode_morpion_state_ref_payload,
        load_morpion_supervised_rows,
        save_morpion_supervised_rows_streaming,
    )
    from chipiron.environments.morpion.players.evaluators.neural_networks.training.state_split import (
        canonical_identity,
    )
    from chipiron.environments.morpion.symmetry import transform_morpion_state
    from chipiron.environments.morpion.types import MorpionDynamics

    old = rows_file(tmp_path / "old")
    row = load_morpion_supervised_rows(old).rows[-1]
    with prepare(old):
        pass
    state = MorpionDynamics().wrap_atomheart_state(
        decode_morpion_state_ref_payload(row.state_ref_payload)
    )
    payload = MorpionStateCheckpointCodec().dump_state_ref(
        transform_morpion_state(state, 1).to_atomheart_state()
    )
    transformed = replace(row, state_ref_payload=payload)
    assert payload != row.state_ref_payload and canonical_identity(
        row
    ) == canonical_identity(transformed)
    for label in ("current", "oracle"):
        save_morpion_supervised_rows_streaming(
            rows=iter([transformed]), metadata={}, path=tmp_path / label / "rows.jsonl"
        )
    expected = native(tmp_path / "oracle/rows.jsonl", None)
    with prepare(tmp_path / "current/rows.jsonl") as actual:
        equal(contents(actual), expected[:2])
        assert actual.split == expected[2]
    result = json.loads((tmp_path / "operational/operation.json").read_text())
    assert result["hits"] == 0 and result["reencoded_rows"] == 1


def test_prior_mutation_during_gather_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even a replaced checksum manifest cannot substitute an already verified prior."""
    with prepare(rows_file(tmp_path / "old")):
        pass
    original = reuse._Reuse.lookup
    mutated = False

    def lookup(self: Any, payload: object) -> Any:
        nonlocal mutated
        result = original(self, payload)
        if not mutated:
            mutated = True
            entry = tmp_path / "operational/entry"
            complete = json.loads((entry / "complete.json").read_text())
            complete["identity"]["semantics"] = "b" * 64
            (entry / "complete.json").write_text(json.dumps(complete))
        return result

    monkeypatch.setattr(reuse._Reuse, "lookup", lookup)
    current = rows_file(tmp_path / "current")
    with pytest.raises(ValueError, match="changed during gathering"), prepare(current):
        pytest.fail("mutated prior was published")
    assert not (current.parent / ".rows.jsonl.shared-preparation-v1/entry").exists()


def test_replay_order_is_not_a_state_key(tmp_path: Path) -> None:
    """Different legal replay payloads for the same state do not borrow inputs."""
    from dataclasses import replace

    from chipiron.environments.morpion.learning import (
        decode_morpion_state_ref_payload,
        load_morpion_supervised_rows,
        save_morpion_supervised_rows_streaming,
    )
    from chipiron.environments.morpion.players.evaluators.neural_networks.training.state_split import (
        canonical_identity,
    )

    old = rows_file(tmp_path / "old")
    row = load_morpion_supervised_rows(old).rows[-1]
    with prepare(old):
        pass
    payload = json.loads(json.dumps(row.state_ref_payload))
    for i in range(len(payload[1]) - 1):
        candidate = json.loads(json.dumps(payload))
        candidate[1][i], candidate[1][i + 1] = candidate[1][i + 1], candidate[1][i]
        try:
            decode_morpion_state_ref_payload(candidate)
        except TypeError, ValueError:
            continue
        transformed = replace(row, state_ref_payload=candidate)
        if canonical_identity(transformed) == canonical_identity(row):
            break
    else:
        pytest.fail("fixture has no independent legal replay moves")
    assert candidate != payload
    for label in ("current", "oracle"):
        save_morpion_supervised_rows_streaming(
            rows=iter([transformed]), metadata={}, path=tmp_path / label / "rows.jsonl"
        )
    with prepare(tmp_path / "current/rows.jsonl") as actual:
        equal(contents(actual), native(tmp_path / "oracle/rows.jsonl", None)[:2])
    result = json.loads((tmp_path / "operational/operation.json").read_text())
    assert result["hits"] == 0 and result["reencoded_rows"] == 1


@pytest.mark.parametrize("max_tokens", [512, 1024])
def test_unsupported_cohort_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, max_tokens: int
) -> None:
    """Unqualified representation settings never consume a fixed-recipe prior."""
    from dataclasses import asdict, replace

    from chipiron.environments.morpion.bootstrap.candidate_job import CandidateJob
    from chipiron.environments.morpion.bootstrap.candidate_preparation import (
        prepare_candidate_inputs,
    )
    from chipiron.environments.morpion.players.evaluators.neural_networks.training import (
        MorpionStreamingTrainingArgs,
        MorpionTrainingArgs,
    )

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
                )
            ),
        )
        for kind in ("linear", "relation_biased_entity_token_transformer_value_net")
    ]
    args = replace(jobs[-1].training.training_args, entity_max_tokens=max_tokens)
    jobs[-1] = replace(
        jobs[-1], training=replace(jobs[-1].training, training_args=args)
    )
    before = [asdict(j) for j in jobs]
    monkeypatch.setenv(shared.ENVIRONMENT, "1")
    with prepare_candidate_inputs(jobs, serial=True) as actual:
        assert actual is None
    assert [asdict(j) for j in jobs] == before
    assert not (tmp_path / "operational").exists()

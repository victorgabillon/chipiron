"""Finite accepted traversal survives actual model changes without claiming freshness."""

import copy
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from chipiron.environments.morpion.bootstrap import reevaluation_transaction as tx
from chipiron.environments.morpion.bootstrap import reevaluation_worker as worker
from chipiron.environments.morpion.bootstrap import run_pipeline_training_stage
from chipiron.environments.morpion.bootstrap.checkpoint_patch_receipts import (
    receipt_ledger,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    PUBLICATION_KEY,
    RETIRED_PAYLOAD_KEY,
    publication_records,
)
from chipiron.environments.morpion.bootstrap.model_activation import (
    activate_model,
    prepare_model_activation_request,
)
from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
    load_reevaluation_patch,
)
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from chipiron.environments.morpion.bootstrap.sharded_training_reader import (
    ShardedTrainingGenerationView,
)
from tests.environments.morpion_activation_helpers import setup_activation
from tests.environments.morpion_fenced_dataset_helpers import (
    local_leaderboard,
    run_dataset,
)
from tests.environments.test_morpion_payload_retirement import grow
from tests.environments.test_morpion_rolling_recovery_report import analyze


def setup(root: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Opt into schema3 only at the existing fresh-genesis factory, not an old run."""
    local_leaderboard(monkeypatch, root)
    return setup_activation(root, count=8, rolling_reevaluation=True)


def propose(args: Any, limit: int = 1) -> Any:
    """Use the real worker, real neural model and indexed immutable source."""
    return worker.run_morpion_reevaluation_worker_once(
        args,
        max_nodes_per_patch=limit,
        use_streaming_sharded_reader=True,
        allow_scientific_v2=True,
    )


def inventory(paths: Any, generation: int) -> tuple[str, ...]:
    """Read the small fixture's exact immutable node set."""
    state = load_bootstrap_run_state(paths.run_state_path)
    record = next(
        r for r in publication_records(state.metadata) if r["generation"] == generation
    )
    with ShardedTrainingGenerationView(
        paths.work_dir / record["tree_snapshot_path"],
        allow_scientific_v2=True,
        allow_closed_snapshot=True,
    ) as view:
        return view.select_finite_node_window(None, view.node_count())[0]


def accepted(paths: Any) -> dict[str, Any]:
    """Reload authoritative state instead of the producer projection."""
    return dict(
        load_bootstrap_run_state(paths.run_state_path).metadata[tx.TRANSACTION_KEY][
            "accepted_cursor"
        ]
    )


def test_abc_progress_and_new_nodes_wait_for_next_finite_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real A→B→C activations do not reset the pass; Growth may keep adding nodes."""
    args, paths, _ = setup(tmp_path / "work", monkeypatch)
    original = inventory(paths, 1)
    assert len(original) >= 4
    seen: list[str] = []
    models: list[str] = []
    for index in range(3):
        assert propose(args).patch_written
        patch = load_reevaluation_patch(paths.pipeline_reevaluation_patch_path)
        assert patch.tree_generation == 1
        seen.extend(row.node_id for row in patch.rows)
        models.append(
            cast("dict[str, Any]", patch.metadata[tx.TRANSACTION_KEY])["model"][
                "sha256"
            ]
        )
        state = grow(paths)
        cursor = accepted(paths)
        assert cursor["next_node_cursor"] == original[index + 1]
        assert cursor["completed_full_pass_count"] == 0
        if index < 2:
            training_generation = 1 if index == 0 else state.generation
            if index:
                run_dataset(args, generation=training_generation)
                run_pipeline_training_stage(args, generation=training_generation)
            request = prepare_model_activation_request(
                paths, training_generation=training_generation
            )
            activate_model(paths, request)
            assert accepted(paths) == cursor
    assert len(set(models)) == 3
    assert seen == list(original[:3])
    assert propose(args, len(original) + 100).patch_written
    patch = load_reevaluation_patch(paths.pipeline_reevaluation_patch_path)
    assert tuple(row.node_id for row in patch.rows) == original[3:]
    assert patch.metadata["completed_full_pass"] is True
    grow(paths)
    assert accepted(paths)["next_node_cursor"] is None
    assert accepted(paths)["completed_full_pass_count"] == 1
    latest = load_bootstrap_run_state(paths.run_state_path).generation
    newer = inventory(paths, latest)
    assert set(newer) > set(original)
    assert propose(args, len(newer) + 100).patch_written
    patch = load_reevaluation_patch(paths.pipeline_reevaluation_patch_path)
    assert patch.tree_generation == latest
    assert tuple(row.node_id for row in patch.rows) == newer
    grow(paths)
    assert accepted(paths)["completed_full_pass_count"] == 2


def test_obsolete_inference_discards_only_batch_and_stale_packet_refuses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Activation racing real inference leaves accepted progress and no stale write."""
    args, paths, _ = setup(tmp_path / "work", monkeypatch)
    assert propose(args).patch_written
    grow(paths)
    before = accepted(paths)
    assert propose(args).patch_written
    obsolete = paths.pipeline_reevaluation_patch_path.read_bytes()
    # Delay this real A packet beyond activation; it has the CURRENT cursor hash.
    paths.pipeline_reevaluation_patch_path.unlink()
    request = prepare_model_activation_request(paths, training_generation=1)
    evaluate = worker.MorpionActiveModelNodeReevaluationEvaluator.evaluate_node_window

    def racing(self: Any, nodes: Any) -> Any:
        rows = evaluate(self, nodes)
        activate_model(paths, request)
        return rows

    with monkeypatch.context() as race:
        race.setattr(
            worker.MorpionActiveModelNodeReevaluationEvaluator,
            "evaluate_node_window",
            racing,
        )
        result = propose(args)
    assert result.reason == "model_activated" and not result.patch_written
    assert not paths.pipeline_reevaluation_patch_path.exists()
    assert accepted(paths) == before
    paths.pipeline_reevaluation_patch_path.write_bytes(obsolete)
    state_bytes = paths.run_state_path.read_bytes()
    with pytest.raises(ValueError, match=r"[Mm]odel"):
        grow(paths)
    assert paths.run_state_path.read_bytes() == state_bytes
    assert accepted(paths) == before
    paths.pipeline_reevaluation_patch_path.unlink()
    assert propose(args).patch_written
    patch = load_reevaluation_patch(paths.pipeline_reevaluation_patch_path)
    assert patch.start_cursor == before["next_node_cursor"]
    assert (
        cast("dict[str, Any]", patch.metadata[tx.TRANSACTION_KEY])["model"]
        == request["candidate"]
    )
    grow(paths)


@pytest.mark.parametrize(
    "damage", ["skip", "repeat", "source", "end", "empty", "counter"]
)
def test_prefix_forgery_refuses_before_native_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    """A plausible cursor digest cannot authorize skipped rows or endless no-ops."""
    args, paths, _ = setup(tmp_path / "work", monkeypatch)
    assert propose(args, 2).patch_written
    patch = load_reevaluation_patch(paths.pipeline_reevaluation_patch_path)
    state = load_bootstrap_run_state(paths.run_state_path)
    metadata = cast("dict[str, Any]", copy.deepcopy(patch.metadata))
    cursor = metadata[tx.TRANSACTION_KEY]["next_cursor"]
    if damage == "skip":
        patch = replace(patch, rows=patch.rows[1:])
    elif damage == "repeat":
        patch = replace(patch, rows=(patch.rows[0], patch.rows[0]))
    elif damage == "source":
        cursor["metadata"]["rolling_source"]["id"] = "f" * 32
    elif damage == "end":
        cursor["next_node_cursor"] = metadata["next_node_cursor"] = None
        metadata["completed_full_pass"] = True
    elif damage == "empty":
        patch = replace(patch, rows=())
    else:
        cursor["completed_full_pass_count"] += 1
    patch = replace(patch, metadata=metadata)
    with pytest.raises(ValueError):
        tx.validate_proposal(state, patch, work_dir=paths.work_dir)


def test_retirement_cannot_drop_incomplete_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The accepted cursor roots its source even when no producer is running."""
    args, paths, _ = setup(tmp_path / "work", monkeypatch)
    assert propose(args).patch_written
    # The analyzer owns pins too. Bound a real subprocess so nested shared/exclusive
    # source-pin regressions fail instead of hanging the suite.
    script = "from pathlib import Path; from chipiron.environments.morpion.bootstrap import MorpionBootstrapPaths; from tests.environments.test_morpion_rolling_recovery_report import analyze; r=analyze(MorpionBootstrapPaths.from_work_dir(Path(__import__('sys').argv[1]))); assert r['validated'], r; assert 'pending-patch' in r['generations'][0]['rolling_reasons']"
    subprocess.run(
        [sys.executable, "-c", script, str(paths.work_dir)],
        check=True,
        timeout=30,
    )
    for _ in range(4):
        grow(paths)
    state = load_bootstrap_run_state(paths.run_state_path)
    assert tx.rolling_source_generation(state) == 1
    report = analyze(paths)
    assert report["validated"], report
    assert (
        "authoritative-accepted-cursor" in report["generations"][0]["rolling_reasons"]
    )
    metadata = copy.deepcopy(state.metadata)
    record = metadata[PUBLICATION_KEY]["generations"][0]
    record[RETIRED_PAYLOAD_KEY] = record["descriptor_sha256"]
    with pytest.raises(ValueError, match="incomplete rolling"):
        publication_records(metadata)
    assert propose(args, 1000).patch_written
    grow(paths)
    state = load_bootstrap_run_state(paths.run_state_path)
    assert tx.rolling_source_generation(state) is None
    report = analyze(paths)
    assert report["validated"], report
    assert (
        "authoritative-accepted-cursor"
        not in report["generations"][0]["rolling_reasons"]
    )
    metadata = copy.deepcopy(state.metadata)
    record = metadata[PUBLICATION_KEY]["generations"][0]
    record[RETIRED_PAYLOAD_KEY] = record["descriptor_sha256"]
    # Completed traversal does not override this source's accepted learning hold.
    with pytest.raises(ValueError, match="Invalid retirement admission"):
        publication_records(metadata)


def test_finite_reader_edges_and_fresh_genesis_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject absent continuations, zero capacity, old passes and old-run adoption."""
    args, paths, state = setup(tmp_path / "work", monkeypatch)
    record = publication_records(state.metadata)[0]
    with ShardedTrainingGenerationView(
        paths.work_dir / record["tree_snapshot_path"],
        allow_scientific_v2=True,
        allow_closed_snapshot=True,
    ) as view:
        ids = inventory(paths, 1)
        assert view.select_finite_node_window(None, len(ids)) == (ids, None, True)
        assert view.select_finite_node_window(ids[-1], 100) == ((ids[-1],), None, True)
        with pytest.raises(ValueError, match="absent"):
            view.select_finite_node_window("not-in-this-snapshot", 1)
        with pytest.raises(ValueError, match="positive"):
            view.select_finite_node_window(None, 0)
    assert propose(args, 1000).patch_written
    patch = load_reevaluation_patch(paths.pipeline_reevaluation_patch_path)
    current = grow(paths)
    metadata = cast("dict[str, Any]", copy.deepcopy(patch.metadata))
    envelope = metadata[tx.TRANSACTION_KEY]
    envelope["expected_cursor_sha256"] = tx.cursor_digest(accepted(paths))
    envelope["next_cursor"]["last_patch_id"] = "repeat-pass"
    envelope["next_cursor"]["completed_full_pass_count"] = 2
    repeated = replace(patch, patch_id="repeat-pass", metadata=metadata)
    with pytest.raises(ValueError, match="later committed inventory"):
        tx.validate_proposal(current, repeated, work_dir=paths.work_dir)
    with pytest.raises(ValueError, match="Growth scoring"):
        tx.initial_transaction(rolling=True)
    from chipiron.environments.morpion.bootstrap.scientific_publication_initialization import (
        initialize_scientific_v2_publication,
    )

    with pytest.raises(ValueError, match="new isolated workspace"):
        initialize_scientific_v2_publication(
            paths,
            state,
            producer=record["scientific_identity"]["producer"]["identity"],
            closed_scientific_snapshot=True,
            atomic_reevaluation=True,
            growth_scoring=True,
            rolling_reevaluation=True,
        )


def crash_grow(root: Path, boundary: str) -> None:
    """Kill an actual native consumer at the existing durable state replace."""
    from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
        MorpionBootstrapPaths,
    )

    paths = MorpionBootstrapPaths.from_work_dir(root)
    original = os.replace

    def replace_state(src: Any, dst: Any, **kwargs: Any) -> None:
        if Path(dst) == paths.run_state_path and boundary == "before":
            os._exit(91)
        original(src, dst, **kwargs)
        if Path(dst) == paths.run_state_path:
            os._exit(91)

    os.replace = replace_state
    grow(paths)
    message = "Crash boundary was not reached"
    raise AssertionError(message)


@pytest.mark.parametrize("boundary", ["before", "after"])
def test_crash_recovers_receipt_cursor_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    """Process death around run_state cannot separate values, receipt and cursor."""
    args, paths, _ = setup(tmp_path / "work", monkeypatch)
    assert propose(args).patch_written
    packet = paths.pipeline_reevaluation_patch_path.read_bytes()
    patch = load_reevaluation_patch(paths.pipeline_reevaluation_patch_path)
    script = "from pathlib import Path; from tests.environments.test_morpion_rolling_reevaluation import crash_grow; crash_grow(Path(__import__('sys').argv[1]), __import__('sys').argv[2])"
    child = subprocess.run(
        [sys.executable, "-c", script, str(paths.work_dir), boundary],
        check=False,
        timeout=60,
    )
    assert child.returncode == 91
    state = load_bootstrap_run_state(paths.run_state_path)
    assert bool(receipt_ledger(state)) == (boundary == "after")
    grow(paths)
    assert (
        accepted(paths)
        == cast("dict[str, Any]", patch.metadata[tx.TRANSACTION_KEY])["next_cursor"]
    )
    assert len(receipt_ledger(load_bootstrap_run_state(paths.run_state_path))) == 1
    assert propose(args).patch_written
    grow(paths)
    newer = accepted(paths)
    paths.pipeline_reevaluation_patch_path.write_bytes(packet)
    grow(paths)
    assert accepted(paths) == newer
    assert len(receipt_ledger(load_bootstrap_run_state(paths.run_state_path))) == 2

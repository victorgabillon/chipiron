"""Explicit fresh proposal policy and exact Growth/Reevaluation score parity."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest
from anemone.node_evaluation.direct import scale_nonterminal_score
from valanga.evaluations import Certainty

from chipiron.environments.morpion.bootstrap import reevaluation_worker as worker
from chipiron.environments.morpion.bootstrap.checkpoint_patch_receipts import (
    receipt_ledger,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    GenerationPublicationError,
)
from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
    load_reevaluation_patch,
)
from chipiron.environments.morpion.bootstrap.reevaluation_publication import (
    publish_reevaluation_patch_if_absent,
)
from chipiron.environments.morpion.bootstrap.reevaluation_transaction import (
    GROWTH_NUMERIC_POLICY,
    TRANSACTION_KEY,
)
from chipiron.environments.morpion.bootstrap.scientific_publication_initialization import (
    initialize_scientific_v2_publication,
)
from tests.environments import test_morpion_durable_closed_crashes as crash_boundaries
from tests.environments.morpion_atomic_reevaluation_helpers import (
    atomic_worker,
    make_proposal,
)
from tests.environments.morpion_durable_closed_helpers import (
    ClosedPublicationRunner,
    run_closed_publication,
    setup_closed_baseline,
)
from tests.environments.morpion_durable_v2_helpers import scientific_producer
from tests.environments.test_morpion_committed_evaluator_restore import nodes
from tests.environments.test_morpion_value_certainty import (
    endgame,
    make_runner,
    neural_evaluator,
)

if TYPE_CHECKING:
    from pathlib import Path


def setup_scaled(root: Path) -> Any:
    """No migration: select the new identity at fresh closed genesis."""
    args, paths, state = setup_closed_baseline(root, initialize=False)
    state = initialize_scientific_v2_publication(
        paths,
        state,
        producer=scientific_producer(args),
        closed_scientific_snapshot=True,
        atomic_reevaluation=True,
        growth_scoring=True,
    )
    assert state.metadata[TRANSACTION_KEY]["schema_version"] == 2
    return args, paths, state


@pytest.mark.fast
@pytest.mark.parametrize("score", [42.0, -42.0, 0.0, -0.0, 1.25])
def test_same_neural_states_depths_and_dag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, score: float
) -> None:
    """Compare actual Growth direct Values bitwise, including ties and multi-parent nodes."""
    current = make_runner(monkeypatch)
    # Increase only the existing live budget before growing a legal DAG.
    from chipiron.environments.morpion.bootstrap import (
        MorpionBootstrapEffectiveRuntimeConfig,
    )
    from chipiron.environments.morpion.bootstrap.runtime.runner import (
        AnemoneMorpionSearchRunner,
    )

    current = AnemoneMorpionSearchRunner(current._args)
    current.load_or_create(
        None, None, MorpionBootstrapEffectiveRuntimeConfig(tree_branch_limit=100000)
    )
    while current.current_tree_size() < 76:
        current.grow(1)
    evaluator = neural_evaluator(score)
    current._runtime.refresh_with_evaluator(evaluator, scope="all", batch_size=7)
    snapshot, _ = current.build_training_tree_snapshot_payload()
    adapter = worker.MorpionActiveModelNodeReevaluationEvaluator(
        tmp_path / "model", GROWTH_NUMERIC_POLICY
    )
    adapter._evaluator = evaluator
    rows = adapter.evaluate_node_window(snapshot.nodes)
    by_id = {str(n.id): n for n in nodes(current)}
    assert any(len(n.parent_nodes) > 1 for n in by_id.values())
    assert any(n.depth > 0 for n in snapshot.nodes)
    for row, source in zip(rows, snapshot.nodes, strict=True):
        direct = by_id[row.node_id].tree_evaluation.direct_value
        assert row.direct_value.hex() == float(direct.score).hex()
        assert row.metadata["tree_depth"] == source.depth
        assert direct.certainty == Certainty.ESTIMATE and not row.is_exact
    raw = worker.MorpionActiveModelNodeReevaluationEvaluator(tmp_path / "model")
    raw._evaluator = evaluator
    raw_rows = raw.evaluate_node_window(snapshot.nodes)
    if score:
        assert any(
            a.direct_value.hex() != b.direct_value.hex()
            for a, b in zip(rows, raw_rows, strict=True)
        )
    fallback = worker.node_values_to_patch_rows(snapshot.nodes)
    assert [r.direct_value for r in fallback] == [
        n.direct_value_scalar for n in snapshot.nodes
    ]


@pytest.mark.fast
def test_real_move34_root_and_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Checkpoint depth zero keeps ESTIMATE42 separate from FORCED35 and TERMINAL35."""
    late, _ = endgame.__wrapped__()
    current = make_runner(monkeypatch, state=late)
    evaluator = neural_evaluator(42)
    current._runtime.refresh_with_evaluator(evaluator, scope="all", batch_size=1)
    snapshot, _ = current.build_training_tree_snapshot_payload()
    adapter = worker.MorpionActiveModelNodeReevaluationEvaluator(
        tmp_path / "model", GROWTH_NUMERIC_POLICY
    )
    adapter._evaluator = evaluator
    rows = adapter.evaluate_node_window(snapshot.nodes)
    root = next(n for n in nodes(current) if n.tree_depth == 0)
    assert root.tree_evaluation.direct_value.certainty == Certainty.ESTIMATE
    assert root.tree_evaluation.tree_value.certainty == Certainty.FORCED
    assert root.tree_evaluation.effective_value.score == 35
    assert snapshot.nodes[0].depth == 0
    assert rows[0].direct_value == 42 and not rows[0].is_exact
    assert rows[1].direct_value == 35 and rows[1].is_terminal and rows[1].is_exact


@pytest.mark.fast
@pytest.mark.parametrize("streaming", [False, True])
def test_scaled_worker_exact_receipt_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, streaming: bool
) -> None:
    """The real producer binds policy/depth; exact retries cannot replay or rewind."""
    args, paths, _ = setup_scaled(tmp_path)
    result = atomic_worker(tmp_path, monkeypatch, streaming=streaming)
    assert result.patch_written
    patch = load_reevaluation_patch(paths.pipeline_reevaluation_patch_path)
    assert patch.metadata[TRANSACTION_KEY]["numeric_policy"] == GROWTH_NUMERIC_POLICY
    row = patch.rows[0]
    # StateScore returns played_moves + 0.125; leaf has checkpoint depth 2.
    assert row.direct_value == scale_nonterminal_score(2.125, 2)
    original = paths.pipeline_reevaluation_patch_path.read_bytes()
    first = ClosedPublicationRunner()
    state = run_closed_publication(args, first)
    assert first.apply_calls == 1 and patch.patch_id in receipt_ledger(state)
    before = paths.run_state_path.read_bytes()
    paths.pipeline_reevaluation_patch_path.write_bytes(original)
    retry = ClosedPublicationRunner()
    recovered = run_closed_publication(args, retry, max_cycles=0)
    assert recovered == state and retry.apply_calls == 0
    assert paths.run_state_path.read_bytes() == before
    assert not paths.pipeline_reevaluation_patch_path.exists()


@pytest.mark.fast
@pytest.mark.parametrize(
    "fault",
    ["old", "policy", "row", "source", "model", "unknown_exact", "unknown_terminal"],
)
def test_incompatible_proposals_rejected(tmp_path: Path, fault: str) -> None:
    """Raw-score identity and source/model substitutions fail before any application."""
    args, paths, _ = setup_scaled(tmp_path)
    patch = make_proposal(paths)
    metadata = copy.deepcopy(patch.metadata)
    envelope = metadata[TRANSACTION_KEY]
    rows = tuple(
        replace(
            r,
            is_exact=False,
            is_terminal=False,
            metadata=dict(
                source="active_model_reevaluation",
                numeric_policy=GROWTH_NUMERIC_POLICY,
                tree_depth=0,
            ),
        )
        for r in patch.rows
    )
    if fault == "old":
        envelope["schema_version"] = 1
        del envelope["numeric_policy"]
    elif fault == "policy":
        envelope["numeric_policy"] = "raw_nonterminal_v1"
    elif fault == "row":
        rows = patch.rows
    elif fault in ("unknown_exact", "unknown_terminal"):
        field = "is_exact" if fault == "unknown_exact" else "is_terminal"
        rows = tuple(replace(r, **{field: None}) for r in rows)
    elif fault == "source":
        envelope["source"]["descriptor_sha256"] = "0" * 64
    else:
        envelope["model"]["sha256"] = "0" * 64
    patch = replace(patch, rows=rows, metadata=metadata)
    assert publish_reevaluation_patch_if_absent(
        patch, paths.pipeline_reevaluation_patch_path
    )
    before = paths.run_state_path.read_bytes()
    current = ClosedPublicationRunner()
    with pytest.raises(ValueError):
        run_closed_publication(args, current)
    assert current.apply_calls == 0 and paths.run_state_path.read_bytes() == before
    assert paths.pipeline_reevaluation_patch_path.exists()


@pytest.mark.fast
def test_scoring_requires_explicit_atomic_genesis(tmp_path: Path) -> None:
    """Neither a legacy workspace nor an unversioned producer can silently switch."""
    args, paths, state = setup_closed_baseline(tmp_path, initialize=False)
    with pytest.raises(GenerationPublicationError, match="explicit fresh atomic"):
        initialize_scientific_v2_publication(
            paths, state, producer=scientific_producer(args), growth_scoring=True
        )


@pytest.mark.fast
def test_scaled_full_and_bounded_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reuse the entire proposal-byte oracle with the explicit new genesis only."""
    from tests.environments import test_morpion_atomic_reevaluation as original

    monkeypatch.setattr(original, "setup_atomic", setup_scaled)
    original.test_full_and_bounded_entire_proposal_bytes_match(tmp_path, monkeypatch)


@pytest.mark.parametrize(
    "boundary",
    [
        pytest.param(
            value,
            marks=pytest.mark.fast
            if value in ["after_state", "before_state"]
            else pytest.mark.exhaustive,
        )
        for value in crash_boundaries.PRE_COMMIT + crash_boundaries.POST_COMMIT
    ],
)
def test_scaled_consumer_process_death(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    """Reuse the actual exit/fsync/commit/repair boundaries with a newly scaled proposal."""
    from tests.environments import test_morpion_atomic_reevaluation_crashes as original

    monkeypatch.setattr(original, "setup_atomic", setup_scaled)

    def produce(paths: Any) -> Any:
        atomic_worker(paths.work_dir, monkeypatch)
        return load_reevaluation_patch(paths.pipeline_reevaluation_patch_path)

    monkeypatch.setattr(original, "publish_proposal", produce)
    original.test_atomic_consumer_death(tmp_path, boundary)


@pytest.mark.parametrize("boundary", ["file_fsync", "before_link", "after_link"])
def test_scaled_producer_process_death(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    """Existing producer crash/restart assertions also preserve numeric identity."""
    from tests.environments import test_morpion_atomic_reevaluation_crashes as original

    monkeypatch.setattr(original, "setup_atomic", setup_scaled)
    original.test_producer_death_recovery(tmp_path, monkeypatch, boundary)

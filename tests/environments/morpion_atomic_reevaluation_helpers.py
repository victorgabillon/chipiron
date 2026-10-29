"""Fresh explicit protocol fixtures, never historical adoption."""

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from chipiron.environments.morpion.bootstrap import load_bootstrap_run_state
from chipiron.environments.morpion.bootstrap.checkpoint_patch_receipts import (
    current_model_identity,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    committed_scientific_source,
)
from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
    MorpionReevaluationCursor,
)
from chipiron.environments.morpion.bootstrap.reevaluation_publication import (
    publish_reevaluation_patch_if_absent,
)
from chipiron.environments.morpion.bootstrap.reevaluation_transaction import (
    TRANSACTION_KEY,
    proposal_patch,
    source_identity,
    transaction_state,
)
from chipiron.environments.morpion.bootstrap.scientific_publication_initialization import (
    initialize_scientific_v2_publication,
)
from tests.environments.morpion_durable_closed_helpers import setup_closed_baseline
from tests.environments.morpion_durable_v2_helpers import scientific_producer
from tests.environments.morpion_patch_receipt_helpers import fixture_patch

if TYPE_CHECKING:
    from pathlib import Path


def setup_atomic(root: Path) -> Any:
    """Initialize empty producer state in the existing durable genesis commit."""
    args, paths, state = setup_closed_baseline(root, initialize=False)
    state = initialize_scientific_v2_publication(
        paths,
        state,
        producer=scientific_producer(args),
        closed_scientific_snapshot=True,
        atomic_reevaluation=True,
    )
    return args, paths, state


def make_proposal(
    paths: Any,
    identifier: str = "proposal-1",
    *,
    empty: bool = False,
    wrap: bool = True,
) -> Any:
    """Capture an exact current predecessor and next continuation for a scalar patch."""
    state = load_bootstrap_run_state(paths.run_state_path)
    captured = transaction_state(state)
    assert captured is not None
    base = fixture_patch(identifier)
    previous = captured["accepted_cursor"]
    next_position = None if wrap else "root"
    patch = replace(
        base,
        tree_generation=state.generation,
        rows=() if empty else base.rows,
        metadata={
            **base.metadata,
            "completed_full_pass": wrap,
            "next_node_cursor": next_position,
            "scientific_publication": committed_scientific_source(
                paths.work_dir / state.latest_tree_snapshot_path
            ),
        },
    )
    cursor = MorpionReevaluationCursor(
        evaluator_generation=patch.evaluator_generation,
        evaluator_name=patch.evaluator_name,
        model_bundle_path=patch.model_bundle_path,
        next_node_cursor=next_position,
        updated_at_utc=patch.created_at_utc,
        tree_generation=patch.tree_generation,
        completed_full_pass_count=(
            0 if previous is None else previous["completed_full_pass_count"]
        )
        + int(wrap),
        last_patch_id=patch.patch_id,
        metadata={"source": "reevaluation_worker"},
    )
    return proposal_patch(
        patch,
        captured,
        source_identity(state, state.generation),
        current_model_identity(paths),
        cursor,
    )


def publish_proposal(paths: Any, **kwargs: Any) -> Any:
    """Publish without any singleton cursor write."""
    patch = make_proposal(paths, **kwargs)
    assert publish_reevaluation_patch_if_absent(
        patch, paths.pipeline_reevaluation_patch_path
    )
    assert not paths.pipeline_reevaluation_cursor_path.exists()
    return patch


def accepted(state: Any) -> Any:
    """Read only the authoritative accepted cursor."""
    return state.metadata[TRANSACTION_KEY]["accepted_cursor"]


def atomic_worker(
    root: Path,
    monkeypatch: Any,
    *,
    streaming: bool = True,
    identifier: str = "worker-proposal",
) -> Any:
    """Run the real worker with an explicit closed-source test adapter and scalar model."""
    from functools import partial

    from chipiron.environments.morpion.bootstrap import reevaluation_worker as worker
    from chipiron.environments.morpion.bootstrap.closed_scientific_snapshot import (
        load_closed_scientific_snapshot,
    )
    from chipiron.environments.morpion.bootstrap.sharded_training_reader import (
        ShardedTrainingGenerationView,
    )
    from tests.environments.morpion_durable_v2_helpers import scientific_args
    from tests.environments.test_morpion_scientific_training_reader import StateScore

    def evaluator(**_kwargs: Any) -> Any:
        result = worker.MorpionActiveModelNodeReevaluationEvaluator(
            root / "unused-model"
        )
        result._evaluator = StateScore()
        return result

    monkeypatch.setattr(worker, "build_active_model_reevaluation_evaluator", evaluator)
    monkeypatch.setattr(worker, "is_sharded_generation_manifest", lambda _: True)
    monkeypatch.setattr(
        worker,
        "ShardedTrainingGenerationView",
        partial(ShardedTrainingGenerationView, allow_closed_snapshot=True),
    )
    monkeypatch.setattr(
        worker,
        "load_reevaluation_training_tree_snapshot",
        lambda p: load_closed_scientific_snapshot(p).snapshot,
    )
    return worker.run_morpion_reevaluation_worker_once(
        scientific_args(root),
        max_nodes_per_patch=1,
        patch_id=identifier,
        now_unix_s=1e9,
        use_streaming_sharded_reader=streaming,
        allow_scientific_v2=True,
    )

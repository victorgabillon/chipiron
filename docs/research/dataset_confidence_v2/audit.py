"""Characterize frozen Dataset v1 and expose the unresolved v2 proof decision.

This is an investigation driver, not a confidence-policy implementation. All
inputs are newly generated. Both full and indexed consumers run existing code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, get_args

import pytest
from anemone.checkpoints import checkpoint_payload_to_jsonable, serialize_value
from anemone.node_evaluation.common import canonical_value
from anemone.training_export import (
    build_training_tree_snapshot,
    load_training_tree_snapshot,
    save_training_tree_snapshot,
)
from anemone.training_export.serialization import training_tree_snapshot_to_dict
from atomheart.games.morpion import MorpionDynamics, initial_state

from chipiron.environments.morpion.bootstrap import MorpionBootstrapArgs
from chipiron.environments.morpion.bootstrap.cycle_dataset import (
    extract_rows_from_training_snapshot,
)
from chipiron.environments.morpion.bootstrap.pv_family_targets import (
    PvFamilyTargetPolicy,
)
from chipiron.environments.morpion.bootstrap.sharded_training_reader import (
    ShardedTrainingGenerationView,
)
from chipiron.environments.morpion.bootstrap.sharded_training_v2 import (
    load_scientific_training_generation,
    save_scientific_training_generation,
    value_score,
)
from chipiron.environments.morpion.bootstrap.streaming_dataset import (
    IndexedDatasetSource,
    disk_dataset_context,
)
from chipiron.environments.morpion.learning import (
    MorpionSupervisedRows,
    load_morpion_supervised_rows,
    morpion_supervised_rows_to_dict,
    save_morpion_supervised_rows_streaming,
)
from chipiron.environments.morpion.types import MorpionState
from tests.environments.morpion_scientific_reader_helpers import ScientificFixtureNode
from tests.environments.test_morpion_sharded_training_v2 import CODEC, Evaluation
from tests.environments.test_morpion_value_certainty import (
    endgame,
    make_runner,
    model_patch,
    neural_evaluator,
)

PRODUCER = {
    "revision": "77267b64a675602f9567920be74555999792cc68",
    "evaluator": "dataset-confidence-review-new-fixtures-only",
    "config_sha256": hashlib.sha256(
        b"dataset-confidence-review-fixtures-1"
    ).hexdigest(),
}


def encoded(value: object) -> bytes:
    """Serialize every field deterministically without dropping scientific data."""
    return json.dumps(
        checkpoint_payload_to_jsonable(value), sort_keys=True, ensure_ascii=True
    ).encode()


def sha(value: bytes) -> str:
    """Hash complete bytes."""
    return hashlib.sha256(value).hexdigest()


def first_legal_states() -> list[Any]:
    """Reach the exact PR #82 endgame by legal moves, without restoring data."""
    dynamics = MorpionDynamics()
    states = [initial_state()]
    for _ in range(200):
        actions = dynamics.all_legal_actions(states[-1])
        if not actions:
            return states
        states.append(dynamics.step(states[-1], actions[0]).next_state)
    pytest.fail("bounded legal fixture failed to terminate")


def family_counterexample(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[tuple[ScientificFixtureNode, ...], dict[str, Any]]:
    """Construct a PV family ending at 35 from a state admitting score 71.

    The first three nodes are explicitly partial estimates. The last two use
    the actual runner's default proof policy, not injected certified values.
    The alternative path is a lower-bound witness, not an optimality claim.
    """
    states = first_legal_states()
    assert len(states) == 36
    dynamics = MorpionDynamics()
    alternate = states[31]
    moves = [dynamics.all_legal_actions(alternate)[1]]
    alternate = dynamics.step(alternate, moves[-1]).next_state
    for _ in range(168):
        actions = dynamics.all_legal_actions(alternate)
        if not actions:
            break
        moves.append(actions[0])
        alternate = dynamics.step(alternate, moves[-1]).next_state
    assert not dynamics.all_legal_actions(alternate)
    assert 31 + len(moves) == 71
    runner = make_runner(monkeypatch, state=states[34])
    solved = tuple(runner.iter_profile_nodes())
    nodes = []
    for index, state in enumerate(states[31:]):
        evaluation = (
            Evaluation(
                canonical_value.make_estimate_value(score=42.0 - index),
                canonical_value.make_estimate_value(score=35.0),
            )
            if index < 3
            else solved[index - 3].tree_evaluation
        )
        nodes.append(
            ScientificFixtureNode(
                id=str(index),
                depth=index,
                state=state,
                tree_evaluation=evaluation,
                parent_ids=() if index == 0 else (str(index - 1),),
                child_ids=(str(index + 1),) if index < 4 else (),
                visit_count=1,
            )
        )
    witness = {
        "shared_state_ref": CODEC.dump_state_ref(states[31]),
        "family_terminal_state_ref": CODEC.dump_state_ref(states[35]),
        "family_terminal_score": 35,
        "alternative_moves_from_shared_state": moves,
        "alternative_terminal_state_ref": CODEC.dump_state_ref(alternate),
        "alternative_terminal_value": checkpoint_payload_to_jsonable(
            serialize_value(neural_evaluator(42).evaluate(alternate))
        ),
        "shared_state_value_lower_bound": 71,
        "claim": "family membership does not establish equal game values",
    }
    return tuple(nodes), witness


@dataclass
class ExportCase:
    """One tiny generated generation and an independent saved flat projection."""

    name: str
    generation: int
    path: Path
    flat_path: Path


def export_case(
    root: Path, name: str, generation: int, nodes: tuple[Any, ...]
) -> ExportCase:
    """Use unchanged authoritative builders, with exactly matched metadata."""
    snapshot = build_training_tree_snapshot(
        nodes,
        root_node_id="0",
        state_ref_dumper=lambda state: CODEC.dump_state_ref(
            state.to_atomheart_state() if isinstance(state, MorpionState) else state
        ),
        direct_value_extractor=value_score,
        tree_value_extractor=value_score,
        effective_value_extractor=value_score,
        target_value_extractor=value_score,
        metadata={"fixture": name, "generation": generation},
    )
    flat_path = root / f"{name}-{generation}-flat.json"
    save_training_tree_snapshot(snapshot, flat_path)
    path, _ = save_scientific_training_generation(
        nodes=nodes,
        root_node_id="0",
        output_dir=root / name,
        generation=generation,
        producer=PRODUCER,
        metadata={"fixture": name, "generation": generation},
    )
    return ExportCase(name, generation, path, flat_path)


def generated_cases(root: Path) -> tuple[list[ExportCase], dict[str, Any]]:
    """Produce real proof/model cases, ordinary estimates and a declared API case."""
    cases = []
    with pytest.MonkeyPatch.context() as monkeypatch:
        late, _ = endgame.__wrapped__()
        runner = make_runner(monkeypatch, state=late)
        cases.append(
            export_case(root, "actual-endgame", 1, tuple(runner.iter_profile_nodes()))
        )
        runner.apply_reevaluation_patch(model_patch(runner, monkeypatch, 42, root))
        cases.append(
            export_case(root, "actual-endgame", 2, tuple(runner.iter_profile_nodes()))
        )
        ordinary = make_runner(monkeypatch)
        cases.append(
            export_case(root, "ordinary", 1, tuple(ordinary.iter_profile_nodes()))
        )
        synthetic = make_runner(monkeypatch)
        parent, child = tuple(synthetic.iter_profile_nodes())
        parent.tree_evaluation.direct_value = canonical_value.make_forced_value(score=9)
        parent.tree_evaluation.tree_value = canonical_value.make_estimate_value(score=3)
        child.tree_evaluation.direct_value = canonical_value.make_estimate_value(
            score=2
        )
        child.tree_evaluation.tree_value = canonical_value.make_estimate_value(score=2)
        cases.append(
            export_case(
                root, "declared-api-proof", 1, tuple(synthetic.iter_profile_nodes())
            )
        )
        nodes, witness = family_counterexample(monkeypatch)
        cases.append(export_case(root, "unequal-state-family", 1, nodes))
    return cases, witness


def compare_case(
    case: ExportCase, root: Path, *, legacy_only: bool = False
) -> dict[str, Any]:
    """Compare literal outputs from flat/reference/indexed existing Dataset v1."""
    if not legacy_only:
        from chipiron.environments.morpion.learning.target_confidence import (
            SAFE_FAMILY_POLICIES,
            DatasetConfidenceError,
        )
    flat = load_training_tree_snapshot(case.flat_path)
    oracle = load_scientific_training_generation(case.path)
    assert training_tree_snapshot_to_dict(flat) == training_tree_snapshot_to_dict(
        oracle.snapshot
    )
    criteria = (
        {},
        {"require_exact_or_terminal": True},
        {"max_rows": 0},
        {"max_rows": 1},
        {"max_rows": 2},
        {"min_depth": 1},
        {"min_visit_count": 999},
    )
    comparisons = []
    with ShardedTrainingGenerationView(
        case.path, cache_dir=root / "cache", allow_scientific_v2=True
    ) as view:
        assert view.provenance == oracle.provenance
        assert tuple(view.iter_nodes_in_export_order()) == oracle.snapshot.nodes
        assert view.load_value_contracts(tuple(oracle.value_contracts)) == tuple(
            oracle.value_contracts.values()
        )
        for confidence_policy, policy, backup, blend, filters in product(
            ("legacy_v1",) if legacy_only else ("legacy_v1", "selected_value_v2"),
            get_args(PvFamilyTargetPolicy),
            (False, True),
            (0.0, 0.25, 1.0),
            criteria,
        ):
            args = MorpionBootstrapArgs(
                work_dir=root,
                dataset_family_target_policy=policy,
                dataset_family_prediction_blend=blend,
                use_backed_up_value=backup,
                **filters,
                **(
                    {}
                    if legacy_only
                    else {"dataset_confidence_policy": confidence_policy}
                ),
            )
            decision = {
                "confidence_policy": confidence_policy,
                "policy": policy,
                "use_backup": backup,
                "blend": blend,
                "filters": filters,
            }
            source_contract = (
                {}
                if confidence_policy == "legacy_v1"
                else {
                    "value_contracts": oracle.value_contracts,
                    "source_provenance": oracle.provenance,
                }
            )
            if (
                confidence_policy == "selected_value_v2"
                and policy not in SAFE_FAMILY_POLICIES
            ):
                for snapshot in (flat, oracle.snapshot):
                    with pytest.raises(
                        DatasetConfidenceError, match="gates pv_exact_then"
                    ):
                        extract_rows_from_training_snapshot(
                            args=args,
                            snapshot=snapshot,
                            generation=case.generation,
                            **source_contract,
                        )
                with (
                    pytest.raises(DatasetConfidenceError, match="gates pv_exact_then"),
                    disk_dataset_context(IndexedDatasetSource(view), args),
                ):
                    pytest.fail("gated policy accepted")
                comparisons.append({**decision, "status": "gated_all_three_paths"})
                continue
            expected = extract_rows_from_training_snapshot(
                args=args, snapshot=flat, generation=case.generation, **source_contract
            )
            reference = extract_rows_from_training_snapshot(
                args=args,
                snapshot=oracle.snapshot,
                generation=case.generation,
                **source_contract,
            )
            assert expected == reference
            with disk_dataset_context(IndexedDatasetSource(view), args) as context:
                stream = context.stream(case.generation)
                actual = MorpionSupervisedRows(tuple(stream.rows), stream.metadata)
                assert actual == reference
                output_paths = []
                for label, data in (
                    ("flat", expected),
                    ("reference", reference),
                    ("indexed", actual),
                ):
                    path = (
                        root
                        / "outputs"
                        / f"{case.name}-{case.generation}-{len(comparisons):04d}-{label}.jsonl"
                    )
                    save_morpion_supervised_rows_streaming(
                        rows=data.rows, metadata=data.metadata, path=path
                    )
                    assert load_morpion_supervised_rows(path) == data
                    output_paths.append(path)
                contents = [path.read_bytes() for path in output_paths]
                assert contents[0] == contents[1] == contents[2]
            comparisons.append({
                **decision,
                "status": "exact_parity",
                "all_three_jsonl_sha256": sha(contents[0]),
                "output": morpion_supervised_rows_to_dict(reference),
            })
        view.validate_source_identity()
    with ShardedTrainingGenerationView(
        case.path, cache_dir=root / "cache", allow_scientific_v2=True
    ) as reused:
        assert reused.index_reused
    return {
        "name": case.name,
        "generation": case.generation,
        "snapshot": training_tree_snapshot_to_dict(oracle.snapshot),
        "value_contracts": oracle.value_contracts,
        "source_provenance": oracle.provenance,
        "comparisons": comparisons,
        "cache_reused": True,
    }


def audit(root: Path, *, legacy_only: bool = False) -> dict[str, Any]:
    """Retain complete outputs and fingerprints; never infer a new proof rule."""
    root.mkdir(exist_ok=False)
    cases, witness = generated_cases(root)
    results = [compare_case(case, root, legacy_only=legacy_only) for case in cases]
    return {
        "status": "frozen legacy baseline"
        if legacy_only
        else "selected-source certainty implemented; cross-state proof policies gated",
        "integration_commit": PRODUCER["revision"],
        "family_counterexample": witness,
        "cases": results,
        "comparisons": sum(len(case["comparisons"]) for case in results),
        "source_fingerprints": {
            str(path.relative_to(root)): sha(path.read_bytes())
            for path in sorted(root.rglob("*.json*"))
            if "outputs" not in path.parts
        },
    }


def main() -> None:
    """Write evidence to a new scratch directory and refuse replacement."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--legacy-only", action="store_true")
    args = parser.parse_args()
    result = audit(args.root, legacy_only=args.legacy_only)
    (args.root / "results.json").write_bytes(encoded(result) + b"\n")


if __name__ == "__main__":
    main()

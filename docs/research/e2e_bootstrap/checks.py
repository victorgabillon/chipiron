"""Independent small-fixture checks; full objects exist only in this oracle process."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import torch
from anemone.checkpoints import (
    build_search_checkpoint_payload,
    checkpoint_payload_to_jsonable,
)
from anemone.checkpoints.sharded_io import (
    _group_manifest_shards,
    _load_mapping_json_shard,
    _only_manifest_shard,
)
from anemone.checkpoints.sharded_manifest import read_sharded_checkpoint_manifest

from chipiron.environments.morpion.bootstrap import MorpionBootstrapPaths
from chipiron.environments.morpion.bootstrap import dataset_publication as dataset
from chipiron.environments.morpion.bootstrap import training_publication as training
from chipiron.environments.morpion.bootstrap.checkpoint_patch_receipts import (
    receipt_ledger,
)
from chipiron.environments.morpion.bootstrap.closed_scientific_snapshot import (
    load_closed_scientific_snapshot,
    read_closed_manifest,
)
from chipiron.environments.morpion.bootstrap.cycle_dataset import (
    extract_rows_from_training_snapshot,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    committed_evaluator,
    file_digest,
    publication_records,
    validate_capsule,
)
from chipiron.environments.morpion.bootstrap.model_activation import _runner
from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
    load_reevaluation_patch,
)
from chipiron.environments.morpion.bootstrap.reevaluation_transaction import (
    validate_proposal,
)
from chipiron.environments.morpion.bootstrap.reevaluation_worker import (
    GROWTH_NUMERIC_POLICY,
    MorpionActiveModelNodeReevaluationEvaluator,
    select_reevaluation_node_window,
)
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from chipiron.environments.morpion.bootstrap.sharded_training_reader import (
    ShardedTrainingGenerationView,
)
from chipiron.environments.morpion.learning import (
    save_morpion_supervised_rows_streaming,
)
from docs.research.atomic_activation.measure import disk
from docs.research.e2e_bootstrap.worker import arguments


def digest(value: object) -> str:
    """Fingerprint a complete JSON value without dropping fields."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def reopen(paths: Any, state: Any) -> tuple[Any, dict, Path]:
    """Restore only the current authority through the real Growth API."""
    runner, effective = _runner(paths, state)
    model, physical = committed_evaluator(paths.work_dir, state)
    with torch.random.fork_rng(devices=[]):
        runner.load_or_create(
            paths.work_dir / state.latest_runtime_checkpoint_path, physical, effective
        )
    runner.validate_model_refresh_certificates()
    return runner, model, physical


def payload(runner: Any) -> dict:
    """Build the complete checkpoint oracle for this small graph."""
    return checkpoint_payload_to_jsonable(
        build_search_checkpoint_payload(
            runner._runtime, state_codec=runner._state_codec
        )
    )


def science(
    runner: Any,
    paths: Any,
    state: Any,
    destination: Path,
    *,
    generation: int | None = None,
) -> Path:
    """Re-export with original producer ordering and immutable creation origins."""
    records = publication_records(state.metadata)
    head = records[-1]
    target = state.generation if generation is None else generation
    previous = (
        (records[-2] if len(records) > 1 else None) if generation is None else head
    )
    producer = (
        read_closed_manifest(paths.work_dir / head["tree_snapshot_path"])["producer"][
            "identity"
        ]
        if generation is None
        else head["scientific_identity"]["producer"]["identity"]
    )
    return runner.export_closed_scientific_snapshot(
        destination,
        generation=target,
        producer=producer,
        committed_source=None
        if previous is None
        else paths.work_dir / previous["tree_snapshot_path"],
        previous_producer=None
        if previous is None
        else previous["scientific_identity"]["producer"]["identity"],
    )


def science_hashes(source: Path) -> dict[str, str]:
    """Hash entire scientific files without canonicalizing away differences."""
    return {p.name: file_digest(p) for p in source.parent.iterdir() if p.is_file()}


def growth_oracle(root: Path, out: Path) -> None:
    """Predict one continuation in a disposable process, without any publication."""
    out.mkdir(parents=True)
    paths = MorpionBootstrapPaths.from_work_dir(root)
    state = load_bootstrap_run_state(paths.run_state_path)
    runner, _, _ = reopen(paths, state)
    runner.grow(1)
    expected = payload(runner)
    source = science(
        runner,
        paths,
        state,
        out / "scientific",
        generation=state.generation + 1 if runner.last_growth_steps else None,
    )
    runner.discard_uncommitted_runtime()
    (out / "expected.json").write_text(
        json.dumps({"payload": expected, "science": science_hashes(source)}, indent=2)
        + "\n"
    )


def verify(root: Path, out: Path, *, full: bool) -> dict:
    """Verify authority, cold restore, per-value science and accepted original inputs."""
    out.mkdir(parents=True)
    paths = MorpionBootstrapPaths.from_work_dir(root)
    before = paths.run_state_path.read_bytes()
    state = load_bootstrap_run_state(paths.run_state_path)
    records = publication_records(state.metadata)
    runner, model, physical = reopen(paths, state)
    actual = payload(runner)
    checkpoint = root / state.latest_runtime_checkpoint_path
    manifest = read_sharded_checkpoint_manifest(checkpoint / "manifest.json")
    grouped = _group_manifest_shards(manifest)
    metadata = _load_mapping_json_shard(
        checkpoint, _only_manifest_shard(grouped, "metadata")
    )
    for field in (
        "evaluator_version",
        "format_version",
        "rng_state",
        "rollout_rng_state",
    ):
        assert actual[field] == metadata[field], field
    assert actual["format_version"] == 3 and actual["rollout_rng_state"] is not None
    assert actual["selector_state"] == _load_mapping_json_shard(
        checkpoint, _only_manifest_shard(grouped, "selector")
    )
    nodes = actual["tree"]["nodes"]
    result = {
        "checkpoint_payload_sha256": digest(actual),
        "evaluator_version": actual["evaluator_version"],
        "rng_sha256": digest(actual["rng_state"]),
        "rollout_rng_sha256": digest(actual["rollout_rng_state"]),
        "selector_sha256": digest(actual["selector_state"]),
        "nodes": len(nodes),
        "edges": sum(len(n["linked_children"]) for n in nodes),
        "checkpoint_disk": disk(checkpoint),
        "scientific_disk": disk((root / state.latest_tree_snapshot_path).parent),
        "model_disk": disk(physical),
        "receipts": {k: asdict(v) for k, v in receipt_ledger(state).items()},
    }
    for node in nodes:
        ev = node["evaluation"]
        direct = ev["direct_value"]
        if (
            direct is not None
            and direct["certainty"] == "ESTIMATE"
            and model["source"] == "local_training"
        ):
            assert ev["direct_evaluation_version"] == actual["evaluator_version"]
    if full:
        source = root / state.latest_tree_snapshot_path
        expected_source = science(runner, paths, state, out / "reexport")
        assert science_hashes(expected_source) == science_hashes(source)
        result["scientific_hashes"] = science_hashes(source)
        (out / "payload.json").write_text(json.dumps(actual, indent=2) + "\n")
    runner.discard_uncommitted_runtime()
    if full and "activation_request_sha256" in records[-1]:
        from chipiron.environments.morpion.players.evaluators.neural_evaluator import (
            load_morpion_evaluator_from_model_bundle,
        )
        from docs.research.model_activation.probe import reference as independent_max
        from docs.research.model_activation.probe import state as values

        previous = records[-2]
        publication_key = "durable_generation_publication_v1"
        # An oracle-only view of the immediately preceding immutable capsule;
        # this does NOT implement intentional rollback or rewrite run_state.
        old = replace(
            state,
            generation=previous["generation"],
            latest_runtime_checkpoint_path=previous["checkpoint_path"],
            latest_tree_snapshot_path=previous["tree_snapshot_path"],
            metadata={
                **state.metadata,
                publication_key: {
                    **state.metadata[publication_key],
                    "generations": records[:-1],
                    "head_id": previous["id"],
                },
            },
        )
        restored, _ = _runner(paths, old)
        _, previous_model = committed_evaluator(root, old)
        # Growth correctly refuses historical-head restore. The small independent
        # oracle calls its existing checkpoint loader without publishing anything.
        restored._runtime = restored._load_runtime_from_checkpoint(
            root / previous["checkpoint_path"],
            search_args=restored._args.search_args,
            model_bundle_path=previous_model,
        )
        restored.validate_model_refresh_certificates()
        with torch.random.fork_rng(devices=[]):
            evaluator = load_morpion_evaluator_from_model_bundle(physical)
        expected_values = independent_max(restored, evaluator)
        restored._runtime.refresh_with_evaluator(evaluator, scope="all", batch_size=64)
        assert values(restored) == expected_values
        assert payload(restored) == actual
        old_science = load_closed_scientific_snapshot(
            root / previous["tree_snapshot_path"]
        )
        new_science = load_closed_scientific_snapshot(source)
        assert [
            (n.state_ref_payload, n.parent_ids, n.child_ids)
            for n in old_science.snapshot.nodes
        ] == [
            (n.state_ref_payload, n.parent_ids, n.child_ids)
            for n in new_science.snapshot.nodes
        ]
        assert (root / previous["tree_snapshot_path"]).parent.joinpath(
            "records.jsonl"
        ).read_bytes().splitlines()[1:] == source.parent.joinpath(
            "records.jsonl"
        ).read_bytes().splitlines()[1:]
        result["activation_independent_max_and_full_checkpoint"] = True
        restored.discard_uncommitted_runtime()
    if full:
        reference = load_closed_scientific_snapshot(source)
        with ShardedTrainingGenerationView(
            source, cache_dir=out / "index", allow_closed_snapshot=True
        ) as view:
            assert tuple(view.iter_nodes_in_export_order()) == reference.snapshot.nodes
            ids = [n.node_id for n in reference.snapshot.nodes]
            assert view.load_value_contracts(ids) == tuple(
                reference.value_contracts[i] for i in ids
            )
            result["creation_generations_sha256"] = digest(
                view.load_creation_generations(ids)
            )
        result["certificates"] = {
            "terminal": sum(n.is_terminal for n in reference.snapshot.nodes),
            "nonterminal_forced": sum(
                not n.is_terminal
                and reference.value_contracts[n.node_id]["certainties"]["effective"]
                == "FORCED"
                for n in reference.snapshot.nodes
            ),
            "frontier_estimate": sum(
                not n.child_ids
                and reference.value_contracts[n.node_id]["certainties"]["effective"]
                == "ESTIMATE"
                for n in reference.snapshot.nodes
            ),
        }
        root_node = reference.snapshot.nodes[0]
        result["root_science"] = {
            "direct": root_node.direct_value_scalar,
            "tree": root_node.tree_value_scalar,
            "effective": root_node.effective_value_scalar,
            "contract": reference.value_contracts[root_node.node_id],
        }
        if paths.pipeline_reevaluation_patch_path.exists():
            patch = load_reevaluation_patch(paths.pipeline_reevaluation_patch_path)
            envelope = validate_proposal(state, patch)
            assert envelope["model"] == model
            assert envelope["schema_version"] == 2
            assert envelope["numeric_policy"] == GROWTH_NUMERIC_POLICY
            cursor = state.metadata["atomic_reevaluation"]["accepted_cursor"]
            same_model = cursor is not None and (
                cursor["model_bundle_path"],
                cursor["evaluator_generation"],
                cursor["evaluator_name"],
            ) == (model["path"], model["generation"], model["name"])
            selected, next_node, wrapped = select_reevaluation_node_window(
                tuple(sorted(n.node_id for n in reference.snapshot.nodes)),
                start_cursor=cursor["next_node_cursor"] if same_model else None,
                max_nodes=patch.metadata["max_nodes_per_patch"],
            )
            assert tuple(r.node_id for r in patch.rows) == selected
            assert patch.metadata["next_node_cursor"] == next_node
            assert patch.metadata["completed_full_pass"] == wrapped
            assert envelope["next_cursor"]["completed_full_pass_count"] == (
                cursor["completed_full_pass_count"] if same_model else 0
            ) + int(wrapped)
            evaluator = MorpionActiveModelNodeReevaluationEvaluator(
                model_bundle_path=root / model["path"],
                sealed_model_bundle_path=physical,
                numeric_policy=GROWTH_NUMERIC_POLICY,
            )
            expected_rows = evaluator.evaluate_patch_rows(
                reference.snapshot, [r.node_id for r in patch.rows]
            )
            assert expected_rows == patch.rows
            assert all(r.is_exact == r.is_terminal for r in patch.rows)
            raw = paths.pipeline_reevaluation_patch_path.read_bytes()
            (out / "pending.json").write_bytes(raw)
            result["pending"] = {
                "id": patch.patch_id,
                "sha256": file_digest(paths.pipeline_reevaluation_patch_path),
                "rows": len(patch.rows),
                "envelope": envelope,
            }
        accepted = []
        for record in records:
            validate_capsule(root, record)
            if "accepted_dataset" not in record:
                continue
            generation = record["generation"]
            identity = dataset.accepted_dataset_identity(paths, state, generation)
            proposal = dataset._proposal(paths, identity["reference"])
            original = dataset._validate_rows(paths, proposal)
            scientific = load_closed_scientific_snapshot(
                root / record["tree_snapshot_path"]
            )
            extracted = extract_rows_from_training_snapshot(
                args=arguments(root),
                snapshot=scientific.snapshot,
                generation=generation,
                value_contracts=scientific.value_contracts,
                source_provenance=scientific.provenance,
            )
            assert len(extracted.rows) > 0, (
                "E2E learning requires nonempty accepted data"
            )
            expected_rows_path = out / f"rows-{generation}.jsonl"
            save_morpion_supervised_rows_streaming(
                rows=extracted.rows,
                metadata=extracted.metadata,
                path=expected_rows_path,
            )
            assert original.read_bytes() == expected_rows_path.read_bytes()
            item = {
                "generation": generation,
                "dataset": identity,
                "rows": len(extracted.rows),
                "metadata": extracted.metadata,
            }
            if "accepted_training" in record:
                accepted_ref, selected = training.accepted_selected_model(
                    paths, state, generation
                )
                training_proposal = training._proposal(paths, accepted_ref)
                admission = training_proposal["admission"]
                assert admission["dataset"] == identity
                assert [
                    c["name"] for c in admission["configuration"]["candidates"]
                ] == ["linear_a", "linear_b"]
                assert all(
                    r["num_train_samples"] > 0
                    for r in training_proposal["status"]["evaluator_results"].values()
                ), "Initialized weights are not trained models"
                item["training"] = {
                    "reference": accepted_ref,
                    "selected": selected,
                    "proposal": training_proposal,
                }
            accepted.append(item)
        result["accepted_results"] = accepted
    assert paths.run_state_path.read_bytes() == before
    return result

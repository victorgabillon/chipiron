"""Read-only audit of accepted continual lineages and real activation checkpoints.

Run after worker shutdown, using the verified installed environment:
python docs/research/continual_learning/audit.py WORKSPACE OUTPUT [--science]
No experiment authority is written; OUTPUT must be outside WORKSPACE.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
from dataclasses import asdict, replace
from itertools import batched
from pathlib import Path
from typing import Any

from anemone.checkpoints import (
    build_search_checkpoint_payload,
    checkpoint_payload_to_jsonable,
)
from valanga.evaluations import Certainty, Value

from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
    MorpionBootstrapPaths,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    PUBLICATION_KEY,
    artifact_digest,
    committed_evaluator,
    file_digest,
    publication_records,
)
from chipiron.environments.morpion.bootstrap.model_activation import (
    _runner,
    activation_completion,
)
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from chipiron.environments.morpion.players.evaluators.neural_evaluator import (
    load_morpion_evaluator_from_model_bundle,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.bundle import (
    load_morpion_model_bundle,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training.continuation import (
    weights_digest,
)


def read(path: Path) -> Any:
    """Read one retained JSON artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def payload(instance: Any) -> Any:
    """Use the existing complete checkpoint equality oracle, including both RNGs."""
    return checkpoint_payload_to_jsonable(
        build_search_checkpoint_payload(
            instance._require_runtime(), state_codec=instance._state_codec
        )
    )


def historical_runtime(paths: Any, state: Any, bundle: Path, effective: Any) -> Any:
    """Read a validated old capsule through its codec; never adopt it for Growth.

    The operator correctly refuses old-head restoration. This offline oracle
    reads its immutable bytes directly, with no save/publication operation.
    """
    instance, _ = _runner(paths, state)
    instance._runtime = instance._load_runtime_from_checkpoint(
        paths.work_dir / state.latest_runtime_checkpoint_path,
        search_args=instance._args.search_args,
        model_bundle_path=bundle,
    )
    instance._last_applied_runtime_config = effective
    return instance


def ordered_nodes(instance: Any) -> list[Any]:
    """Use the existing depth registry, independent of checkpoint record order."""
    descendants = instance._require_runtime().tree.descendants
    return [n for depth in descendants.range() for n in descendants[depth].values()]


def values(instance: Any) -> dict[int, Any]:
    """Observe direct, tree, effective, source and complete selected PV."""
    return {
        n.id: dict(
            direct=n.tree_evaluation.direct_value,
            tree=n.tree_evaluation.tree_value,
            effective=n.tree_evaluation.effective_value,
            source=n.tree_evaluation.effective_value_source.value,
            pv=tuple(n.tree_evaluation.best_branch_sequence),
        )
        for n in ordered_nodes(instance)
    }


def independent_values(instance: Any, evaluator: Any) -> dict[int, Any]:
    """Batched version of the existing research max oracle; no backup/selector API.

    Adapted from docs/research/model_activation/probe.py:reference. It computes
    all direct predictions and max backups independently of Anemone propagation.
    """

    def exact(value: Any) -> bool:
        return value is not None and value.certainty in (
            Certainty.FORCED,
            Certainty.TERMINAL,
        )

    nodes = ordered_nodes(instance)
    direct = {n.id: n.tree_evaluation.direct_value for n in nodes}
    pending = [
        n for n in nodes if not n.state.is_game_over() and not exact(direct[n.id])
    ]
    for batch in batched(pending, 64, strict=False):
        raw = evaluator.evaluate_batch_items(batch)
        for n, value in zip(batch, raw, strict=True):
            assert value.certainty == Certainty.ESTIMATE
            direct[n.id] = Value(
                score=(1 / 0.99999999) ** n.tree_depth * value.score,
                certainty=Certainty.ESTIMATE,
            )
    result: dict[int, Any] = {}
    for n in reversed(nodes):
        children = [(b, c) for b, c in n.iter_child_links() if c is not None]
        children.sort(
            key=lambda bc: (
                -result[bc[1].id]["effective"].score,
                -int(exact(result[bc[1].id]["effective"])),
                len(result[bc[1].id]["pv"]),
                bc[1].id,
                str(bc[0]),
            )
        )
        tree = None
        pv = ()
        if children:
            branch, child = children[0]
            chosen = result[child.id]["effective"]
            proven = (
                n.tree_node.all_branches_generated
                and len(children) == n.tree_node.child_link_count()
                and all(exact(result[c.id]["effective"]) for _, c in children)
            )
            tree = Value(
                score=chosen.score,
                certainty=Certainty.FORCED if proven else Certainty.ESTIMATE,
                over_event=chosen.over_event if proven else None,
            )
            pv = (branch, *result[child.id]["pv"])
        d = direct[n.id]
        if tree is None:
            effective, source = d, "direct_self"
        elif (
            d is None
            or n.tree_node.all_branches_generated
            or ((d.score, exact(d)) <= (tree.score, exact(tree)))
        ):
            effective, source = tree, "tree_child"
        else:
            effective, source = d, "direct_self"
        result[n.id] = dict(
            direct=d, tree=tree, effective=effective, source=source, pv=pv
        )
    return result


def audit(paths: MorpionBootstrapPaths, *, science: bool) -> dict[str, Any]:
    """Check every accepted tensor chain, and optionally the first two real refreshes."""
    monitor = paths.work_dir / "resource_monitor"
    usage = read(monitor / "usage.json")
    assert usage["reason"] != "running"
    assert read(monitor / f"session_{usage['session']}.json")["shutdown_confirmed"]
    authority = paths.run_state_path.read_bytes()
    state = load_bootstrap_run_state(paths.run_state_path)
    records = publication_records(state.metadata)
    previous: dict[str, Any] = {}
    rounds = []
    for record in records:
        if "accepted_training" not in record:
            continue
        ref = record["accepted_training"]
        proposal_path = (
            paths.work_dir
            / "training_publications_v1/objects"
            / ref["attempt_id"]
            / "proposal.json"
        )
        assert file_digest(proposal_path) == ref["descriptor_sha256"]
        proposal = read(proposal_path)
        candidates = {}
        for name, model_ref in proposal["models"].items():
            bundle = paths.work_dir / model_ref["path"]
            assert artifact_digest(bundle) == model_ref["sha256"]
            model, _, manifest = load_morpion_model_bundle(bundle)
            c = manifest.metadata["continuation"]
            assert weights_digest(model) == c["final_weights_sha256"]
            if name in previous:
                old = previous[name]
                assert c["loaded_previous_weights"] and c["optimizer_state_loaded"]
                assert c["starting_weights_sha256"] == old["tensor"]
                assert c["parent_bundle_sha256"] == old["bundle"]
                assert c["parent_weights_sha256"] == old["param"]
                assert c["starting_optimizer_step_count"] == old["steps"]
            else:
                initial, _, _ = load_morpion_model_bundle(bundle / "initial")
                assert weights_digest(initial) == c["starting_weights_sha256"]
                assert artifact_digest(bundle / "initial") == c["initial_bundle_sha256"]
            previous[name] = dict(
                tensor=c["final_weights_sha256"],
                bundle=model_ref["sha256"],
                param=file_digest(bundle / "param.pt"),
                steps=c["final_optimizer_step_count"],
            )
            candidates[name] = {k: v for k, v in c.items() if "rng" not in k}
        selection = proposal["status"]["metadata"]["continual_selection"]
        losses = selection["candidate_validation_losses"]
        assert set(losses) == set(candidates)
        assert selection["candidate"] == min(
            losses, key=lambda name: (losses[name], name)
        )
        assert selection["improves_incumbent"] == (
            losses[selection["candidate"]] < selection["incumbent"]["validation_loss"]
        )
        rounds.append(
            dict(
                generation=record["generation"],
                candidates=candidates,
                selection=selection,
                performance=proposal["status"]["metadata"]["round_performance"],
                candidate_metrics=proposal["status"]["evaluator_results"],
            )
        )
    activations = []
    for index, record in enumerate(records):
        if "activation_request_sha256" not in record:
            continue
        completion = activation_completion(paths, state, record)
        activations.append(dict(generation=record["generation"], completion=completion))
        if not science or len(activations) > 2:
            continue

        def historical(position: int) -> Any:
            metadata = copy.deepcopy(state.metadata)
            metadata[PUBLICATION_KEY]["generations"] = records[: position + 1]
            metadata[PUBLICATION_KEY]["head_id"] = records[position]["id"]
            r = records[position]
            return replace(
                state,
                generation=r["generation"],
                metadata=metadata,
                latest_runtime_checkpoint_path=r["checkpoint_path"],
                latest_tree_snapshot_path=r["tree_snapshot_path"],
            )

        old, new = historical(index - 1), historical(index)
        live, effective = _runner(paths, old)
        _, old_bundle = committed_evaluator(paths.work_dir, old)
        _, new_bundle = committed_evaluator(paths.work_dir, new)
        live = historical_runtime(paths, old, old_bundle, effective)
        evaluator = load_morpion_evaluator_from_model_bundle(new_bundle)
        expected = independent_values(live, evaluator)
        live._require_runtime().refresh_with_evaluator(
            evaluator, scope="all", batch_size=64
        )
        assert values(live) == expected
        expected_payload = payload(live)
        live.discard_uncommitted_runtime()
        restored, effective = _runner(paths, new)
        restored = historical_runtime(paths, new, new_bundle, effective)
        assert values(restored) == expected
        assert payload(restored) == expected_payload
        restored.validate_model_refresh_certificates()
        restored.discard_uncommitted_runtime()
        activations[-1]["independent_complete_refresh_and_checkpoint_equal"] = True
    continuation = []
    best_terminal = None
    if science:
        for _ in range(2):
            runner, effective = _runner(paths, state)
            _, bundle = committed_evaluator(paths.work_dir, state)
            runner.load_or_create(
                paths.work_dir / state.latest_runtime_checkpoint_path, bundle, effective
            )
            runner.validate_model_refresh_certificates()
            terminals = [
                n
                for n in ordered_nodes(runner)
                if n.tree_evaluation.direct_value is not None
                and n.tree_evaluation.direct_value.certainty == Certainty.TERMINAL
            ]
            if terminals:
                best = max(terminals, key=lambda n: len(n.state.played_moves))
                assert best.state.is_game_over()
                terminal_payload = runner._state_codec.dump_state_ref(best.state)
                replayed = runner._state_codec.load_state_ref(terminal_payload)
                assert replayed.is_game_over()
                assert len(replayed.played_moves) == len(best.state.played_moves)
                best_terminal = dict(
                    node_id=best.id,
                    moves_since_start=len(best.state.played_moves),
                    total_points=len(best.state.points),
                    certainty="TERMINAL",
                    state_ref_payload=terminal_payload,
                )
            runner.grow(1)
            continuation.append(
                hashlib.sha256(
                    json.dumps(payload(runner), sort_keys=True).encode()
                ).hexdigest()
            )
            runner.discard_uncommitted_runtime()
        assert continuation[0] == continuation[1]
    measurements = [
        read(p) for p in paths.work_dir.glob("pipeline/performance/*/*.json")
    ]
    assert paths.run_state_path.read_bytes() == authority
    return dict(
        workspace=str(paths.work_dir),
        generation=state.generation,
        nodes=state.tree_size_at_last_save,
        termination=state.metadata.get("experiment_termination"),
        best_certified=asdict(state.latest_record_status)
        if state.latest_record_status
        else None,
        independently_verified_best_terminal=best_terminal,
        publication_count=len(records),
        accepted_datasets=sum("accepted_dataset" in r for r in records),
        rounds=rounds,
        activations=activations,
        measurements=measurements,
        identical_offline_growth_continuations=continuation,
        authority_unchanged=True,
        resource_sessions=[
            read(p)
            for p in sorted(monitor.glob("session_*.json"))
            if not p.name.endswith(".native.json")
        ],
        native_resources=[
            read(p) for p in sorted(monitor.glob("session_*.native.json"))
        ],
    )


def main() -> None:
    """Audit only a fresh task-owned experiment; retain the JSON outside its authority."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--science", action="store_true")
    options = parser.parse_args()
    assert not options.output.resolve().is_relative_to(options.workspace.resolve())
    logging.disable(logging.CRITICAL)
    result = audit(
        MorpionBootstrapPaths.from_work_dir(options.workspace), science=options.science
    )
    options.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({
            k: v
            for k, v in result.items()
            if k not in ("rounds", "activations", "measurements")
        })
    )


if __name__ == "__main__":
    main()

"""Serial, fresh-process refresh kernel measurements; NO model activation commit."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from anemone.checkpoints import checkpoint_payload_to_jsonable, serialize_value

from chipiron.environments.morpion.bootstrap import (
    MorpionBootstrapEffectiveRuntimeConfig,
)
from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
    MorpionBootstrapPaths,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    artifact_digest,
    sync_artifact,
)
from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
    load_pipeline_manifest,
)
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from chipiron.environments.morpion.bootstrap.sharded_training_reader import (
    ShardedTrainingGenerationView,
)
from chipiron.environments.morpion.bootstrap.training_publication import (
    project_training,
)
from chipiron.environments.morpion.players.evaluators.neural_evaluator import (
    load_morpion_evaluator_from_model_bundle,
)
from docs.research.durable_scientific_v2 import measure as instruments
from docs.research.model_activation.probe import (
    backup_phase,
    direct_phase,
    graph,
    nodes,
    selector_phase,
)
from docs.research.rolling_snapshot.measure import measured, runner
from scripts.measure_morpion_generation_publication import disk_usage


def fingerprint(instance: Any) -> str:
    """Bounded stream of all current values, DAG links, PV and direct versions."""
    h = hashlib.sha256()
    for n in nodes(instance):
        e = n.tree_evaluation
        payload = dict(
            id=n.id,
            depth=n.tree_depth,
            parents=sorted(p.id for p in n.parent_nodes),
            children=sorted((str(b), c.id) for b, c in n.iter_child_links()),
            values=[
                None
                if v is None
                else checkpoint_payload_to_jsonable(serialize_value(v))
                for v in (e.direct_value, e.tree_value, e.effective_value)
            ],
            source=e.effective_value_source.value,
            pv=[str(b) for b in e.best_branch_sequence],
            version=e.direct_evaluation_version,
        )
        h.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    return h.hexdigest()


def link_iteration_digest(instance: Any) -> str:
    """Retain ordered link identity separately; graph equality does not hide it."""
    h = hashlib.sha256()
    for n in nodes(instance):
        h.update(
            repr((n.id, [(str(b), c.id) for b, c in n.iter_child_links()])).encode()
        )
    return h.hexdigest()


def main() -> None:
    """Measure one requested phase on new synthetic inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("mode", choices=("prepare", "refresh", "restart"))
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--accepted-model-workspace", type=Path)
    options = parser.parse_args()
    root = options.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "work").mkdir(exist_ok=True)
    logging.disable(logging.CRITICAL)
    instruments.options = SimpleNamespace(profile=True)
    result = {
        "mode": options.mode,
        "requested_nodes": options.count,
        "cuda_available": torch.cuda.is_available(),
    }
    source = root / "work/source-checkpoint"
    output = root / "work/refreshed-checkpoint"
    if options.mode == "prepare":
        assert not source.exists()
        instance, result["growth"] = measured(
            "growth", root, lambda: graph(options.count)
        )

        def save_source() -> None:
            instance.save_checkpoint(source)
            sync_artifact(source, root)

        _, result["checkpoint"] = measured("source-checkpoint", root, save_source)
        result.update(
            nodes=instance.current_tree_size(),
            edges=instance.current_tree_branch_count(),
            dag_nodes=sum(len(n.parent_nodes) > 1 for n in nodes(instance)),
            checkpoint_sha256=artifact_digest(source),
            science_sha256=fingerprint(instance),
            link_iteration_sha256=link_iteration_digest(instance),
            disk=disk_usage(source),
            max_depth=max(n.tree_depth for n in nodes(instance)),
        )
    elif options.mode == "refresh":
        assert not output.exists()
        expected = json.loads((root / "prepare.json").read_text())
        _, result["source_verification"] = measured(
            "source-verification",
            root,
            lambda: verify_digest(source, expected["checkpoint_sha256"]),
        )
        instance = runner()
        _, result["restore"] = measured(
            "restore",
            root,
            lambda: instance.load_or_create(
                source,
                None,
                MorpionBootstrapEffectiveRuntimeConfig(tree_branch_limit=100000),
            ),
        )
        assert fingerprint(instance) == expected["science_sha256"]
        result["restored_link_iteration_sha256"] = link_iteration_digest(instance)
        result["ordered_link_parity"] = (
            link_iteration_digest(instance) == expected["link_iteration_sha256"]
        )
        paths = MorpionBootstrapPaths.from_work_dir(
            options.accepted_model_workspace.resolve()
        )
        authority = load_bootstrap_run_state(paths.run_state_path)

        def validate_model() -> Any:
            manifest = load_pipeline_manifest(
                paths.pipeline_manifest_path_for_generation(1)
            )
            return project_training(paths, authority, manifest, verify=True)

        manifest, result["accepted_training_verification"] = measured(
            "accepted-model", root, validate_model
        )
        selected = manifest.selected_evaluator_name
        model = paths.work_dir / manifest.model_bundle_paths[selected]
        digest = artifact_digest(model)
        with torch.random.fork_rng(devices=[]):
            evaluator, result["model_loading"] = measured(
                "model-loading",
                root,
                lambda: load_morpion_evaluator_from_model_bundle(model),
            )
        rng = instance._random_generator.getstate()
        count = instance._runtime.node_selector.base._selection_step_count
        counts, result["inference"] = measured(
            "inference", root, lambda: direct_phase(instance, evaluator, 64)
        )
        backed, result["backup"] = measured(
            "backup", root, lambda: backup_phase(instance)
        )
        _, result["selector"] = measured(
            "selector", root, lambda: selector_phase(instance)
        )
        assert rng == instance._random_generator.getstate()
        assert count == instance._runtime.node_selector.base._selection_step_count

        def save_output() -> None:
            instance.save_checkpoint(output)
            sync_artifact(output, root)

        _, result["checkpoint"] = measured("checkpoint", root, save_output)

        def snapshot() -> Path:
            p = instance.export_closed_scientific_snapshot(
                root / "work/science",
                generation=1,
                producer=dict(
                    revision="dc142573-model-activation-research",
                    evaluator=digest,
                    config_sha256="0" * 64,
                ),
            )
            sync_artifact(p.parent, root)
            return p

        path, result["snapshot"] = measured("snapshot", root, snapshot)

        def validate() -> int:
            with ShardedTrainingGenerationView(
                path, cache_dir=root / "work/index", allow_closed_snapshot=True
            ) as view:
                n = sum(1 for _ in view.iter_nodes_in_export_order())
                view.validate_source_identity()
                return n

        n, result["snapshot_validation"] = measured("validate", root, validate)
        assert n == instance.current_tree_size()
        result.update(
            counts,
            backed_up_nodes=backed,
            nodes=n,
            edges=instance.current_tree_branch_count(),
            model_path=str(model),
            model_sha256=digest,
            selected_candidate=selected,
            checkpoint_sha256=artifact_digest(output),
            science_sha256=fingerprint(instance),
            link_iteration_sha256=link_iteration_digest(instance),
            rng_sha256=hashlib.sha256(repr(rng).encode()).hexdigest(),
            checkpoint_disk=disk_usage(output),
            snapshot_disk=disk_usage(path.parent),
            temporary_disk=disk_usage(root / "work"),
            snapshot_stats=instance._latest_training_export_stats,
            scientific_commit_performed=False,
        )
    else:
        expected = json.loads((root / "refresh.json").read_text())
        instance = runner()
        _, result["verification"] = measured(
            "restart-verify",
            root,
            lambda: verify_digest(output, expected["checkpoint_sha256"]),
        )
        _, result["restore"] = measured(
            "restart-restore",
            root,
            lambda: instance.load_or_create(
                output,
                None,
                MorpionBootstrapEffectiveRuntimeConfig(tree_branch_limit=100000),
            ),
        )
        assert fingerprint(instance) == expected["science_sha256"]
        result["restored_link_iteration_sha256"] = link_iteration_digest(instance)
        result["ordered_link_parity"] = (
            link_iteration_digest(instance) == expected["link_iteration_sha256"]
        )
        assert (
            hashlib.sha256(
                repr(instance._random_generator.getstate()).encode()
            ).hexdigest()
            == expected["rng_sha256"]
        )
        result.update(
            exact_values_topology_pv_versions=True,
            exact_rng=True,
            nodes=instance.current_tree_size(),
        )
    (root / f"{options.mode}.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n"
    )
    print(
        json.dumps(
            dict(
                mode=options.mode,
                count=options.count,
                result=str(root / f"{options.mode}.json"),
            )
        ),
        flush=True,
    )


def verify_digest(path: Path, expected: str) -> None:
    """Verify actual bytes rather than a parent hash assertion."""
    assert artifact_digest(path) == expected


if __name__ == "__main__":
    main()

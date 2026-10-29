"""Serial fresh-process observations using the existing Morpion measurement tools."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from chipiron.environments.morpion.bootstrap import (
    MorpionBootstrapPaths,
    load_bootstrap_run_state,
    run_pipeline_growth_stage,
)
from chipiron.environments.morpion.bootstrap.closed_scientific_snapshot import (
    load_closed_scientific_snapshot,
)
from chipiron.environments.morpion.bootstrap.config import (
    bootstrap_config_from_args,
    bootstrap_config_sha256,
    bootstrap_config_to_dict,
)
from chipiron.environments.morpion.bootstrap.derived.provenance import atomic_json
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    publication_records,
    validate_capsule,
)
from chipiron.environments.morpion.bootstrap.run_state import (
    initialize_bootstrap_run_state,
)
from chipiron.environments.morpion.bootstrap.runtime.runner import (
    AnemoneMorpionSearchRunner,
    AnemoneMorpionSearchRunnerArgs,
)
from chipiron.environments.morpion.bootstrap.scientific_publication_initialization import (
    initialize_scientific_v2_publication,
)
from chipiron.environments.morpion.bootstrap.sharded_training_reader import (
    ShardedTrainingGenerationView,
)
from docs.research.closed_scientific_snapshot.measure import scientific_digest
from docs.research.durable_scientific_v2 import measure as instruments
from docs.research.rolling_snapshot.measure import measured
from docs.research.selected_generation.measure import operation
from scripts.measure_morpion_generation_publication import disk_usage
from tests.environments.morpion_durable_v2_helpers import scientific_args


class Runtime(AnemoneMorpionSearchRunner):
    """Runtime."""

    checkpoint_seconds = 0.0

    def save_checkpoint(self, path: Path) -> None:
        """Save checkpoint."""
        start = perf_counter()
        super().save_checkpoint(path)
        self.checkpoint_seconds += perf_counter() - start


def main() -> None:
    """Run one explicitly requested synthetic measurement."""
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("root", type=Path)
    p.add_argument(
        "mode",
        choices=[
            "initial",
            "repeat",
            "cold",
            "reuse",
            "oracle",
            "discover",
            "dataset",
            "dataset_stream",
            "continuation",
            "streams",
            "recover",
        ],
    )
    p.add_argument("--count", type=int, default=1000)
    p.add_argument("--history", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--encoding", choices=["jsonl-zstd-1"])
    opt = p.parse_args()
    root = opt.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    paths = MorpionBootstrapPaths.from_work_dir(root / "work")
    paths.ensure_directories()
    args = replace(
        scientific_args(paths.work_dir),
        runtime_checkpoint_format="sharded",
        save_after_seconds=0.0,
        tree_branch_limit=100000,
    )
    producer = {
        "revision": "durable-closed-generated-fixture",
        "evaluator": "default-runtime",
        "config_sha256": bootstrap_config_sha256(bootstrap_config_from_args(args)),
    }
    runner = Runtime(
        AnemoneMorpionSearchRunnerArgs(
            random_seed=4, runtime_checkpoint_format="sharded"
        )
    )
    instruments.options = SimpleNamespace(profile=opt.profile)
    logging.getLogger().setLevel(logging.WARNING)
    hashed = {"calls": 0, "bytes": 0}
    original = hashlib.file_digest

    def hashing(handle: Any, *a: Any, **kw: Any) -> Any:
        """Hashing."""
        hashed["calls"] += 1
        hashed["bytes"] += os.fstat(handle.fileno()).st_size
        return original(handle, *a, **kw)

    if opt.mode == "initial":
        assert not paths.run_state_path.exists()
        config = bootstrap_config_from_args(args)
        atomic_json(paths.bootstrap_config_path, bootstrap_config_to_dict(config))
        from chipiron.environments.morpion.bootstrap import (
            MorpionPipelineActiveModel,
            save_pipeline_active_model,
        )
        from tests.environments.morpion_patch_receipt_helpers import MODEL

        model = paths.work_dir / MODEL
        import torch

        from tests.environments.test_morpion_anemone_runner import _make_model_bundle

        torch.manual_seed(4)
        _make_model_bundle(model)
        save_pipeline_active_model(
            MorpionPipelineActiveModel(
                generation=1,
                evaluator_name="default",
                model_bundle_path=MODEL,
                updated_at_utc="2026-10-22T00:00:00Z",
            ),
            paths.pipeline_active_model_path,
        )
        from chipiron.environments.morpion.bootstrap import (
            MorpionBootstrapEffectiveRuntimeConfig,
        )

        runner.load_or_create(
            None, None, MorpionBootstrapEffectiveRuntimeConfig(tree_branch_limit=100000)
        )
        while runner.current_tree_size() < opt.count:
            runner.grow(1)
        count = runner.current_tree_size()

        def initial() -> Any:
            """Write and commit the initial frozen runtime."""
            checkpoint = paths.work_dir / "genesis-checkpoint.sharded"
            source = runner.export_closed_scientific_snapshot(
                paths.work_dir / "genesis-source",
                generation=1,
                producer=producer,
                encoding=opt.encoding,
            )
            runner.save_checkpoint(checkpoint)
            state = replace(
                initialize_bootstrap_run_state(),
                generation=1,
                tree_size_at_last_save=count,
                last_save_unix_s=1e9,
                latest_runtime_checkpoint_path=paths.relative_to_work_dir(checkpoint),
                latest_tree_snapshot_path=paths.relative_to_work_dir(source),
                metadata={"bootstrap_config_hash": producer["config_sha256"]},
            )
            return initialize_scientific_v2_publication(
                paths, state, producer=producer, closed_scientific_snapshot=True
            )

        with patch.object(hashlib, "file_digest", hashing):
            state, obs = measured("initial", root, initial)
        result = {
            "observation": obs,
            "hashing": hashed,
            "checkpoint_seconds": getattr(runner, "checkpoint_seconds", None),
            "writer": runner.latest_training_export_stats(),
        }
    else:
        state = load_bootstrap_run_state(paths.run_state_path)
        source = paths.work_dir / state.latest_tree_snapshot_path
        if opt.mode == "repeat":
            from chipiron.environments.morpion.bootstrap.generation_publication_io import (
                committed_scientific_source,
            )
            from tests.environments.morpion_patch_receipt_helpers import (
                fixture_patch,
                publish_fixture_patch,
            )

            previous_generation = state.generation
            with ShardedTrainingGenerationView(
                source, allow_closed_snapshot=True, cache_dir=root / "patch-cache"
            ) as view:
                node = view.load_nodes(["0"])[0]
                assert node.direct_value_scalar is not None
                values = {"0": node.direct_value_scalar}
            receipt_patch = fixture_patch(f"noop-{state.generation + 1}")
            receipt_patch = replace(
                receipt_patch,
                tree_generation=state.generation,
                rows=tuple(
                    replace(receipt_patch.rows[0], node_id=key, direct_value=value)
                    for key, value in values.items()
                ),
                metadata={
                    **receipt_patch.metadata,
                    "scientific_publication": committed_scientific_source(source),
                },
            )
            publish_fixture_patch(paths, receipt_patch)
            with patch.object(hashlib, "file_digest", hashing):
                state, obs = measured(
                    f"publication-{state.generation + 1}",
                    root,
                    lambda: run_pipeline_growth_stage(
                        args,
                        runner,
                        max_cycles=1,
                        checkpoint_bound_receipts=True,
                        durable_generation_publication=True,
                        scientific_v2_producer=producer,
                        closed_scientific_snapshot=True,
                    ),
                )
            assert state.generation == previous_generation + 1
            result = {
                "observation": obs,
                "hashing": hashed,
                "checkpoint_seconds": getattr(runner, "checkpoint_seconds", None),
                "writer": runner.latest_training_export_stats(),
            }
        elif opt.mode in ("cold", "reuse"):

            def read() -> Any:
                """Read."""
                with ShardedTrainingGenerationView(
                    source,
                    allow_closed_snapshot=True,
                    cache_dir=root
                    / ("profile-reader-cache" if opt.profile else "reader-cache"),
                ) as view:
                    digest = scientific_digest(
                        view.iter_nodes_in_export_order(),
                        lambda key: view.load_value_contracts([key])[0],
                        lambda key: view.load_creation_generations([key])[0],
                    )
                    view.validate_source_identity()
                    return {
                        "scientific_sha256": digest,
                        "initialization": view.initialization_observation,
                        "reused": view.index_reused,
                        "index_bytes": view.index_bytes,
                    }

            with patch.object(hashlib, "file_digest", hashing):
                value, obs = measured(opt.mode, root, read)
            assert value["reused"] == (opt.mode == "reuse")
            result = {"observation": obs, "hashing": hashed, **value}
        elif opt.mode == "streams":
            from chipiron.environments.morpion.bootstrap.scientific_stream import (
                open_scientific_stream,
            )
            from chipiron.environments.morpion.bootstrap.sharded_training_reader import (
                read_closed_manifest,
            )

            header = read_closed_manifest(source)

            def read_streams() -> Any:
                values = {}
                for key in ("record_shard", "update_shard"):
                    descriptor = header[key]
                    digest = hashlib.sha256()
                    size = 0
                    with open_scientific_stream(
                        source.parent / descriptor["path"], descriptor
                    ) as stream:
                        while data := stream.read(64 * 1024):
                            digest.update(data)
                            size += len(data)
                    values[key] = {"sha256": digest.hexdigest(), "bytes": size}
                return values

            value, obs = measured("streams", root, read_streams)
            result = {"observation": obs, "decoded": value}
        elif opt.mode == "recover":
            from chipiron.environments.morpion.bootstrap import (
                MorpionBootstrapEffectiveRuntimeConfig,
            )

            def recover() -> Any:
                validate_capsule(
                    paths.work_dir, publication_records(state.metadata)[-1]
                )
                runner.load_or_create(
                    paths.work_dir / state.latest_runtime_checkpoint_path,
                    None,
                    MorpionBootstrapEffectiveRuntimeConfig(tree_branch_limit=100000),
                )
                runner.validate_model_refresh_certificates()
                return runner.current_tree_size()

            value, obs = measured("recover", root, recover)
            result = {"observation": obs, "nodes": value}
        elif opt.mode in ("dataset", "dataset_stream"):
            from typing import get_args

            from chipiron.environments.morpion.bootstrap.cycle_dataset import (
                extract_rows_from_training_snapshot,
            )
            from chipiron.environments.morpion.bootstrap.pv_family_targets import (
                PvFamilyTargetPolicy,
            )
            from chipiron.environments.morpion.bootstrap.streaming_dataset import (
                IndexedDatasetSource,
                disk_dataset_context,
            )
            from chipiron.environments.morpion.learning import (
                save_morpion_supervised_rows_streaming,
            )
            from chipiron.environments.morpion.learning.target_confidence import (
                SAFE_FAMILY_POLICIES,
            )

            reference = (
                load_closed_scientific_snapshot(source)
                if opt.mode == "dataset"
                else None
            )
            pairs = []
            with ShardedTrainingGenerationView(
                source,
                allow_closed_snapshot=True,
                cache_dir=root
                / ("profile-reader-cache" if opt.profile else "reader-cache"),
            ) as view:
                policies = [
                    ("selected_value_v2", p) for p in sorted(SAFE_FAMILY_POLICIES)
                ]
                if opt.count == 1000:
                    policies += [
                        ("legacy_v1", p) for p in get_args(PvFamilyTargetPolicy)
                    ]
                for confidence, policy in policies:
                    configured = replace(
                        args,
                        dataset_confidence_policy=confidence,
                        dataset_family_target_policy=policy,
                    )
                    expected = (
                        extract_rows_from_training_snapshot(
                            args=configured,
                            snapshot=reference.snapshot,
                            generation=state.generation,
                            value_contracts=reference.value_contracts,
                            source_provenance=reference.provenance,
                        )
                        if reference is not None
                        else None
                    )
                    label = f"g{state.generation}-{confidence}-{policy}"
                    left = root / f"{label}-oracle.jsonl"
                    right = root / f"{label}-{opt.mode}.jsonl"
                    if expected is not None:
                        save_morpion_supervised_rows_streaming(
                            rows=expected.rows, metadata=expected.metadata, path=left
                        )

                    def write_dataset(
                        configured: Any = configured,
                        expected: Any = expected,
                        right: Path = right,
                    ) -> Any:
                        """Measure only the actual indexed extraction and serialization."""
                        with disk_dataset_context(
                            IndexedDatasetSource(view), configured
                        ) as context:
                            stream = context.stream(state.generation)
                            if expected is not None:
                                assert stream.metadata == expected.metadata
                            return save_morpion_supervised_rows_streaming(
                                rows=context.rows(),
                                metadata=stream.metadata,
                                path=right,
                            )

                    started = perf_counter()
                    observation = None
                    if opt.mode == "dataset_stream":
                        written, observation = measured(
                            f"dataset-{confidence}-{policy}", root, write_dataset
                        )
                    else:
                        written = write_dataset()
                    elapsed = perf_counter() - started
                    if expected is not None:
                        with left.open("rb") as a, right.open("rb") as b:
                            for x, y in zip(a, b, strict=True):
                                assert x == y
                    pairs.append({
                        "confidence": confidence,
                        "policy": policy,
                        "rows": written.row_count,
                        "indexed_seconds": elapsed,
                        "jsonl_sha256": instruments.sha(right),
                        "observation": observation,
                    })
            result = {
                "paired_jsonl_rows": sum(p["rows"] for p in pairs),
                "policies": pairs,
            }
        elif opt.mode == "continuation":
            from docs.research.rolling_snapshot.checkpoint_parity import (
                checkpoint_differences,
            )
            from docs.research.rolling_snapshot.measure import checkpoint_digest
            from tests.environments.test_morpion_checkpoint_parity import (
                _selection_signature,
            )

            first = publication_records(state.metadata)[0]
            baseline = paths.work_dir / first["checkpoint_path"]
            current = paths.work_dir / state.latest_runtime_checkpoint_path
            initial_sections, current_sections = (
                checkpoint_digest(baseline),
                checkpoint_digest(current),
            )
            left = Runtime(
                AnemoneMorpionSearchRunnerArgs(
                    random_seed=4, runtime_checkpoint_format="sharded"
                )
            )
            right = Runtime(
                AnemoneMorpionSearchRunnerArgs(
                    random_seed=999, runtime_checkpoint_format="sharded"
                )
            )
            from chipiron.environments.morpion.bootstrap import (
                MorpionBootstrapEffectiveRuntimeConfig,
            )

            runtime_config = MorpionBootstrapEffectiveRuntimeConfig(
                tree_branch_limit=100000
            )
            left.load_or_create(baseline, None, runtime_config)
            right.load_or_create(current, None, runtime_config)
            for _ in range(2):
                assert (
                    left._random_generator.getstate()
                    == right._random_generator.getstate()
                )
                assert left.current_tree_size() == right.current_tree_size()
                left.grow(1)
                right.grow(1)
                assert left.last_growth_steps == right.last_growth_steps == 1
                assert _selection_signature(left) == _selection_signature(right)
            assert (
                left._random_generator.getstate() == right._random_generator.getstate()
            )
            assert _selection_signature(left) == _selection_signature(right)
            result = {
                "selector_rng_continuation": True,
                "checkpoint_differences": checkpoint_differences(baseline, current),
                "initial_checkpoint_sections": initial_sections,
                "current_checkpoint_sections": current_sections,
                "raw_sections_differing": [
                    k
                    for k in initial_sections
                    if initial_sections[k] != current_sections[k]
                ],
            }
        elif opt.mode == "oracle":
            reference, obs = measured(
                "oracle", root, lambda: load_closed_scientific_snapshot(source)
            )
            result = {
                "observation": obs,
                "scientific_sha256": scientific_digest(
                    reference.snapshot.nodes,
                    reference.value_contracts.__getitem__,
                    reference.creation_generations.__getitem__,
                ),
                "nodes": len(reference.snapshot.nodes),
            }
            first = publication_records(state.metadata)[0]
            initial_reference = load_closed_scientific_snapshot(
                paths.work_dir / first["tree_snapshot_path"]
            )
            initial_digest = scientific_digest(
                initial_reference.snapshot.nodes,
                initial_reference.value_contracts.__getitem__,
                initial_reference.creation_generations.__getitem__,
            )
            assert result["scientific_sha256"] == initial_digest
            result["unchanged_generations_scientific_parity"] = True
            for mode in ("cold", "reuse"):
                earlier = json.loads(
                    (root / f"{mode}-g{state.generation}.json").read_text()
                )
                assert result["scientific_sha256"] == earlier["scientific_sha256"]
        else:
            with patch.object(hashlib, "file_digest", hashing):
                _, obs = measured("discover", root, lambda: operation(paths, "after"))
            result = {"observation": obs, "hashing": hashed}
    record = publication_records(state.metadata)[-1]
    seal = validate_capsule(paths.work_dir, record)
    result.update(
        generation=state.generation,
        requested_nodes=opt.count,
        fixed_population_history=opt.history,
        record=record,
        selected_files=seal["files"],
        workspace_disk=disk_usage(paths.work_dir),
        snapshot_disk=disk_usage(
            (paths.work_dir / state.latest_tree_snapshot_path).parent
        ),
        checkpoint_disk=disk_usage(
            paths.work_dir / state.latest_runtime_checkpoint_path
        ),
        run_state_bytes=paths.run_state_path.stat().st_size,
    )
    assert (
        len([e for e in seal["files"] if e["path"].startswith("scientific_snapshot/")])
        == 3
    )
    from chipiron.environments.morpion.bootstrap.performance import process_peak_rss_mb

    result["final_process_peak_rss_mb"] = process_peak_rss_mb()
    result["source_fingerprints"] = {e["path"]: e["sha256"] for e in seal["files"]}
    (
        root / f"{'profile-' if opt.profile else ''}{opt.mode}-g{state.generation}.json"
    ).write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({
            "mode": opt.mode,
            "generation": state.generation,
            "root": str(root),
        }),
        flush=True,
    )


if __name__ == "__main__":
    main()

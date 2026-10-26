"""Fixed legal graph, genuine neural receipts, existing publication and retirement.

This is an explicitly bounded metadata workload, not a search coordinator. It
reuses the normal format-3 writer and commit path without adding Growth nodes.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import psutil

from chipiron.environments.morpion.bootstrap import MorpionBootstrapPaths
from chipiron.environments.morpion.bootstrap.checkpoint_patch_receipts import (
    GrowthPatchReceiptSession,
    receipt_ledger,
)
from chipiron.environments.morpion.bootstrap.config import load_bootstrap_config
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    RETIRED_PAYLOAD_KEY,
    audit_generation_publications,
    committed_evaluator,
    generation_read_pin,
    growth_receipt_ownership,
    publication_records,
)
from chipiron.environments.morpion.bootstrap.launcher import _scientific_args
from chipiron.environments.morpion.bootstrap.model_activation import _runner
from chipiron.environments.morpion.bootstrap.reevaluation_worker import (
    run_morpion_reevaluation_worker_once,
)
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from docs.research.atomic_activation.measure import disk
from docs.research.e2e_bootstrap.checks import digest, payload
from docs.research.e2e_bootstrap.worker import arguments
from scripts.report_morpion_rolling_recovery import analyze, retire


def metadata_bytes(value: Any) -> int:
    """Count the reachable metadata Python objects once, independently of RSS."""
    seen: set[int] = set()

    def size(item: Any) -> int:
        if id(item) in seen:
            return 0
        seen.add(id(item))
        result = sys.getsizeof(item)
        if isinstance(item, dict):
            result += sum(size(k) + size(v) for k, v in item.items())
        elif isinstance(item, (tuple, list)):
            result += sum(size(v) for v in item)
        return result

    return size(value)


def sample(
    paths: MorpionBootstrapPaths, raw_patch: bytes, *, generation: int
) -> dict[str, Any]:
    """Cold runtime recovery in a warm process, plus separate validator traces."""
    started = time.perf_counter()
    state = load_bootstrap_run_state(paths.run_state_path)
    runner, effective = _runner(paths, state)
    _, model = committed_evaluator(paths.work_dir, state)
    try:
        runner.load_or_create(
            paths.work_dir / state.latest_runtime_checkpoint_path, model, effective
        )
        checkpoint = payload(runner)
    finally:
        runner.discard_uncommitted_runtime()
    recovery = time.perf_counter() - started
    before = paths.run_state_path.read_bytes()
    paths.pipeline_reevaluation_patch_path.write_bytes(raw_patch)
    producer = publication_records(state.metadata)[-1]["scientific_identity"][
        "producer"
    ]["identity"]
    session = GrowthPatchReceiptSession(
        paths,
        object(),
        durable_generation_publication=True,
        scientific_v2_producer=producer,
        closed_scientific_snapshot=True,
    )
    started = time.perf_counter()
    with growth_receipt_ownership(paths):
        assert session.acknowledge_pending(state)
    ack = time.perf_counter() - started
    assert paths.run_state_path.read_bytes() == before

    def trace_probe(operation: Any) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        previous_profile = sys.getprofile()

        def observe(frame: Any, event: str, _arg: Any) -> None:
            if (
                event == "call"
                and frame.f_code.co_name == "validate_capsule"
                and frame.f_globals.get("__name__", "").endswith(
                    "generation_publication_io"
                )
            ):
                record = frame.f_locals["record"]
                calls.append({
                    "generation": record["generation"],
                    "retired_payload": RETIRED_PAYLOAD_KEY in record,
                    "historical_full_payload": record["generation"] < state.generation
                    and RETIRED_PAYLOAD_KEY not in record,
                })

        sys.setprofile(observe)
        try:
            operation()
        finally:
            sys.setprofile(previous_profile)
        return calls

    def restore_probe() -> None:
        fresh = load_bootstrap_run_state(paths.run_state_path)
        runtime, budget = _runner(paths, fresh)
        _, physical = committed_evaluator(paths.work_dir, fresh)
        try:
            runtime.load_or_create(
                paths.work_dir / fresh.latest_runtime_checkpoint_path, physical, budget
            )
        finally:
            runtime.discard_uncommitted_runtime()

    def ack_probe() -> None:
        paths.pipeline_reevaluation_patch_path.write_bytes(raw_patch)
        with growth_receipt_ownership(paths):
            assert session.acknowledge_pending(state)

    traces = {
        "cold_runtime_recovery": trace_probe(restore_probe),
        "exact_old_receipt_ack": trace_probe(ack_probe),
        "explicit_archive_audit": trace_probe(
            lambda: audit_generation_publications(paths.work_dir)
        ),
    }
    assert paths.run_state_path.read_bytes() == before
    records = publication_records(state.metadata)
    return {
        "generation": generation,
        "validation_traces": traces,
        "nodes": len(checkpoint["tree"]["nodes"]),
        "edges": sum(len(n["linked_children"]) for n in checkpoint["tree"]["nodes"]),
        "checkpoint_payload_sha256": digest(checkpoint),
        "selector_sha256": digest(checkpoint["selector_state"]),
        "rng_sha256": digest(checkpoint["rng_state"]),
        "rollout_rng_sha256": digest(checkpoint["rollout_rng_state"]),
        "recovery_wall_s": recovery,
        "first_exact_receipt_ack_wall_s": ack,
        "receipt_count": len(receipt_ledger(state)),
        "authority_bytes": len(before),
        "metadata_owned_heap_bytes": metadata_bytes(state.metadata),
        "complete_boundaries": [
            r["generation"] for r in records if RETIRED_PAYLOAD_KEY not in r
        ],
        "allocation": disk(paths.work_dir),
        "process_rss_bytes": psutil.Process().memory_info().rss,
    }


def main() -> None:
    """Use an already freshly initialized private G1; never initialize on read failure."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--retire-every", type=int, default=5)
    parser.add_argument("--max-seconds", type=int, default=1200)
    options = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    assert 4 <= options.generations <= 100 and 0 <= options.retire_every <= 20
    out = options.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    paths = MorpionBootstrapPaths.from_work_dir(options.workspace.resolve())
    initial = load_bootstrap_run_state(paths.run_state_path)
    assert initial.generation == 1 and not receipt_ledger(initial)
    args = _scientific_args(
        arguments(paths.work_dir), load_bootstrap_config(paths.bootstrap_config_path)
    )
    records = []
    first_patch = None
    started = time.monotonic()
    initial_node_count = initial.tree_size_at_last_save
    process = psutil.Process()
    for generation in range(2, options.generations + 1):
        assert time.monotonic() - started < options.max_seconds
        assert psutil.virtual_memory().available > 8 * 2**30
        assert shutil.disk_usage(paths.work_dir).free > 20 * 2**30
        assert process.memory_info().rss < 2 * 2**30
        assert disk(paths.work_dir)["allocated_bytes"] < 2 * 2**30
        io_before = process.io_counters()
        tick = time.perf_counter()
        with TemporaryDirectory(prefix="metadata-reader-", dir=out) as cache:
            result = run_morpion_reevaluation_worker_once(
                args,
                max_nodes_per_patch=7,
                use_streaming_sharded_reader=True,
                reader_cache_dir=cache,
                allow_scientific_v2=True,
            )
            scratch_bytes = disk(Path(cache))["allocated_bytes"]
            assert scratch_bytes < 2**30
        assert result.patch_written
        raw = paths.pipeline_reevaluation_patch_path.read_bytes()
        (out / f"patch-{generation}.json").write_bytes(raw)
        first_patch = raw if first_patch is None else first_patch
        with growth_receipt_ownership(paths):
            previous = load_bootstrap_run_state(paths.run_state_path)
            runner, effective = _runner(paths, previous)
            model, _ = committed_evaluator(paths.work_dir, previous)
            producer = publication_records(previous.metadata)[-1][
                "scientific_identity"
            ]["producer"]["identity"]
            session = GrowthPatchReceiptSession(
                paths,
                runner,
                durable_generation_publication=True,
                scientific_v2_producer=producer,
                closed_scientific_snapshot=True,
            )
            try:
                previous = session.prepare(previous, args)
                session.begin_cycle(paths.work_dir / model["path"])
                runner.load_or_create(
                    paths.work_dir / previous.latest_runtime_checkpoint_path,
                    session.runtime_model_path(None),
                    effective,
                )
                session.apply(previous, effective)
                checkpoint = session.checkpoint_path(
                    generation, args.runtime_checkpoint_format
                )
                runner.save_checkpoint(checkpoint)
                science = session.export_scientific(generation)
                next_state = replace(
                    previous,
                    generation=generation,
                    cycle_index=previous.cycle_index + 1,
                    latest_runtime_checkpoint_path=paths.relative_to_work_dir(
                        checkpoint
                    ),
                    latest_tree_snapshot_path=paths.relative_to_work_dir(science),
                )
                state = session.commit(previous, next_state)
            finally:
                runner.discard_uncommitted_runtime()
                if session.publication is not None:
                    session.publication.close()
        assert state.tree_size_at_last_save == initial_node_count
        # Normal reader admission creates/adopts the stable pin of this new head.
        # The analyzer itself must never fabricate a missing pin as GC evidence.
        with generation_read_pin(paths.work_dir / state.latest_tree_snapshot_path):
            pass
        publication_seconds = time.perf_counter() - tick
        allocated_before = disk(paths.work_dir)
        authority_before_retirement = paths.run_state_path.stat().st_size
        retired = None
        if options.retire_every and (
            generation % options.retire_every == 0 or generation == options.generations
        ):
            plan = analyze(paths, retirement_plan=True)
            assert plan["validated"], plan
            if plan["conditional_candidates"]:
                retired = retire(
                    paths, generations=tuple(plan["conditional_candidates"])
                )
        io_after = process.io_counters()
        item = {
            "generation": generation,
            "publication_wall_s": publication_seconds,
            "reader_scope_allocated_bytes_after_build": scratch_bytes,
            "authority_bytes": paths.run_state_path.stat().st_size,
            "authority_json_bytes_written": authority_before_retirement
            + (paths.run_state_path.stat().st_size if retired is not None else 0),
            "new_descriptor_bytes": (
                paths.work_dir
                / publication_records(state.metadata)[-1]["descriptor_path"]
            )
            .stat()
            .st_size,
            "process_write_bytes_delta": io_after.write_bytes - io_before.write_bytes,
            "process_write_chars_delta": io_after.write_chars - io_before.write_chars,
            "allocated_before_retirement": allocated_before,
            "allocated_after": disk(paths.work_dir),
            "retirement": retired,
        }
        if generation in {2, 5, 10, 25, 50, 75, options.generations}:
            item["sample"] = sample(paths, first_patch, generation=generation)
        records.append(item)
        (out / "measurements.json").write_text(json.dumps(records, indent=2) + "\n")
    (out / "budget.json").write_text(
        json.dumps(
            {
                "generations": options.generations,
                "retire_every": options.retire_every,
                "maximum_seconds": options.max_seconds,
                "actual_seconds": time.monotonic() - started,
                "maximum_rss_bytes": 2 * 2**30,
                "minimum_available_ram_bytes": 8 * 2**30,
                "minimum_disk_headroom_bytes": 20 * 2**30,
                "maximum_workspace_bytes": 2 * 2**30,
                "nodes_fixed": initial_node_count,
                "genuine_neural_receipts": options.generations - 1,
                "search_steps": 0,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

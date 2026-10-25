"""Matched receipt acknowledgement traces using existing counters and cProfile."""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import logging
import pstats
import resource
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import patch

import chipiron
from chipiron.environments.morpion.bootstrap import MorpionBootstrapPaths
from chipiron.environments.morpion.bootstrap import checkpoint_patch_receipts as cr
from chipiron.environments.morpion.bootstrap import generation_publication as gp
from chipiron.environments.morpion.bootstrap import generation_publication_io as io
from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
    reevaluation_patch_from_dict,
)
from chipiron.environments.morpion.bootstrap.reevaluation_publication import (
    publish_reevaluation_patch_if_absent,
)
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from scripts.benchmark_morpion_scientific_reader import costs, delta


def acknowledge(paths: MorpionBootstrapPaths) -> bool:
    """Use the real owned recovery API, never a synthetic receipt or runtime."""
    with io.growth_receipt_ownership(paths):
        state = load_bootstrap_run_state(paths.run_state_path)
        producer = io.publication_records(state.metadata)[-1]["scientific_identity"][
            "producer"
        ]["identity"]
        session = cr.GrowthPatchReceiptSession(
            paths,
            object(),
            durable_generation_publication=True,
            scientific_v2_producer=producer,
            closed_scientific_snapshot=True,
        )
        return session.acknowledge_pending(state)


def main() -> None:
    """Trace baseline and corrected code against copies of one generated workspace."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("operation", choices=("receipt", "head"))
    parser.add_argument("workspace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--proposal", type=Path)
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    root = args.workspace.resolve()
    paths = MorpionBootstrapPaths.from_work_dir(root)
    state = load_bootstrap_run_state(paths.run_state_path)
    authority = paths.run_state_path.read_bytes()
    records = io.publication_records(state.metadata)
    head_id = records[-1]["id"]
    raw = None if args.proposal is None else args.proposal.read_bytes()
    proposal = None if raw is None else reevaluation_patch_from_dict(json.loads(raw))
    if args.operation == "receipt":
        assert proposal is not None
        assert (
            cr.receipt_ledger(state)[proposal.patch_id].content_sha256
            == hashlib.sha256(raw).hexdigest()
        )
        if paths.pipeline_reevaluation_patch_path.exists():
            # Only the disposable copy's new M3 proposal is moved; retain its bytes.
            paths.pipeline_reevaluation_patch_path.rename(
                root.parent / "pending-preserved.json"
            )

    def publish() -> None:
        if proposal is not None:
            assert publish_reevaluation_patch_if_absent(
                proposal, paths.pipeline_reevaluation_patch_path
            )
            assert paths.pipeline_reevaluation_patch_path.read_bytes() == raw

    def operation() -> None:
        if args.operation == "receipt":
            assert acknowledge(paths)
        else:
            io.committed_evaluator(root, state)
        assert paths.run_state_path.read_bytes() == authority

    samples = []
    before = costs()
    for _ in range(5):
        publish()
        start = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - start)
    measured = delta(before)
    trace: dict[str, list[Any]] = {
        key: []
        for key in (
            "hashes",
            "descriptors",
            "complete_capsules",
            "model_bundles",
            "receipts",
        )
    }

    def relative(path: Path) -> str:
        return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)

    original_hash, original_read = hashlib.file_digest, io.read_object
    original_validate, original_model, original_ledger = (
        io.validate_capsule,
        io.artifact_digest,
        cr.receipt_ledger,
    )
    original_receipt_model = cr.artifact_sha256

    def hashed(stream: Any, *a: Any, **kw: Any) -> Any:
        path = Path(stream.name)
        trace["hashes"].append({"path": relative(path), "bytes": path.stat().st_size})
        return original_hash(stream, *a, **kw)

    def read(path: Path) -> Any:
        if path.name == ".generation_publication_v3.json":
            trace["descriptors"].append(relative(path))
        return original_read(path)

    def validate(work: Path, record: dict[str, Any]) -> Any:
        trace["complete_capsules"].append(record["generation"])
        return original_validate(work, record)

    def model(path: Path) -> str:
        if path.is_dir() and path.parent.name == "models":
            trace["model_bundles"].append(relative(path))
        return original_model(path)

    def receipt_model(path: Path) -> str:
        if path.is_dir() and path.parent.name == "models":
            trace["model_bundles"].append(relative(path))
        return original_receipt_model(path)

    def ledger(value: Any) -> Any:
        result = original_ledger(value)
        trace["receipts"].append({k: asdict(v) for k, v in result.items()})
        return result

    publish()
    profile = cProfile.Profile()
    with (
        patch.object(hashlib, "file_digest", hashed),
        patch.object(io, "read_object", read),
        patch.object(io, "validate_capsule", validate),
        patch.object(gp, "validate_capsule", validate),
        patch.object(io, "artifact_digest", model),
        patch.object(gp, "artifact_digest", model),
        patch.object(cr, "receipt_ledger", ledger),
        patch.object(cr, "artifact_sha256", receipt_model),
    ):
        profile.runcall(operation)
    profile.dump_stats(args.output.with_suffix(".prof"))
    with args.output.with_suffix(".profile.txt").open("w") as stream:
        pstats.Stats(profile, stream=stream).sort_stats("cumulative").print_stats(35)
    historical = [
        h
        for h in trace["hashes"]
        if "/objects/" in h["path"]
        and head_id not in h["path"]
        and ("/checkpoint/" in h["path"] or "/scientific_snapshot/" in h["path"])
    ]
    result = {
        "operation": args.operation,
        "implementation": chipiron.__file__,
        "authority_sha256": hashlib.sha256(authority).hexdigest(),
        "authority_bytes": len(authority),
        "authority_unchanged": True,
        "uninstrumented_seconds": samples,
        "median_seconds": statistics.median(samples),
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "five_calls_including_publication_costs": measured,
        "trace": trace,
        "historical_payload_hash_calls": len(historical),
        "historical_payload_hash_bytes": sum(h["bytes"] for h in historical),
        "file_hash_calls": len(trace["hashes"]),
        "file_hash_bytes": sum(h["bytes"] for h in trace["hashes"]),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print({
        k: result[k]
        for k in (
            "operation",
            "median_seconds",
            "file_hash_calls",
            "historical_payload_hash_calls",
        )
    })


if __name__ == "__main__":
    main()

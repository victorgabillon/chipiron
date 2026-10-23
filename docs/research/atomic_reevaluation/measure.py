"""Fresh-process two-node protocol comparison using the existing resource instruments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from chipiron.environments.morpion.bootstrap import load_bootstrap_run_state
from chipiron.environments.morpion.bootstrap.checkpoint_patch_receipts import (
    GrowthPatchReceiptSession,
)
from docs.research.durable_scientific_v2 import measure as instruments
from scripts.measure_morpion_generation_publication import disk_usage
from tests.environments.morpion_durable_closed_helpers import (
    ClosedPublicationRunner,
    run_closed_publication,
    setup_closed_baseline,
)
from tests.environments.morpion_durable_v2_helpers import scientific_args


def main() -> None:
    """Measure one stage only; no worker processes or historical input."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("protocol", choices=["legacy", "atomic"])
    parser.add_argument("stage", choices=["prepare", "accept", "recover", "worker"])
    opt = parser.parse_args()
    root = opt.root.resolve()
    instruments.options = SimpleNamespace(profile=False)
    work = root / "work"
    if opt.stage in ("prepare", "worker"):
        root.mkdir(parents=True, exist_ok=False)
        if opt.protocol == "atomic":
            from tests.environments.morpion_atomic_reevaluation_helpers import (
                setup_atomic,
            )

            args, paths, _state = setup_atomic(work)
        else:
            args, paths, _state = setup_closed_baseline(work)
    else:
        from chipiron.environments.morpion.bootstrap import MorpionBootstrapPaths

        args, paths = scientific_args(work), MorpionBootstrapPaths.from_work_dir(work)
    counters = {"calls": 0, "bytes": 0}
    digest = hashlib.file_digest

    def hashing(handle: Any, *a: Any, **kw: Any) -> Any:
        counters["calls"] += 1
        counters["bytes"] += os.fstat(handle.fileno()).st_size
        return digest(handle, *a, **kw)

    def operation() -> Any:
        if opt.stage == "worker":
            from functools import partial

            import pytest

            from chipiron.environments.morpion.bootstrap import (
                reevaluation_worker as worker,
            )
            from chipiron.environments.morpion.bootstrap.sharded_training_reader import (
                ShardedTrainingGenerationView,
            )
            from tests.environments.test_morpion_scientific_training_reader import (
                StateScore,
            )

            def evaluator(**_kwargs: Any) -> Any:
                value = worker.MorpionActiveModelNodeReevaluationEvaluator(
                    Path("models/generation_000001/default")
                )
                value._evaluator = StateScore()
                return value

            with pytest.MonkeyPatch.context() as adapter:
                adapter.setattr(
                    worker, "build_active_model_reevaluation_evaluator", evaluator
                )
                adapter.setattr(
                    worker, "is_sharded_generation_manifest", lambda _: True
                )
                adapter.setattr(
                    worker,
                    "ShardedTrainingGenerationView",
                    partial(ShardedTrainingGenerationView, allow_closed_snapshot=True),
                )
                return worker.run_morpion_reevaluation_worker_once(
                    args,
                    max_nodes_per_patch=1,
                    patch_id="measured-proposal",
                    now_unix_s=1e9,
                    use_streaming_sharded_reader=True,
                    allow_scientific_v2=True,
                )
        if opt.stage == "prepare":
            # Legacy proposal has exactly the same scientific rows/source. Only the
            # new transaction envelope and independently saved cursor differ.
            if opt.protocol == "atomic":
                from chipiron.environments.morpion.bootstrap.reevaluation_publication import (
                    publish_reevaluation_patch_if_absent,
                )
                from tests.environments.morpion_atomic_reevaluation_helpers import (
                    make_proposal,
                )

                value = make_proposal(paths)
                return publish_reevaluation_patch_if_absent(
                    value, paths.pipeline_reevaluation_patch_path
                )
            from tests.environments.morpion_durable_v2_helpers import (
                publish_scientific_patch,
            )

            return publish_scientific_patch(paths)
        runner = ClosedPublicationRunner()
        if opt.stage == "accept":
            original = GrowthPatchReceiptSession.acknowledge_pending

            def lost_ack(session: Any, current: Any) -> Any:
                return False if current.generation == 2 else original(session, current)

            with patch.object(
                GrowthPatchReceiptSession, "acknowledge_pending", lost_ack
            ):
                return run_closed_publication(args, runner)
        return run_closed_publication(args, runner, max_cycles=0)

    with patch.object(hashlib, "file_digest", hashing):
        _, observation = instruments.measured(opt.stage, root, operation)
    current = load_bootstrap_run_state(paths.run_state_path)
    source = work / current.latest_tree_snapshot_path
    scientific = {
        name: hashlib.sha256((source.parent / name).read_bytes()).hexdigest()
        for name in ("records.jsonl", "updates.jsonl")
    }
    metadata = current.metadata
    payload = {
        "protocol": opt.protocol,
        "stage": opt.stage,
        "observation": observation,
        "hashing": counters,
        "disk": disk_usage(work),
        "scientific": scientific,
        "active_cursor_bytes": len(
            json.dumps(metadata.get("atomic_reevaluation"), sort_keys=True)
        ),
        "receipt_bytes": len(
            json.dumps(metadata["checkpoint_bound_reevaluation_v1"], sort_keys=True)
        ),
        "run_state_bytes": paths.run_state_path.stat().st_size,
        "pending": paths.pipeline_reevaluation_patch_path.exists(),
    }
    if payload["pending"]:
        rows = json.loads(paths.pipeline_reevaluation_patch_path.read_text())["rows"]
        payload["patch_rows_sha256"] = hashlib.sha256(
            json.dumps(rows, sort_keys=True).encode()
        ).hexdigest()
    (root / f"{opt.stage}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload))


if __name__ == "__main__":
    main()

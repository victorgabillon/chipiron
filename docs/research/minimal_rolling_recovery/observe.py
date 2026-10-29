"""Measure the read-only report with existing RSS/CPU/I/O instrumentation."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from chipiron.environments.morpion.bootstrap import (
    MorpionBootstrapPaths,
    load_bootstrap_run_state,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    generation_read_pin,
)
from docs.research.durable_scientific_v2 import measure as instruments
from scripts.report_morpion_rolling_recovery import analyze


def observe(root: Path) -> None:
    """Prepare a reader pin on the new fixture, then measure only read-only analysis."""
    paths = MorpionBootstrapPaths.from_work_dir(root / "work")
    state = load_bootstrap_run_state(paths.run_state_path)
    with generation_read_pin(paths.work_dir / state.latest_tree_snapshot_path):
        pass
    # The test fingerprint also checks mtime/ctime/inodes and all control files.
    from tests.environments.test_morpion_rolling_recovery_report import fingerprint

    before = fingerprint(paths.work_dir)
    instruments.options = SimpleNamespace(profile=False)
    hashed = {"calls": 0, "bytes": 0}
    original = hashlib.file_digest

    def counted(handle: Any, *args: Any, **kwargs: Any) -> Any:
        hashed["calls"] += 1
        hashed["bytes"] += os.fstat(handle.fileno()).st_size
        return original(handle, *args, **kwargs)

    with patch.object(hashlib, "file_digest", counted):
        report, costs = instruments.measured(
            "reachability", root, lambda: analyze(paths)
        )
    assert report["validated"], report
    assert fingerprint(paths.work_dir) == before
    assert report["proposed_active_roots"]["additional"] == []
    assert report["conditional_candidates"] == list(range(1, state.generation - 2))
    result = {
        "report": report,
        "costs": costs,
        "hashing": hashed,
        "generation": state.generation,
        "nodes": state.tree_size_at_last_save,
        "historical_metrics_bytes": paths.history_jsonl_path.stat().st_size,
        "historical_stage_metrics_bytes": sum(
            p.stat().st_size
            for p in (paths.pipeline_dir / "performance").rglob("*.json")
        ),
        "exact_workspace_fingerprint_unchanged": True,
    }
    (root / f"reachability-g{state.generation}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        json.dumps({
            "generation": state.generation,
            "active_root_bytes": report["active_root_bytes"],
            "receipt_bytes": report["receipt_metadata_bytes"],
        }),
        flush=True,
    )


if __name__ == "__main__":
    observe(Path(sys.argv[1]).resolve())

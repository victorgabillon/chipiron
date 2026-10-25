"""Matched-source index experiment using existing process/disk instrumentation.

Run serially in fresh processes, before and after installing the candidate wheel.
The source and model paths are deliberately identical between implementations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sqlite3
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from chipiron.environments.morpion.bootstrap import load_bootstrap_run_state
from chipiron.environments.morpion.bootstrap.reevaluation_worker import (
    GROWTH_NUMERIC_POLICY,
    RAW_NUMERIC_POLICY,
    MorpionActiveModelNodeReevaluationEvaluator,
)
from chipiron.environments.morpion.bootstrap.sharded_training_reader import (
    ShardedTrainingGenerationView,
)
from docs.research.durable_scientific_v2 import measure as instruments
from docs.research.rolling_snapshot.measure import measured as native_measurement
from docs.research.scientific_compression.measure import measured
from scripts.benchmark_morpion_scientific_reader import encode, sha
from tests.environments.morpion_patch_receipt_helpers import MODEL


def main() -> None:
    """Measure selected-window reads and neural patches without a full snapshot."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("label")
    parser.add_argument("mode", choices=("reads", "scope"))
    parser.add_argument("--native", action="store_true")
    opt = parser.parse_args()
    measure = native_measurement if opt.native else measured
    root = opt.root.resolve()
    state = load_bootstrap_run_state(root / "work/run_state.json")
    source = root / "work" / state.latest_tree_snapshot_path
    instruments.options = SimpleNamespace(profile=False)
    result = {"source": str(source), "manifest_sha256": sha(source)}
    hashes = {"calls": 0, "bytes": 0}
    original = hashlib.file_digest

    def hashing(handle: Any, *args: Any, **kwargs: Any) -> Any:
        hashes["calls"] += 1
        hashes["bytes"] += os.fstat(handle.fileno()).st_size
        return original(handle, *args, **kwargs)

    def opened(cache: Path) -> ShardedTrainingGenerationView:
        return ShardedTrainingGenerationView(
            source, cache_dir=cache, allow_closed_snapshot=True
        )

    if opt.mode == "scope":
        with TemporaryDirectory(dir=root, prefix="owned-session-") as directory:
            cache = Path(directory)
            with patch.object(hashlib, "file_digest", hashing):
                observations = []
                for _ in range(3):
                    with opened(cache) as view:
                        observations.append(view.initialization_observation)
                result["opens"] = observations
                result["retained_in_scope"] = [
                    p.stat().st_blocks * 512 for p in cache.iterdir()
                ]
        result["scope_removed"] = not cache.exists()
    else:
        with (
            patch.object(hashlib, "file_digest", hashing),
            opened(root / "reader-cache") as view,
        ):
            result["initialization"] = view.initialization_observation
            with sqlite3.connect(f"file:{view.index_path}?mode=ro", uri=True) as con:
                result["dbstat"] = list(
                    con.execute(
                        "SELECT name,sum(pgsize),sum(payload),sum(unused),count(*) FROM dbstat GROUP BY name"
                    )
                )
                result["payloads"] = {
                    table: con.execute(
                        f"SELECT count(*),sum(length(payload)),min(length(payload)),max(length(payload)),typeof(payload) FROM {table}"
                    ).fetchone()
                    for table in ("records", "updates")
                }
                result["schema"] = list(
                    con.execute("SELECT sql FROM sqlite_schema ORDER BY name")
                )
            rng = random.Random(123)
            # Bounded 256-ID sample, not a second in-memory tree.
            selected = view.select_sorted_node_window(None, 256)[0]
            keys = rng.choices(selected, k=2048)

            def random_reads() -> str:
                digest = hashlib.sha256()
                for key in keys:
                    node = view.load_nodes([key])[0]
                    digest.update(encode(asdict(node)))
                return digest.hexdigest()

            value, observation = measure("random", root, random_reads)
            result["random"] = {
                "sha256": value,
                "reads": len(keys),
                "observation": observation,
            }
            for policy in (RAW_NUMERIC_POLICY, GROWTH_NUMERIC_POLICY):
                adapter = MorpionActiveModelNodeReevaluationEvaluator(
                    root / "work" / MODEL, numeric_policy=policy
                )

                def reevaluate(policy: str = policy, adapter: Any = adapter) -> str:
                    cursor = None
                    remaining = view.node_count()
                    output = root / f"{opt.label}-patch-{policy}.jsonl"
                    with output.open("wb") as stream:
                        while remaining:
                            ids, cursor, wrapped = view.select_sorted_node_window(
                                cursor, min(128, remaining)
                            )
                            rows = adapter.evaluate_node_window(view.load_nodes(ids))
                            stream.write(
                                encode({
                                    "cursor": cursor,
                                    "wrapped": wrapped,
                                    "rows": [asdict(row) for row in rows],
                                })
                                + b"\n"
                            )
                            remaining -= len(ids)
                    view.validate_source_identity()
                    return sha(output)

                value, observation = measure("reevaluation", root, reevaluate)
                result[policy] = {
                    "sha256": value,
                    "rows": view.node_count(),
                    "observation": observation,
                }
    result["hashes"] = hashes
    (root / f"{opt.label}-{opt.mode}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps({"label": opt.label, "mode": opt.mode, "hashes": hashes}))


if __name__ == "__main__":
    main()

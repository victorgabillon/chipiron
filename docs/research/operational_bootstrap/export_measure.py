"""Matched frozen-source export measurements, reusing the existing instrumentation.

The before writer is extracted verbatim from 7326a6f5. Both runs cold-restore the
same committed legal graph and write offline G+1 snapshots; neither commits it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import tracemalloc
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
    MorpionBootstrapPaths,
)
from chipiron.environments.morpion.bootstrap.closed_scientific_snapshot import (
    save_closed_scientific_snapshot,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    artifact_digest,
    publication_records,
)
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from docs.research.durable_scientific_v2 import measure as instruments
from docs.research.e2e_bootstrap.checks import reopen
from docs.research.rolling_snapshot.measure import measured


def main() -> None:
    """Measure only the exporter after cold restore, with a separate trace pass."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--before", action="store_true")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    paths = MorpionBootstrapPaths.from_work_dir(args.workspace)
    state = load_bootstrap_run_state(paths.run_state_path)
    runner, _, _ = reopen(paths, state)
    head = publication_records(state.metadata)[-1]
    source = paths.work_dir / head["tree_snapshot_path"]
    output = args.destination.resolve()
    output.mkdir(parents=True, exist_ok=False)
    function = save_closed_scientific_snapshot
    source_code = None
    if args.before:
        source_code = subprocess.check_output(
            [
                "git",
                "show",
                "7326a6f5:src/chipiron/environments/morpion/bootstrap/closed_scientific_snapshot.py",
            ],
            text=True,
        )
        namespace = {
            "__name__": "chipiron.environments.morpion.bootstrap.closed_before",
            "__package__": "chipiron.environments.morpion.bootstrap",
        }
        exec(
            compile(source_code, "7326a6f5:closed_scientific_snapshot.py", "exec"),
            namespace,
        )
        function = namespace["save_closed_scientific_snapshot"]
    instruments.options = SimpleNamespace(profile=False)

    def export() -> Any:
        return function(
            nodes=runner._runtime._all_nodes_in_tree_order()
            if args.before
            else runner._runtime._iter_nodes_by_depth(),
            root_node_id=str(runner._runtime.tree.root_node.id),
            destination=output / "science",
            generation=state.generation + 1,
            producer=head["scientific_identity"]["producer"]["identity"],
            committed_source=source,
            encoding="jsonl-zstd-1",
        )

    if args.trace:
        tracemalloc.start()
    (manifest, stats), observation = measured("export", output, export)
    auxiliary = tracemalloc.get_traced_memory() if args.trace else None
    if args.trace:
        tracemalloc.stop()
    (output / "result.json").write_text(
        json.dumps(
            {
                "mode": "before" if args.before else "iterator",
                "source_digest": artifact_digest(source.parent),
                "before_source_sha256": hashlib.sha256(source_code.encode()).hexdigest()
                if source_code
                else None,
                "generation": state.generation,
                "nodes": runner.current_tree_size(),
                "auxiliary_python_retained_peak": auxiliary,
                "writer": stats,
                "observation": observation,
                "manifest": str(manifest),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

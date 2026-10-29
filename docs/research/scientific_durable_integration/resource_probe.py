"""Reuse existing measurements on one fresh source across two pinned revisions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest
from anemone.node_evaluation.common import canonical_value

from chipiron.environments.morpion.bootstrap import MorpionBootstrapArgs
from chipiron.environments.morpion.bootstrap.performance import StageMeasurement
from chipiron.environments.morpion.bootstrap.sharded_training_reader import (
    ShardedTrainingGenerationView,
)
from chipiron.environments.morpion.bootstrap.streaming_dataset import (
    IndexedDatasetSource,
    disk_dataset_context,
)
from chipiron.environments.morpion.learning import (
    save_morpion_supervised_rows_streaming,
)
from scripts.benchmark_morpion_scientific_reader import costs, delta
from tests.environments.morpion_scientific_reader_helpers import (
    generated_scientific_source,
)


def digest(path: Path) -> str:
    """Hash complete artifacts without retaining their contents."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    """Generate separately, then run each labeled measurement in a fresh process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["prepare", "cold", "dataset"], required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--label", choices=["before", "after"], default="after")
    args = parser.parse_args()
    root = args.root
    if args.mode == "prepare":
        root.mkdir(exist_ok=False)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                canonical_value,
                "make_forced_value",
                canonical_value.make_estimate_value,
            )
            generated_scientific_source(root / "source", args.size, padding=4096)
        return
    stage, counters = StageMeasurement(), costs()
    path = root / "source/generation_000001/generation.json"
    with ShardedTrainingGenerationView(
        path, cache_dir=root / f"{args.label}-cache", allow_scientific_v2=True
    ) as view:
        initialization = stage.finish(
            costs=delta(counters),
            index_reused=view.index_reused,
            nodes=view.node_count(),
        )
        dataset = None
        output_hash = None
        if args.mode == "dataset":
            stage, counters = StageMeasurement(), costs()
            extraction = MorpionBootstrapArgs(
                work_dir=root,
                dataset_confidence_policy="selected_value_v2",
                dataset_family_target_policy="pv_blend_mean_prediction",
                use_backed_up_value=False,
            )
            output = root / f"{args.label}.jsonl"
            with disk_dataset_context(
                IndexedDatasetSource(view), extraction
            ) as context:
                stream = context.stream(1)
                statistics = save_morpion_supervised_rows_streaming(
                    rows=stream.rows, metadata=stream.metadata, path=output
                )
                temporary_bytes = context.scratch_bytes()
            dataset = stage.finish(
                costs=delta(counters),
                rows=statistics.row_count,
                output_bytes=statistics.bytes_written,
                temp_bytes=temporary_bytes,
            )
            output_hash = digest(output)
        view.validate_source_identity()
        identity = hashlib.sha256(view.identity.encode()).hexdigest()
        index_bytes = view.index_path.stat().st_size
    result = {
        "label": args.label,
        "mode": args.mode,
        "size": args.size,
        "initialization": initialization,
        "dataset": dataset,
        "jsonl_sha256": output_hash,
        "source_identity_sha256": identity,
        "index_bytes": index_bytes,
        "source_sha256": {
            str(p.relative_to(root)): digest(p)
            for p in sorted((root / "source").rglob("*"))
            if p.is_file()
        },
    }
    (root / f"{args.label}-{args.mode}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()

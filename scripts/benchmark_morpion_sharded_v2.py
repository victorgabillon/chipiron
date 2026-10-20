"""Small scratch-only export benchmark; single process, no bootstrap workers.

Prepare one deterministic ~1k/~5k real search fixture, then run each format in
a separate fresh process. Stage RSS uses the existing memory observability.
Validation belongs after measurement, to avoid warming the export inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

from anemone.checkpoints.state_handles import CheckpointBackedStateHandle
from anemone.progress_monitor.progress_monitor import TreeBranchLimitArgs
from atomheart.games.morpion import checkpoints as atom_checkpoints
from atomheart.games.morpion.checkpoints import MorpionStateCheckpointCodec

from chipiron.environments.morpion.bootstrap import sharded_training_v2
from chipiron.environments.morpion.bootstrap.performance import StageMeasurement
from chipiron.environments.morpion.bootstrap.runtime.runner import (
    AnemoneMorpionSearchRunner,
    AnemoneMorpionSearchRunnerArgs,
    default_search_args,
)
from chipiron.environments.morpion.bootstrap.sharded_training_export import (
    save_morpion_sharded_training_tree_from_live_nodes,
)
from chipiron.environments.morpion.bootstrap.sharded_training_v2 import (
    save_scientific_training_generation,
    value_score,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def runner() -> AnemoneMorpionSearchRunner:
    """Construct a bounded deterministic in-process search, without a worker."""
    args = AnemoneMorpionSearchRunnerArgs(
        search_args=replace(
            default_search_args(),
            stopping_criterion=TreeBranchLimitArgs(
                type="tree_branch_limit", tree_branch_limit=6000
            ),
        ),
        random_seed=4,
    )
    return AnemoneMorpionSearchRunner(args)


def counted(
    counter: Counter[str], label: str, original: Callable[..., Any]
) -> Callable[..., Any]:
    """Count actual state requests, anchor/delta decodes and canonical dumps."""

    def call(*args: Any, **kwargs: Any) -> Any:
        counter[label] += 1
        return original(*args, **kwargs)

    return call


def main() -> None:
    """Prepare or measure synthetic fixtures; refuse existing output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", choices=("prepare", "measure"), required=True)
    parser.add_argument("--nodes", type=int, choices=(1000, 5000), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--format", choices=("flat", "v1", "v2"), default="v2")
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to(Path("/tmp")):
        parser.error("only new /tmp synthetic fixtures are allowed")
    logging.disable(logging.CRITICAL)
    instance = runner()
    if args.operation == "prepare":
        root.mkdir(parents=True, exist_ok=False)
        instance.load_or_create(None, None)
        while instance.current_tree_size() < args.nodes:
            instance.grow(1)
        instance.save_checkpoint(root / "checkpoint.json")
        (root / "fixture.json").write_text(
            json.dumps(
                {
                    "synthetic": True,
                    "seed": 4,
                    "nominal_nodes": args.nodes,
                    "actual_nodes": instance.current_tree_size(),
                },
                indent=2,
            )
            + "\n"
        )
        return
    fixture = json.loads((root / "fixture.json").read_text())
    if fixture["synthetic"] is not True or fixture["nominal_nodes"] != args.nodes:
        parser.error("fixture identity mismatch")
    output = root / args.format
    output.mkdir(exist_ok=False)
    instance.load_or_create(root / "checkpoint.json", None)
    counter: Counter[str] = Counter()
    results = []
    with ExitStack() as stack:
        for cls, name, label in (
            (CheckpointBackedStateHandle, "get", "state_handle_get"),
            (
                MorpionStateCheckpointCodec,
                "load_anchor_ref",
                "checkpoint_anchor_decode",
            ),
            (
                MorpionStateCheckpointCodec,
                "load_child_from_delta",
                "checkpoint_delta_decode",
            ),
            (MorpionStateCheckpointCodec, "dump_state_ref", "canonical_dump"),
        ):
            stack.enter_context(
                patch.object(cls, name, counted(counter, label, getattr(cls, name)))
            )
        stack.enter_context(
            patch.object(
                atom_checkpoints,
                "_apply_move",
                counted(counter, "codec_replay_steps", atom_checkpoints._apply_move),
            )
        )
        for generation in (1, 2, 3):
            if generation == 3:
                instance.grow(1)
            nodes = tuple(instance.iter_profile_nodes())
            counter.clear()
            measurement = StageMeasurement()
            stats = None
            if args.format == "flat":
                instance.export_training_tree_snapshot(
                    output / f"generation-{generation}.json"
                )
            elif args.format == "v1":
                _, stats = save_morpion_sharded_training_tree_from_live_nodes(
                    nodes=nodes,
                    root_node_id="0",
                    output_dir=output,
                    generation=generation,
                    state_ref_dumper=instance.profile_state_codec().dump_state_ref,
                    direct_value_extractor=value_score,
                    backed_up_value_extractor=value_score,
                )
            else:
                _, stats = save_scientific_training_generation(
                    nodes=nodes,
                    root_node_id="0",
                    output_dir=output,
                    generation=generation,
                    producer={
                        "revision": "67a7cfd1955f43d5947af01dbb324fb5efd93a0d+"
                        + hashlib.sha256(
                            Path(sharded_training_v2.__file__).read_bytes()
                        ).hexdigest(),
                        "evaluator": "default-heuristic",
                        "config_sha256": hashlib.sha256(
                            b"default_search_args;tree_branch_limit=6000;random_seed=4"
                        ).hexdigest(),
                    },
                )
            results.append(
                measurement.finish(
                    format=args.format,
                    generation=generation,
                    node_count=len(nodes),
                    operations=dict(counter),
                    writer_stats=None if stats is None else asdict(stats),
                )
            )
    result = {
        "fixture": fixture,
        "stages": results,
        "export_bytes": sum(
            path.stat().st_size for path in output.rglob("*") if path.is_file()
        ),
    }
    (root / f"{args.format}-measurements.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()

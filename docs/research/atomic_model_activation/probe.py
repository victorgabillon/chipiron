"""Checkpoint prerequisite probe; no model publication or authority mutation."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import resource
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from anemone.checkpoints.rng_state import rollout_random_generator

from chipiron.environments.morpion.bootstrap import (
    MorpionBootstrapEffectiveRuntimeConfig,
)
from chipiron.environments.morpion.bootstrap.config import MorpionBootstrapRolloutConfig
from chipiron.environments.morpion.bootstrap.runtime.runner import (
    AnemoneMorpionSearchRunner,
    AnemoneMorpionSearchRunnerArgs,
    default_search_args,
)
from docs.research.model_activation.probe import nodes, state
from tests.environments.test_morpion_checkpoint_parity import _selection_signature
from tests.environments.test_morpion_committed_evaluator_restore import (
    fingerprint,
    link_iteration_digest,
)
from tests.environments.test_morpion_value_certainty import neural_evaluator


def ordered(instance: Any) -> dict[int, dict[str, Any]]:
    """Observe actual link iteration, without sorting away the discrepancy."""
    return {
        n.id: {
            "parents": [
                (p.id, sorted(map(str, branches)))
                for p, branches in n.parent_nodes.items()
            ],
            "children": [(str(b), c.id) for b, c in n.iter_child_links()],
        }
        for n in nodes(instance)
    }


def compare(left: Any, right: Any) -> dict[str, Any]:
    """Separate unordered scientific equality from exact runtime order."""
    a, b = ordered(left), ordered(right)
    av, bv = state(left), state(right)
    lr, rr = (
        rollout_random_generator(left._runtime),
        rollout_random_generator(right._runtime),
    )
    return {
        "nodes": [left.current_tree_size(), right.current_tree_size()],
        "scientific_hashes": [fingerprint(left), fingerprint(right)],
        "ordered_hashes": [link_iteration_digest(left), link_iteration_digest(right)],
        "order_differences": [
            dict(node_id=i, before=a[i], after=b[i])
            for i in a.keys() & b.keys()
            if a[i] != b[i]
        ],
        "value_differences": [
            dict(node_id=i, fields=[k for k in av[i] if av[i][k] != bv[i][k]])
            for i in av.keys() & bv.keys()
            if av[i] != bv[i]
        ],
        "search_rng_equal": left._random_generator.getstate()
        == right._random_generator.getstate(),
        "rollout_rng_equal": (None if lr is None else lr.getstate())
        == (None if rr is None else rr.getstate()),
    }


def main() -> None:
    """Generate a fresh legal graph and compare uninterrupted/restored continuation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--format", choices=("sharded", "json-zst"), default="sharded")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--rollout", action="store_true")
    options = parser.parse_args()
    options.root.mkdir(parents=True, exist_ok=False)
    logging.disable(logging.CRITICAL)
    started = time.perf_counter()
    args = AnemoneMorpionSearchRunnerArgs(
        random_seed=4,
        runtime_checkpoint_format=options.format,
        search_args=default_search_args(
            rollout=MorpionBootstrapRolloutConfig(
                enabled=options.rollout, max_extra_steps=1
            )
        ),
    )
    config = MorpionBootstrapEffectiveRuntimeConfig(tree_branch_limit=100000)
    live = AnemoneMorpionSearchRunner(args)
    live.load_or_create(None, None, config)
    while live.current_tree_size() < options.count:
        live.grow(1)
        assert live.last_growth_steps > 0
    checkpoint = options.root / (
        "checkpoint" if options.format == "sharded" else "checkpoint.json.zst"
    )
    live.save_checkpoint(checkpoint)
    restored = AnemoneMorpionSearchRunner(replace(args, random_seed=999))
    restored.load_or_create(checkpoint, None, config)
    result = {
        "options": vars(options) | {"root": str(options.root)},
        "initial": compare(live, restored),
        "steps": [],
    }
    if options.refresh:
        for instance in (live, restored):
            instance._runtime.refresh_with_evaluator(
                neural_evaluator(42), scope="all", batch_size=64
            )
        result["after_refresh"] = compare(live, restored)
    for step in range(1, options.steps + 1):
        live.grow(1)
        restored.grow(1)
        item = compare(live, restored)
        item["step"] = step
        item["selector_equal"] = _selection_signature(live) == _selection_signature(
            restored
        )
        item["selected_nodes"] = [
            i._runtime.node_selector.base.latest_selection_report.selected_node_id
            for i in (live, restored)
        ]
        # Full initial discrepancy is retained; subsequent order summaries avoid
        # a cumulative copy of the same diagnostic arrays.
        item["order_difference_count"] = len(item.pop("order_differences"))
        result["steps"].append(item)
        if (
            item["scientific_hashes"][0] != item["scientific_hashes"][1]
            or not item["selector_equal"]
            or not item["search_rng_equal"]
            or not item["rollout_rng_equal"]
        ):
            break
    result["wall_seconds"] = time.perf_counter() - started
    result["max_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    result["source_hashes"] = {
        str(p.relative_to(options.root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in options.root.rglob("*")
        if p.is_file()
    }
    (options.root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "initial_order_differences": len(
                    result["initial"]["order_differences"]
                ),
                "last": result["steps"][-1],
                "wall_seconds": result["wall_seconds"],
                "max_rss_mib": result["max_rss_mib"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

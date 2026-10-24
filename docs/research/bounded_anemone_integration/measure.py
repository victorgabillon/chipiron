"""Adapt the existing serial profiler; matched real neural feature workloads."""

from __future__ import annotations

import hashlib
import json
import resource
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

from anemone.checkpoints import CheckpointStateResolver
from atomheart.games.morpion.checkpoints import MorpionStateCheckpointCodec

from docs.research.model_activation import measure as prior
from docs.research.model_activation.probe import nodes
from tests.environments.test_morpion_checkpoint_parity import _selection_signature


def main() -> None:
    """Keep the existing fixture, acceptance validation, cProfile, I/O and RSS tools."""
    stats = dict(
        hits=0, misses=0, anchors=0, deltas=0, canonical_loads=0, max_retained=0
    )
    phases = []
    live = []
    original = CheckpointStateResolver.resolve

    def resolve(resolver: Any, node_id: int) -> Any:
        stats["hits" if node_id in resolver._resolved_states else "misses"] += 1
        result = original(resolver, node_id)
        stats["max_retained"] = max(
            stats["max_retained"], len(resolver._resolved_states)
        )
        return result

    def operation(instance: Any, evaluator: Any, batch_size: int) -> dict:
        live.append(instance)
        resolver = next(nodes(instance)).tree_node.state_handle.resolver
        for _ in range(2):
            before = dict(stats)
            retained_before = len(resolver._resolved_states)
            usage = resource.getrusage(resource.RUSAGE_SELF)
            start = time.perf_counter()
            report = instance._runtime.refresh_with_evaluator(
                evaluator, scope="all", batch_size=batch_size
            )
            after = resource.getrusage(resource.RUSAGE_SELF)
            decoded = (
                stats["anchors"]
                + stats["deltas"]
                - before["anchors"]
                - before["deltas"]
            )
            retained = len(resolver._resolved_states)
            phases.append(
                dict(
                    wall_seconds=time.perf_counter() - start,
                    cpu_seconds=after.ru_utime
                    + after.ru_stime
                    - usage.ru_utime
                    - usage.ru_stime,
                    peak_rss_mib=after.ru_maxrss / 1024,
                    retained=retained,
                    cache_hits=stats["hits"] - before["hits"],
                    cache_misses=stats["misses"] - before["misses"],
                    anchor_decodes=stats["anchors"] - before["anchors"],
                    delta_decodes=stats["deltas"] - before["deltas"],
                    canonical_loads=stats["canonical_loads"]
                    - before["canonical_loads"],
                    evictions=decoded - retained + retained_before,
                    max_retained=stats["max_retained"],
                    evaluations=report.reevaluated_count,
                )
            )
        return dict(
            evaluations=report.reevaluated_count,
            requested=report.requested_count,
            skipped_terminal=report.skipped_terminal_count,
            skipped_certified=report.skipped_certified_count,
            batch_size=batch_size,
        )

    prior.direct_phase = operation
    prior.backup_phase = lambda _: 0
    prior.selector_phase = lambda _: None
    with ExitStack() as stack:
        stack.enter_context(patch.object(CheckpointStateResolver, "resolve", resolve))
        for method, counter in [
            ("load_anchor_ref", "anchors"),
            ("load_child_from_delta", "deltas"),
            ("load_state_ref", "canonical_loads"),
        ]:
            actual = getattr(MorpionStateCheckpointCodec, method)

            def counted(
                *args: Any,
                _actual: Any = actual,
                _counter: str = counter,
                **kwargs: Any,
            ) -> Any:
                stats[_counter] += 1
                return _actual(*args, **kwargs)

            stack.enter_context(
                patch.object(MorpionStateCheckpointCodec, method, counted)
            )
        prior.main()
    if live:
        current = live[0]
        current.grow(2)
        continuation = dict(
            scientific_sha256=prior.fingerprint(current),
            selection=repr(_selection_signature(current)),
            rng_sha256=hashlib.sha256(
                repr(current._random_generator.getstate()).encode()
            ).hexdigest(),
        )
    else:
        continuation = None
    root, mode = Path(sys.argv[1]), sys.argv[2]
    output = root / f"{mode}.json"
    result = json.loads(output.read_text())
    result.update(
        cache_phases=phases,
        total_cache_stats=stats,
        subsequent_growth=continuation,
        peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    )
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

"""Reuse the previous measurement driver on matched freshly generated checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from docs.research.model_activation import measure as previous
from docs.research.model_activation.probe import nodes


def main() -> None:
    """Measure either the previous kernel or the corrected public dependency API."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--engine", choices=("prior_kernel", "anemone"), required=True)
    options, remaining = parser.parse_known_args()
    prior_direct = previous.direct_phase

    def operation(instance: Any, evaluator: Any, batch_size: int) -> dict:
        resolver = next(nodes(instance)).tree_node.state_handle.resolver
        before = len(resolver._resolved_states)
        if options.engine == "anemone":
            report = instance._runtime.refresh_with_evaluator(
                evaluator, scope="all", batch_size=batch_size
            )
            result = dict(
                evaluations=report.reevaluated_count,
                requested=report.requested_count,
                skipped_terminal=report.skipped_terminal_count,
                skipped_certified=report.skipped_certified_count,
                batch_size=batch_size,
            )
        else:
            result = prior_direct(instance, evaluator, batch_size)
        result.update(
            decoded_cache_before=before,
            decoded_cache_after=len(resolver._resolved_states),
            engine=options.engine,
        )
        return result

    previous.direct_phase = operation
    if options.engine == "anemone":
        # These driver slots are unused: Anemone already performed both operations.
        previous.backup_phase = lambda _: 0
        previous.selector_phase = lambda _: None
    sys.argv = [sys.argv[0], *remaining]
    previous.main()
    root, mode = Path(remaining[0]), remaining[1]
    path = root / f"{mode}.json"
    result = json.loads(path.read_text())
    result["engine"] = options.engine
    if mode == "refresh" and options.engine == "anemone":
        result["complete_refresh"] = result.pop("inference")
        del result["backup"], result["selector"], result["backed_up_nodes"]
        result["profile_note"] = (
            "inference.prof contains the ENTIRE Anemone refresh: preflight, direct "
            "windows, backup and selector. Separate times come from cProfile; "
            "no additional backup/selector algorithm was executed."
        )
    path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

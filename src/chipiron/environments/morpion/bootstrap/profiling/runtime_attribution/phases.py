"""Observe existing restore callbacks and narrow function boundaries in this process."""

from __future__ import annotations

import json
import time
import traceback
from contextlib import ExitStack, contextmanager
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

from chipiron.environments.morpion.bootstrap.performance import process_peak_rss_mb
from chipiron.environments.morpion.bootstrap.pipeline_memory import (
    available_ram_mb,
    current_rss_mb,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping
    from pathlib import Path


class PhaseRecorder:
    """Persist scalar-only events immediately; never retain checkpoint/runtime objects."""

    def __init__(self, output: Path) -> None:
        """Start a bounded event stream in an already validated external directory."""
        self.output = output
        self.started = time.perf_counter()
        self.events: list[dict[str, Any]] = []
        self.decode_counts = {"anchor": 0, "delta": 0}
        self.first_decode_stacks: list[dict[str, Any]] = []

    def log(self, phase: str, **metadata: object) -> None:
        """Implement the existing restore logger interface without recursive profiling."""
        self.callback(phase, metadata)

    def callback(self, phase: str, metadata: Mapping[str, object]) -> None:
        """Record memory and elapsed time at a known restore boundary."""
        event = {
            "phase": phase,
            "elapsed_s": time.perf_counter() - self.started,
            "rss_mib": current_rss_mb(),
            "process_lifetime_peak_rss_mib": process_peak_rss_mb(),
            "available_ram_mib": available_ram_mb(),
            "state_decodes": dict(self.decode_counts),
            "metadata": {
                k: v
                for k, v in metadata.items()
                if v is None or isinstance(v, str | int | float | bool)
            },
        }
        # Current restore phases are O(shards), not O(nodes). Bound unusual inputs too.
        if len(self.events) >= 4096:
            message = "Restore diagnostic event cap exceeded."
            raise RuntimeError(message)
        self.events.append(event)
        with self.output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")
        print(
            f"[restore-memory] {phase} elapsed={event['elapsed_s']:.2f}s rss={event['rss_mib']}MiB available={event['available_ram_mib']}MiB decodes={sum(self.decode_counts.values())}",
            flush=True,
        )


def _after(
    function: Callable[..., Any], recorder: PhaseRecorder, phase: str
) -> Callable[..., Any]:
    def observed(*args: Any, **kwargs: Any) -> Any:
        result = function(*args, **kwargs)
        recorder.log(phase)
        return result

    return observed


@contextmanager
def observe_restore(recorder: PhaseRecorder) -> Iterator[None]:
    """Scope extra observation to this single-threaded diagnostic invocation.

    Released Anemone already provides coarse phase callbacks. Temporary wrappers
    fill the missing boundaries, forwarding every argument and return unchanged.
    They are removed even on interruption; no Anemone source or package is edited.
    """
    from anemone.checkpoints import (
        node_restore,
        selector_payloads,
        sharded_restore,
        state_handles,
        state_handles_restore,
        tree_expansions_payloads,
    )

    from chipiron.environments.morpion.bootstrap.runtime import runner

    boundaries = (
        (sharded_restore, "_build_sharded_payload_store", "after_payload_store_build"),
        (
            state_handles_restore,
            "_create_state_handles_from_node_ids",
            "after_state_handle_creation",
        ),
        (node_restore, "_create_nodes_from_node_shells", "after_live_node_creation"),
        (node_restore, "_build_tree_from_node_shells", "after_tree_bookkeeping"),
        (
            tree_expansions_payloads,
            "_restore_tree_expansions_runtime_state",
            "after_latest_expansions_restoration",
        ),
        (
            selector_payloads,
            "_restore_explicit_selector_state",
            "after_explicit_selector_restoration",
        ),
    )
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                runner,
                "restore_memory_logger_for_checkpoint_path",
                lambda *a, **kw: recorder,
            )
        )
        for module, name, phase in boundaries:
            stack.enter_context(
                patch.object(
                    module, name, _after(getattr(module, name), recorder, phase)
                )
            )
        read_name = "_read_jsonl_shard_records"
        original_read = getattr(sharded_restore, read_name)

        def read_records(*args: Any, **kwargs: Any) -> Any:
            records = original_read(*args, **kwargs)
            shard = args[1]
            recorder.log(f"after_{shard.kind}_shard_load", records=len(records))
            return records

        stack.enter_context(
            patch.object(sharded_restore, "_read_jsonl_shard_records", read_records)
        )
        for name, kind in (("_resolve_anchor", "anchor"), ("_resolve_delta", "delta")):
            original = getattr(state_handles.CheckpointStateResolver, name)

            def counted(
                *args: Any,
                _original: Callable[..., Any] = original,
                _kind: str = kind,
                **kwargs: Any,
            ) -> Any:
                recorder.decode_counts[_kind] += 1
                total_decodes = sum(recorder.decode_counts.values())
                if total_decodes == 1 or total_decodes % 2000 == 0:
                    recorder.log("state_decode_progress", total_decodes=total_decodes)
                if recorder.decode_counts[_kind] <= 3:
                    recorder.first_decode_stacks.append({
                        "kind": _kind,
                        "stack": [
                            {
                                "file": frame.filename,
                                "line": frame.lineno,
                                "function": frame.name,
                            }
                            for frame in traceback.extract_stack(limit=16)
                        ],
                    })
                return _original(*args, **kwargs)

            stack.enter_context(
                patch.object(state_handles.CheckpointStateResolver, name, counted)
            )
        yield

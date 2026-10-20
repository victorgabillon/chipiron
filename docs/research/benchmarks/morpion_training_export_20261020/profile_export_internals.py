"""Intrusive cProfile/tracemalloc pass; excluded from throughput comparisons."""

import importlib.util
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any
from unittest.mock import patch

from chipiron.environments.morpion.bootstrap import sharded_training_export as writer

spec = importlib.util.spec_from_file_location(
    "export_driver", Path(__file__).with_name("profile_morpion_export.py")
)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
original = driver.ProfileRunner.export_call
write = writer._write_json
samples = []


def observed_write(path: Any, payload: Any) -> Any:
    """Sample held rows and traced memory at existing JSON writer boundaries."""
    caller = sys._getframe(1)
    counts = {
        name: len(caller.f_locals.get(name, []))
        for name in ["node_records", "node_updates"]
    }
    del caller
    before = tracemalloc.get_traced_memory()
    cpu = time.process_time()
    wall = time.perf_counter()
    result = write(path, payload)
    samples.append(
        dict(
            path=str(path),
            before=before,
            after=tracemalloc.get_traced_memory(),
            held_dataclass_rows=counts,
            payload_rows=sum(len(payload.get(k, [])) for k in ["nodes", "updates"]),
            cpu_s=time.process_time() - cpu,
            wall_s=time.perf_counter() - wall,
        )
    )
    return result


def observed_export(self: Any, kind: Any, function: Any) -> Any:
    """Add intrusive allocation samples to a diagnostic export only."""
    tracemalloc.start()
    start = len(samples)
    try:
        result = original(self, kind, function)
        (
            self.exports[-1]["traced_current_bytes"],
            self.exports[-1]["traced_peak_bytes"],
        ) = tracemalloc.get_traced_memory()
        self.exports[-1]["writer_memory_snapshots"] = samples[start:]
        return result
    finally:
        tracemalloc.stop()


with (
    patch.object(writer, "_write_json", observed_write),
    patch.object(driver.ProfileRunner, "export_call", observed_export),
):
    driver.main()

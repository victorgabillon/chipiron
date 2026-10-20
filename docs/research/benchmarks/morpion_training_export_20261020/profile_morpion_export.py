"""Bounded fresh-process synthetic export study using existing Chipiron profiles."""

from __future__ import annotations

import argparse
import cProfile
import gc
import hashlib
import importlib.util
import json
import logging
import os
import shutil
import time
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

import anemone
from anemone.checkpoints import prepare_checkpoint_adoption

from chipiron.environments.morpion.bootstrap.performance import StageMeasurement
from chipiron.environments.morpion.bootstrap.runtime import runner as runner_module
from chipiron.environments.morpion.bootstrap.runtime.training_export_profile import (
    training_export_profile_to_dict,
)

SOURCE = Path(
    os.environ.get("CHIPIRON_EXPORT_SOURCE", "/tmp/chipiron-export-profile-integration")
)
spec = importlib.util.spec_from_file_location(
    "existing_benchmark", SOURCE / "scripts/benchmark_morpion_warm_growth.py"
)
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)
CURRENT = None
OPTIONS = None


def digest(value: Any) -> Any:
    """Hash the complete normalized JSON structure without removing fields."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class ProfileRunner(harness.BenchmarkRunner):
    """Observe the existing benchmark runner without changing search/export data."""

    def __init__(self, *, cold: Any) -> None:
        """Attach phase-scoped codec counters and GC observations."""
        super().__init__(cold=cold)
        global CURRENT
        CURRENT = self
        self.pending = None
        self.exports = []
        self.events = []
        self.phase = "other"
        self.decodes = Counter()
        self.stack = ExitStack()
        self.gc_started = {}
        self.gc_times = Counter()
        self.gc_counts = Counter()

        def callback(phase: Any, info: Any) -> Any:
            """Attribute GC wall time to the current observed phase."""
            generation = info["generation"]
            if phase == "start":
                self.gc_started[generation] = (time.perf_counter(), self.phase)
            else:
                started, label = self.gc_started.pop(
                    generation, (time.perf_counter(), self.phase)
                )
                self.gc_times[label] += time.perf_counter() - started
                self.gc_counts[label] += 1

        gc.callbacks.append(callback)
        self.stack.callback(lambda: gc.callbacks.remove(callback))
        cls = type(self._state_codec)
        for name in ["load_anchor_ref", "load_child_from_delta"]:
            original = getattr(cls, name)

            def counted(
                *args: Any, _name: Any = name, _original: Any = original, **kwargs: Any
            ) -> Any:
                """Count actual codec decode calls before invoking the original codec."""
                self.decodes[(self.phase, _name)] += 1
                return _original(*args, **kwargs)

            self.stack.enter_context(patch.object(cls, name, counted))

    def save_checkpoint(self, path: Any) -> Any:
        """Prepare native adoption while the ordinary saved payload is available."""
        self.phase = "checkpoint"
        self.pending = None
        build = runner_module.build_search_checkpoint_payload

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            """Keep a compact adoption plan without retaining the full typed payload."""
            payload = build(*args, **kwargs)
            self.events.append(dict(event="payload_built", time=time.perf_counter()))
            if not self.cold and OPTIONS.preparation == "before":
                self.pending = prepare_checkpoint_adoption(
                    self.profile_runtime_root(), payload, state_codec=self._state_codec
                )
                self.events.append(
                    dict(event="adoption_prepared", time=time.perf_counter())
                )
            return payload

        with patch.object(runner_module, "build_search_checkpoint_payload", wrapped):
            super().save_checkpoint(path)
        self.events.append(
            dict(event="checkpoint_returned_payload_released", time=time.perf_counter())
        )
        self.phase = "other"

    def acknowledge_growth_publication(self, **kwargs: Any) -> Any:
        """Adopt only after the existing publisher acknowledges a committed save."""
        super().acknowledge_growth_publication(**kwargs)
        self.events.append(
            dict(event="publication_acknowledged", time=time.perf_counter())
        )
        if (
            not self.cold
            and kwargs["generation_published"]
            and self.pending is not None
        ):
            self.phase = "adopt"
            self.pending.adopt()
            self.pending = None
            self._reset_growth_state_eviction_runtime()
            self.events.append(
                dict(event="adoption_committed_plan_released", time=time.perf_counter())
            )
            self.phase = "other"

    def build_training_tree_snapshot_payload(self) -> Any:
        """Capture the existing flat-export profile and preserve the snapshot."""
        snapshot, profile = super().build_training_tree_snapshot_payload()
        self.flat_profile = training_export_profile_to_dict(profile)
        return snapshot, profile

    def export_call(self, kind: Any, function: Any) -> Any:
        """Observe existing export profiles, process CPU, GC and codec counts."""
        self.phase = "export"
        before = dict(self.decodes)
        gc_time = self.gc_times["export"]
        gc_count = self.gc_counts["export"]
        observation = StageMeasurement()
        cpu = time.process_time()
        self.events.append(
            dict(
                event="export_started",
                time=time.perf_counter(),
                pending_plan=self.pending is not None,
            )
        )
        profiler = cProfile.Profile() if OPTIONS.cprofile else None
        if profiler:
            profiler.enable()
        was_enabled = gc.isenabled()
        if OPTIONS.disable_gc:
            gc.disable()
        try:
            result = function()
        finally:
            if OPTIONS.disable_gc and was_enabled:
                gc.enable()
            if profiler:
                profiler.disable()
                profiler.dump_stats(
                    OPTIONS.work / f"export-{len(self.exports) + 1}.prof"
                )
        record = observation.finish(
            export_mode=kind,
            cpu_s=time.process_time() - cpu,
            gc_s=self.gc_times["export"] - gc_time,
            gc_collections=self.gc_counts["export"] - gc_count,
            decodes={
                name: self.decodes[("export", name)] - before.get(("export", name), 0)
                for name in ["load_anchor_ref", "load_child_from_delta"]
            },
        )
        if kind == "sharded":
            record.update(
                components=self._latest_training_export_stats,
                profile=self._latest_training_export_profile,
            )
        else:
            record["profile"] = self.flat_profile
        self.exports.append(record)
        self.events.append(dict(event="export_returned", time=time.perf_counter()))
        self.phase = "other"
        return result

    def export_training_tree_snapshot(self, path: Any) -> Any:
        """Measure the existing flat writer without replacing it."""
        return self.export_call(
            "flat",
            lambda: super(ProfileRunner, self).export_training_tree_snapshot(path),
        )

    def export_sharded_training_tree_snapshot(
        self, path: Any, *, generation: Any
    ) -> Any:
        """Measure the existing sharded writer without replacing it."""
        return self.export_call(
            "sharded",
            lambda: super(ProfileRunner, self).export_sharded_training_tree_snapshot(
                path, generation=generation
            ),
        )


harness.BenchmarkRunner = ProfileRunner


def main() -> None:
    """Validate temporary synthetic inputs and run a single fresh-process case."""
    global OPTIONS
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--work", type=Path)
    p.add_argument("--action", choices=["prepare", "measure"], required=True)
    p.add_argument("--nodes", type=int, default=1000)
    p.add_argument("--mode", choices=["cold", "warm"], default="cold")
    p.add_argument("--export-mode", choices=["flat", "sharded"], default="sharded")
    p.add_argument("--cycles", type=int, default=3)
    p.add_argument("--preparation", choices=["before", "skip"], default="before")
    p.add_argument("--cprofile", action="store_true")
    p.add_argument("--disable-gc", action="store_true")
    OPTIONS = p.parse_args()
    assert OPTIONS.root.resolve().is_relative_to(Path("/tmp"))
    assert 1 <= OPTIONS.nodes <= 10000 and 1 <= OPTIONS.cycles <= 3
    assert OPTIONS.preparation != "skip" or OPTIONS.cycles == 1
    OPTIONS.root.mkdir(parents=True, exist_ok=True)
    logging.disable(logging.CRITICAL)
    original_args = harness.MorpionBootstrapArgs

    def args(**kwargs: Any) -> Any:
        """Select the requested existing export mode explicitly."""
        kwargs["training_export_mode"] = OPTIONS.export_mode
        return original_args(**kwargs)

    harness.MorpionBootstrapArgs = args
    if OPTIONS.action == "prepare":
        assert not (OPTIONS.root / "prepare").exists()
        assert not (OPTIONS.root / "prepare.json").exists()
        result = harness.prepare(OPTIONS.root / "prepare", OPTIONS.nodes)
        output = OPTIONS.root / "prepare.json"
    else:
        assert OPTIONS.work is not None and not OPTIONS.work.exists()
        assert not OPTIONS.work.with_suffix(".json").exists()
        assert OPTIONS.work.resolve().is_relative_to(OPTIONS.root.resolve())
        marker = OPTIONS.root / "prepare/pipeline/generation_000001/manifest.json"
        assert json.loads(marker.read_text())["created_at_utc"] == "synthetic-benchmark"
        shutil.copytree(OPTIONS.root / "prepare", OPTIONS.work)
        result = harness.measure(
            OPTIONS.work,
            cold=OPTIONS.mode == "cold",
            cycles=OPTIONS.cycles,
            skip_export=False,
        )
        result["exports"] = CURRENT.exports
        result["events"] = CURRENT.events
        result["decodes_by_phase"] = {
            f"{k[0]}:{k[1]}": v for k, v in CURRENT.decodes.items()
        }
        result["export_files"] = {}
        export_root = OPTIONS.work / (
            "tree_exports_sharded"
            if OPTIONS.export_mode == "sharded"
            else "tree_exports"
        )
        for path in sorted(export_root.rglob("*.json")):
            result["export_files"][str(path.relative_to(export_root))] = digest(
                json.loads(path.read_text())
            )
        output = OPTIONS.work.with_suffix(".json")
    result.update(
        export_mode=OPTIONS.export_mode,
        mode=OPTIONS.mode,
        cycles=OPTIONS.cycles,
        preparation=OPTIONS.preparation,
        disable_gc=OPTIONS.disable_gc,
        anemone_sha=harness.git(
            Path(anemone.__file__).resolve().parents[2], "rev-parse", "HEAD"
        ),
        anemone_import=anemone.__file__,
        chipiron_sha=harness.git(SOURCE, "rev-parse", "HEAD"),
        chipiron_import=runner_module.__file__,
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    output.write_text(json.dumps(result, indent=2))
    CURRENT.stack.close()
    print(
        json.dumps(
            dict(
                output=str(output),
                exports=[
                    {
                        "wall": e["elapsed_s"],
                        "cpu": e["cpu_s"],
                        "gc": e["gc_s"],
                        "decodes": e["decodes"],
                        "profile": e["profile"],
                        "components": e.get("components"),
                    }
                    for e in CURRENT.exports
                ],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""Small, CPU-only warm/cold Growth benchmark; creates only temporary fixtures.

Run with the isolated test Python and --anemone-source pointing at clean PR #77.
Each preparation and measured mode gets a fresh process. No existing run input
is accepted. Full decompressed checkpoint contents must match exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import anemone
import torch
import zstandard

from chipiron.environments.morpion.bootstrap import (
    AnemoneMorpionSearchRunner,
    AnemoneMorpionSearchRunnerArgs,
    MorpionBootstrapArgs,
    MorpionBootstrapEffectiveRuntimeConfig,
    MorpionBootstrapPaths,
)
from chipiron.environments.morpion.bootstrap.performance import StageMeasurement
from chipiron.environments.morpion.bootstrap.pipeline.stages import (
    run_pipeline_dataset_stage,
    run_pipeline_growth_stage,
    run_pipeline_training_stage,
)
from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
    MorpionPipelineGenerationManifest,
    save_pipeline_manifest,
)
from chipiron.environments.morpion.bootstrap.reevaluation_worker import (
    run_morpion_reevaluation_worker_once,
)
from chipiron.environments.morpion.bootstrap.run_state import (
    initialize_bootstrap_run_state,
    load_bootstrap_run_state,
    save_bootstrap_run_state,
)

PR77_SHA = "ec9fec7e47ca661eb1dd9330d115d010d9e5e64d"


def git(directory: Path, *arguments: str) -> str:
    """Read exact local source identity."""
    return subprocess.check_output(
        ["git", "-C", str(directory), *arguments], text=True
    ).strip()


def checkpoint_digest(root: Path) -> str:
    """Hash all decompressed scientific fields; normalize only JSON formatting."""
    records: dict[str, Any] = {}
    for shard in json.loads((root / "manifest.json").read_text())["shards"]:
        with zstandard.open(root / shard["path"], "rt") as stream:
            if shard["encoding"] == "jsonl_zst":
                records.setdefault(shard["kind"], []).extend(
                    json.loads(line) for line in stream
                )
            else:
                records[shard["kind"]] = json.load(stream)
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


class BenchmarkRunner(AnemoneMorpionSearchRunner):
    """Count real restores and record actual selected nodes."""

    def __init__(self, *, cold: bool) -> None:
        """Use identical search settings in both modes."""
        super().__init__(
            AnemoneMorpionSearchRunnerArgs(runtime_checkpoint_format="sharded")
        )
        self.cold = cold
        self.restores = 0
        self.runtime_ids: list[int] = []
        self.selected: list[object] = []

    def _load_runtime_from_checkpoint(self, *args: Any, **kwargs: Any) -> object:
        self.restores += 1
        return super()._load_runtime_from_checkpoint(*args, **kwargs)

    def load_growth_runtime(self, *args: Any, **kwargs: Any) -> None:
        """Force the reference cold path without replacing any search operation."""
        if self.cold:
            self._warm_growth.invalidate()
        super().load_growth_runtime(*args, **kwargs)
        runtime = self.profile_runtime_root()
        self.runtime_ids.append(id(runtime))
        if getattr(runtime, "_benchmark_recording", False):
            return
        original = runtime.step

        def step() -> object:
            report = original()
            self.selected.append((report.selected_node_id, report.selected_depth))
            return report

        runtime.step = step
        runtime._benchmark_recording = True


def prepare(work: Path, nodes: int) -> dict[str, object]:
    """Generate a bounded real tree, never read an existing scientific workspace."""
    runner = BenchmarkRunner(cold=False)
    runner.load_or_create(
        None, None, MorpionBootstrapEffectiveRuntimeConfig(tree_branch_limit=100000)
    )
    while runner.current_tree_size() < nodes:
        runner.grow(1)
        if runner.last_growth_steps == 0:
            message = "Synthetic generation stopped before the requested node count"
            raise RuntimeError(message)
    paths = MorpionBootstrapPaths.from_work_dir(work)
    paths.ensure_directories()
    output = paths.sharded_runtime_checkpoint_path_for_generation(1)
    runner.save_checkpoint(output)
    state = replace(
        initialize_bootstrap_run_state(),
        generation=1,
        latest_runtime_checkpoint_path=paths.relative_to_work_dir(output),
        tree_size_at_last_save=runner.current_tree_size(),
        last_save_unix_s=time.time(),
    )
    save_bootstrap_run_state(state, paths.run_state_path)
    save_pipeline_manifest(
        MorpionPipelineGenerationManifest(
            generation=1,
            created_at_utc="synthetic-benchmark",
            runtime_checkpoint_path=state.latest_runtime_checkpoint_path,
        ),
        paths.pipeline_manifest_path_for_generation(1),
    )
    return {"initial_nodes": runner.current_tree_size()}


def measure(work: Path, *, cold: bool, cycles: int) -> dict[str, Any]:
    """Measure real pipeline saves/exports using existing boundary observations."""
    runner = BenchmarkRunner(cold=cold)
    args = MorpionBootstrapArgs(
        work_dir=work,
        pipeline_mode="artifact_pipeline",
        runtime_checkpoint_format="sharded",
        training_export_mode="flat",
        tree_branch_limit=100000,
        max_growth_steps_per_cycle=2,
        save_after_tree_growth_factor=1.0,
        save_after_seconds=0.0,
    )
    observation = StageMeasurement()
    state = run_pipeline_growth_stage(args, runner, max_cycles=cycles)
    result = observation.finish(
        mode="cold" if cold else "warm",
        restores=runner.restores,
        same_runtime=len(set(runner.runtime_ids)) == 1,
        nodes=runner.current_tree_size(),
        branches=runner.current_tree_branch_count(),
        selected=runner.selected,
        eviction=runner.profile_state_eviction_runtime(),
    )
    result["checkpoint_digest"] = checkpoint_digest(
        work / state.latest_runtime_checkpoint_path
    )
    result["observations"] = [
        json.loads(p.read_text())
        for p in sorted(work.glob("pipeline/performance/*/*.json"))
    ]
    return result


def auxiliary(work: Path) -> dict[str, Any]:
    """Measure small CPU stage functions, without launching worker daemons."""
    torch.set_num_threads(1)
    args = MorpionBootstrapArgs(
        work_dir=work,
        pipeline_mode="artifact_pipeline",
        training_device="cpu",
        training_export_mode="flat",
        num_epochs=1,
        batch_size=64,
        skip_evaluator_diagnostics=True,
        max_rows=1024,
        training_max_rows=1024,
    )
    paths = MorpionBootstrapPaths.from_work_dir(work)
    generation = load_bootstrap_run_state(paths.run_state_path).generation
    started = time.perf_counter()
    manifest = run_pipeline_dataset_stage(args, generation=generation)
    dataset_s = time.perf_counter() - started
    started = time.perf_counter()
    trained = run_pipeline_training_stage(args, generation=generation)
    training_s = time.perf_counter() - started
    started = time.perf_counter()
    reevaluation = run_morpion_reevaluation_worker_once(args, max_nodes_per_patch=64)
    reevaluation_s = time.perf_counter() - started
    observations = [
        json.loads(p.read_text())
        for p in sorted(work.glob("pipeline/performance/*/*.json"))
    ]
    return {
        "settings": "CPU, one Torch thread, <=1024 rows, linear, 1 epoch, batch64, diagnostics off, patch64",
        "dataset_s": dataset_s,
        "training_s": training_s,
        "reevaluation_s": reevaluation_s,
        "dataset_metadata": manifest.metadata,
        "training_manifest": asdict(trained),
        "reevaluation_rows": reevaluation.num_rows,
        "observations": observations,
    }


def main() -> None:
    """Verify source identities and run fresh-process bounded synthetic fixtures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anemone-source", type=Path, required=True)
    parser.add_argument("--nodes", type=int, nargs="+", default=[1000])
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", choices=["prepare", "warm", "cold", "auxiliary"])
    parser.add_argument(
        "--auxiliary",
        action="store_true",
        help="Also time bounded CPU Dataset/Training/Reevaluation stage functions",
    )
    parser.add_argument("--temporary-work", type=Path)
    args = parser.parse_args()
    if not all(1 <= n <= 10000 for n in args.nodes) or not 2 <= args.cycles <= 10:
        parser.error("Only small fixtures (<=10000 nodes, 2..10 cycles) are allowed")
    source = args.anemone_source.resolve()
    if git(source, "rev-parse", "HEAD") != PR77_SHA or git(
        source, "status", "--porcelain"
    ):
        parser.error("Anemone source must be clean, exactly pinned PR #77")
    if not Path(anemone.__file__).resolve().is_relative_to(source / "src"):
        parser.error("Anemone import path does not use the explicitly pinned source")
    logging.disable(logging.CRITICAL)
    if args.child:
        work = args.temporary_work
        if work is None or not work.resolve().is_relative_to(
            Path(tempfile.gettempdir())
        ):
            parser.error(
                "Child fixtures must stay under the system temporary directory"
            )
        result = (
            auxiliary(work)
            if args.child == "auxiliary"
            else (
                prepare(work, args.nodes[0])
                if args.child == "prepare"
                else measure(work, cold=args.child == "cold", cycles=args.cycles)
            )
        )
        repository = Path(__file__).resolve().parents[1]
        result["source_sha256"] = {
            name: hashlib.sha256((repository / name).read_bytes()).hexdigest()
            for name in [
                "src/chipiron/environments/morpion/bootstrap/runtime/runner.py",
                "src/chipiron/environments/morpion/bootstrap/runtime/warm_growth.py",
                "src/chipiron/environments/morpion/bootstrap/pipeline/stages.py",
            ]
        }
        args.output.write_text(json.dumps(result, indent=2))
        return
    repository = Path(__file__).resolve().parents[1]
    results: dict[str, Any] = {
        "anemone_sha": PR77_SHA,
        "anemone_import": anemone.__file__,
        "chipiron_sha": git(repository, "rev-parse", "HEAD"),
        "chipiron_dirty": bool(git(repository, "status", "--porcelain")),
        "python": sys.version,
        "cases": [],
        "scope": "Synthetic CPU fixtures; eviction disabled; no gen38 projection",
    }
    with tempfile.TemporaryDirectory(prefix="chipiron-warm-benchmark-") as directory:
        root = Path(directory)
        for nodes in args.nodes:
            case: dict[str, Any] = {"requested_nodes": nodes}
            for mode in [
                "prepare",
                "cold",
                "warm",
                *(["auxiliary"] if args.auxiliary else []),
            ]:
                work = root / str(nodes) / mode
                if mode != "prepare":
                    shutil.copytree(
                        root
                        / str(nodes)
                        / ("warm" if mode == "auxiliary" else "prepare"),
                        work,
                    )
                output = root / f"{nodes}-{mode}.json"
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--anemone-source",
                    str(source),
                    "--nodes",
                    str(nodes),
                    "--cycles",
                    str(args.cycles),
                    "--child",
                    mode,
                    "--temporary-work",
                    str(work),
                    "--output",
                    str(output),
                ]
                subprocess.run(command, check=True, env=os.environ.copy(), timeout=300)
                case[mode] = json.loads(output.read_text())
            if (
                case["cold"]["checkpoint_digest"] != case["warm"]["checkpoint_digest"]
                or case["cold"]["selected"] != case["warm"]["selected"]
            ):
                message = f"Scientific parity failed for the {nodes}-node fixture"
                raise RuntimeError(message)
            if (
                case["warm"]["restores"] != 1
                or not case["warm"]["same_runtime"]
                or case["cold"]["restores"] != args.cycles
            ):
                message = (
                    "The expected cold-versus-warm restore counts were not observed"
                )
                raise RuntimeError(message)
            case["exact_checkpoint_and_selection_parity"] = True
            results["cases"].append(case)
            args.output.write_text(json.dumps(results, indent=2))
            print(
                f"{nodes}: cold={case['cold']['elapsed_s']:.3f}s warm={case['warm']['elapsed_s']:.3f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()

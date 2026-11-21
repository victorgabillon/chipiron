"""Read-only checkpoint restore, bounded memory attribution, then process exit."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import logging
from dataclasses import asdict, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chipiron.environments.morpion.bootstrap.config import bootstrap_config_from_dict
from chipiron.environments.morpion.bootstrap.control import (
    MorpionBootstrapControl,
    effective_runtime_config_from_config_and_control,
)
from chipiron.environments.morpion.bootstrap.derived.cli import code_identity
from chipiron.environments.morpion.bootstrap.derived.prepare import migrate_config
from chipiron.environments.morpion.bootstrap.derived.provenance import (
    atomic_json,
    inside,
    read_json,
)
from chipiron.environments.morpion.bootstrap.pipeline_memory import available_ram_mb
from chipiron.environments.morpion.bootstrap.runtime.runner import (
    AnemoneMorpionSearchRunner,
    AnemoneMorpionSearchRunnerArgs,
    default_search_args,
)

from .attribution import ProfileLimits, runtime_profile
from .checkpoint_audit import inspect_checkpoint
from .phases import PhaseRecorder, observe_restore

if TYPE_CHECKING:
    from chipiron.environments.morpion.bootstrap.control import (
        MorpionBootstrapEffectiveRuntimeConfig,
    )


class RestoreOnlyRunner(AnemoneMorpionSearchRunner):
    """Explicitly disable scientific advancement and persistence in diagnostic mode."""

    def load_or_create(
        self,
        tree_snapshot_path: str | Path | None,
        model_bundle_path: str | Path | None,
        effective_runtime_config: MorpionBootstrapEffectiveRuntimeConfig | None = None,
        *,
        reevaluate_tree: bool = False,
    ) -> None:
        """Reject fresh creation, export fallback and any reevaluation request."""
        if (
            tree_snapshot_path is None
            or not Path(tree_snapshot_path).is_dir()
            or reevaluate_tree
        ):
            message = "Diagnostics require a runtime checkpoint and forbid fresh creation/reevaluation."
            raise RuntimeError(message)
        super().load_or_create(
            tree_snapshot_path,
            model_bundle_path,
            effective_runtime_config,
            reevaluate_tree=False,
        )

    def grow(self, max_growth_steps: int) -> None:
        """Diagnostics cannot grow even if a caller mistakenly invokes this hook."""
        message = "Growth is forbidden in restore-only diagnostics."
        raise RuntimeError(message)

    def save_checkpoint(self, output_path: str | Path) -> None:
        """Diagnostics cannot save a runtime checkpoint."""
        message = "Checkpoint saving is forbidden in restore-only diagnostics."
        raise RuntimeError(message)


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    """Validate read-only inputs and persist reports exclusively outside experiments."""
    identity = (
        code_identity(args.code_root)
        if args.code_root is not None
        else {"code_sha": "unrecorded-development-invocation"}
    )
    work_dir = args.work_dir.resolve()
    output = args.output_dir.resolve()
    forbidden_roots = [work_dir]
    provenance_path = work_dir / "derived_experiment_provenance.json"
    if provenance_path.exists():
        forbidden_roots.append(
            Path(read_json(provenance_path)["source_work_dir"]).resolve()
        )
    if any(
        output == p or output.is_relative_to(p) or p.is_relative_to(output)
        for p in forbidden_roots
    ):
        message = "Diagnostic output must be outside both historical and derived experiment directories."
        raise ValueError(message)
    limits = ProfileLimits(
        args.sample_nodes, args.recursive_max_objects, args.recursive_max_depth
    )
    state = read_json(work_dir / "run_state.json")
    checkpoint = inside(work_dir, state["latest_runtime_checkpoint_path"])
    manifest = read_json(checkpoint / "manifest.json")
    if (
        not checkpoint.is_dir()
        or manifest["total_node_count"] != state["tree_size_at_last_save"]
    ):
        message = "A matching runtime checkpoint is required; fresh/export fallback is forbidden."
        raise ValueError(message)
    before = {
        str(p.relative_to(work_dir)): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in work_dir.rglob("*")
        if p.is_file()
    }
    raw = read_json(work_dir / "bootstrap_config.json")
    effective, _ = migrate_config(
        raw, growth_steps=raw["runtime"]["max_growth_steps_per_cycle"]
    )
    config = bootstrap_config_from_dict(effective)
    output.mkdir(parents=True, exist_ok=False)
    audit = inspect_checkpoint(checkpoint, sample_nodes=limits.sample_nodes)
    atomic_json(output / "checkpoint-audit.json", audit)
    if args.inspect_only:
        return audit
    minimum_free = max(
        config.runtime.min_available_ram_mb or 0, args.minimum_free_before_restore_mib
    )
    available = available_ram_mb()
    if available is None or available < minimum_free:
        message = f"Insufficient available RAM before restore: {available} MiB; require {minimum_free} MiB. No restore started."
        raise RuntimeError(message)
    runtime_args = asdict(config.runtime)
    accepted = {f.name for f in fields(AnemoneMorpionSearchRunnerArgs)}
    runner = RestoreOnlyRunner(
        AnemoneMorpionSearchRunnerArgs(
            search_args=default_search_args(rollout=config.search.rollout),
            **{k: v for k, v in runtime_args.items() if k in accepted},
        )
    )
    bundle = None
    if args.attach_evaluator:
        active = read_json(work_dir / "pipeline/active_model.json")
        bundle = inside(work_dir, active["model_bundle_path"])
        inside(bundle, "param.pt")
    recorder = PhaseRecorder(output / "restore-phases.jsonl")
    recorder.log("before_diagnostic_restore")
    with observe_restore(recorder):
        runner.load_or_create(
            checkpoint,
            bundle,
            effective_runtime_config_from_config_and_control(
                config, MorpionBootstrapControl()
            ),
            reevaluate_tree=False,
        )
        if runner.current_tree_size() != state["tree_size_at_last_save"]:
            message = "Restored tree size differs from checkpoint; diagnostic refused."
            raise RuntimeError(message)
        recorder.log("final_live_runtime", evaluator_attached=bundle is not None)
        decodes_after_restore = dict(recorder.decode_counts)
        cheap = runtime_profile(runner, limits=limits)
        atomic_json(output / "cheap-profile.json", cheap)
        if args.deep:
            remaining_ram = available_ram_mb()
            if remaining_ram is not None and remaining_ram >= (
                config.runtime.min_available_ram_mb or 0
            ):
                atomic_json(
                    output / "bounded-deep-profile.json",
                    runtime_profile(runner, limits=limits, deep=True),
                )
            else:
                recorder.log("deep_profile_skipped_ram_guard")
        if recorder.decode_counts != decodes_after_restore:
            message = "Memory attribution unexpectedly decoded a state."
            raise RuntimeError(message)
        recorder.log("after_memory_attribution")
    after = {
        str(p.relative_to(work_dir)): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in work_dir.rglob("*")
        if p.is_file()
    }
    if before != after:
        message = "Experiment files changed during diagnostics."
        raise RuntimeError(message)
    result = {
        "status": "RESTORE_ONLY_COMPLETE",
        **identity,
        "growth_steps": 0,
        "scientific_artifacts_unchanged": True,
        "node_count": runner.current_tree_size(),
        "state_decodes": recorder.decode_counts,
        "attribution_added_decodes": 0,
        "first_decode_stacks": recorder.first_decode_stacks,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("chipiron", "algorhino-anemone", "algorhino-coral", "torch")
        },
        "limits": asdict(limits),
        "checkpoint": str(checkpoint),
        "output": str(output),
    }
    atomic_json(output / "summary.json", result)
    del runner
    collected = gc.collect()
    recorder.log("after_runtime_release_and_gc", collected_objects=collected)
    return result


def main(argv: list[str] | None = None) -> int:
    """Require an explicit read-only operation; there is no launch/growth option."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--code-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--restore-only", action="store_true")
    mode.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--attach-evaluator", action="store_true")
    parser.add_argument("--sample-nodes", type=int, default=128)
    parser.add_argument("--recursive-max-objects", type=int, default=20000)
    parser.add_argument("--recursive-max-depth", type=int, default=6)
    parser.add_argument("--minimum-free-before-restore-mib", type=int, default=0)
    parser.add_argument("--deep", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    try:
        print(json.dumps(diagnose(args), indent=2))
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Diagnostic refused/stopped: {exc}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

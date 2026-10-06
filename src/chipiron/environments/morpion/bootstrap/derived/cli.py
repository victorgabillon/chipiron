"""Read-only planning, preparation-only copying, and explicitly confirmed execution."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import platform
import subprocess
from pathlib import Path
from typing import Any

import chipiron

from .prepare import inspect_source, prepare_workspace
from .provenance import (
    DerivationError,
    inside,
    load_provenance,
    read_json,
    sha256,
)


def package_fingerprint(root: Path) -> str:
    """Pin Python implementation bytes independently of editable/wheel installation."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def code_identity(root: Path) -> dict[str, str]:
    """Require a clean, exact Git checkout and matching loaded package implementation."""
    sha = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    )
    if dirty:
        message = "Commit and validate the implementation before preparing a launchable workspace."
        raise DerivationError(message)
    fingerprint = package_fingerprint(root / "src/chipiron")
    if package_fingerprint(Path(chipiron.__file__).parent) != fingerprint:
        message = "Loaded Chipiron package does not match the requested code checkout."
        raise DerivationError(message)
    return {
        "code_sha": sha,
        "code_root": str(root.resolve()),
        "package_fingerprint": fingerprint,
    }


def environment_identity() -> dict[str, str]:
    """Record the scientific dependency versions used by preparation and execution."""
    return {
        "python": platform.python_version(),
        **{
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "numpy",
                "atomheart",
                "algorhino-anemone",
                "algorhino-coral",
                "valanga",
            )
        },
    }


def validate_prepared(
    work_dir: Path, *, require_observability: bool = False
) -> dict[str, Any]:
    """Fail before constructing workers if provenance, code or restore inputs differ."""
    provenance = load_provenance(work_dir)
    if (work_dir / "preparation_incomplete.json").exists():
        message = "Workspace preparation was interrupted."
        raise DerivationError(message)
    identity = code_identity(Path(provenance["code_root"]))
    if any(value != provenance[key] for key, value in identity.items()):
        message = "Code changed since derivation; revalidation is required."
        raise DerivationError(message)
    if environment_identity() != provenance["environment"]:
        message = "Scientific dependency versions changed since preparation."
        raise DerivationError(message)
    if (
        sha256(work_dir / "bootstrap_config.json")
        != provenance["prepared_config_sha256"]
    ):
        message = "Prepared scientific configuration was modified."
        raise DerivationError(message)
    state = read_json(work_dir / "run_state.json")
    journal = read_json(work_dir / "derived_orchestration.json")
    if state["generation"] not in (
        journal["completed_generation"],
        journal["completed_generation"] + 1,
    ):
        message = "Run state and orchestration journal disagree."
        raise DerivationError(message)
    checkpoint = inside(work_dir, state["latest_runtime_checkpoint_path"])
    manifest = read_json(checkpoint / "manifest.json")
    if manifest["total_node_count"] != state["tree_size_at_last_save"]:
        message = "Checkpoint node count differs from run state."
        raise DerivationError(message)
    if state["generation"] == provenance["source_tree_generation"]:
        for relative, checksum in provenance["prepared_restore_sha256"].items():
            if sha256(inside(work_dir, relative)) != checksum:
                message = f"Prepared restore input changed: {relative}"
                raise DerivationError(message)
    active = read_json(work_dir / "pipeline/active_model.json")
    inside(work_dir, active["model_bundle_path"] + "/param.pt")
    has_observability = (
        importlib.util.find_spec("chipiron.environments.morpion.bootstrap.performance")
        is not None
    )
    if require_observability and not has_observability:
        message = "PR #70 observability is required before launch."
        raise DerivationError(message)
    return {
        **provenance,
        "current_generation": state["generation"],
        "current_node_count": state["tree_size_at_last_save"],
        "observability_available": has_observability,
    }


def print_plan(plan: dict[str, Any]) -> None:
    """Print the full operational/scientific plan, excluding large checksum inventories."""
    print(
        json.dumps(
            {
                key: value
                for key, value in plan.items()
                if key not in {"source_file_sha256", "prepared_restore_sha256"}
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    """Require explicit prepare/run choices; absence of --launch never starts work."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    derive = commands.add_parser("derive")
    derive.add_argument("--source-work-dir", type=Path, required=True)
    derive.add_argument("--target-work-dir", type=Path, required=True)
    derive.add_argument("--code-root", type=Path, required=True)
    derive.add_argument("--checkpoint-generation", type=int, required=True)
    derive.add_argument("--expected-node-count", type=int, required=True)
    derive.add_argument("--reset-rng", action="store_true", required=True)
    derive.add_argument(
        "--enable-legacy-graph-tokens-v1", action="store_true", required=True
    )
    derive.add_argument("--search-seed", type=int, required=True)
    derive.add_argument("--rollout-seed", type=int, required=True)
    derive.add_argument("--training-seed", type=int, required=True)
    derive.add_argument("--max-growth-steps-per-cycle", type=int, default=2)
    derive.add_argument("--max-generations", type=int, default=10)
    mode = derive.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--prepare-only", action="store_true")
    run = commands.add_parser("run")
    run.add_argument("--work-dir", type=Path, required=True)
    mode = run.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--launch", action="store_true")
    run.add_argument("--confirm-derived-experiment", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Prepare without search, or enter the serial runner only on double opt-in."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    try:
        return _execute(args)
    except (DerivationError, OSError, ValueError) as exc:
        print(f"Derived experiment refused/stopped: {exc}", flush=True)
        return 2


def _execute(args: argparse.Namespace) -> int:
    """Keep the prepare and execution paths behind the parsed explicit mode."""
    if args.command == "derive":
        identity = code_identity(args.code_root)
        plan = inspect_source(
            args.source_work_dir,
            args.target_work_dir,
            generation=args.checkpoint_generation,
            expected_nodes=args.expected_node_count,
            growth_steps=args.max_growth_steps_per_cycle,
            max_generations=args.max_generations,
            search_seed=args.search_seed,
            rollout_seed=args.rollout_seed,
            training_seed=args.training_seed,
            code_sha=identity["code_sha"],
        )
        plan.update(identity)
        plan["environment"] = environment_identity()
        print_plan(plan)
        if args.prepare_only:
            print(f"Prepared without launch: {prepare_workspace(plan)}")
        return 0
    work_dir = args.work_dir.resolve()
    plan = validate_prepared(work_dir, require_observability=args.launch)
    print_plan(plan)
    if args.dry_run:
        print("DRY RUN — no workers started.")
        return 0
    _require_launch_confirmation(args.confirm_derived_experiment)
    # This is the only entry to execution; dry-run/preparation never acquire a run lock.
    with (work_dir / ".derived_execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from .orchestrator import PipelineGenerationStages, run_sequential

        run_sequential(work_dir, PipelineGenerationStages(work_dir))
    return 0


def _require_launch_confirmation(confirmed: bool) -> None:
    """Require a second explicit opt-in before acquiring any execution lock."""
    if not confirmed:
        message = "Real execution also requires --confirm-derived-experiment."
        raise DerivationError(message)


if __name__ == "__main__":
    raise SystemExit(main())

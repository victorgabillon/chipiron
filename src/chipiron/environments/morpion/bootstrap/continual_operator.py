"""Prepare and resource-monitor the existing scientific four-worker supervisor.

All work scheduling stays in the existing workers and Growth learning boundary.
This operator owns only a bounded supervisor process and its resource accounting.
"""

import fcntl
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack, suppress
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from .config import (
    bootstrap_config_from_args,
    load_bootstrap_config,
    save_bootstrap_config,
)
from .derived.provenance import atomic_json
from .evaluator_family import (
    CANONICAL_LINEAR_MLP_ENTITY_TRANSFORMER_SMALL_MORPION_EVALUATOR_FAMILY_PRESET,
)
from .generation_publication_io import (
    file_digest,
    publication_records,
    read_object,
    require,
)
from .learning_policy import MorpionContinualLearningPolicy, policy_from_metadata
from .pipeline_memory import available_ram_mb, directory_disk_usage
from .run_state import load_bootstrap_run_state

if TYPE_CHECKING:
    from argparse import ArgumentParser

    from .bootstrap_args import MorpionBootstrapArgs
    from .bootstrap_paths import MorpionBootstrapPaths
    from .run_state import MorpionBootstrapRunState


def add_continual_arguments(parser: ArgumentParser) -> None:
    """Expose the explicit experiment policy through the existing launcher."""
    defaults = MorpionContinualLearningPolicy()
    for name, default in asdict(defaults).items():
        if name == "min_available_ram_mb":
            continue  # Reuse the existing launcher option.
        option = "--learning-seed" if name == "seed" else "--" + name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(option, dest=name, action="store_true", default=default)
        else:
            parser.add_argument(option, dest=name, type=type(default), default=default)
    parser.add_argument(
        "--cluster-script",
        type=Path,
        help="Validated launch_morpion_gnome_cluster.sh used by run-continuous.",
    )


def prepare_continual_experiment(
    args: MorpionBootstrapArgs, policy: MorpionContinualLearningPolicy
) -> MorpionBootstrapRunState:
    """Write an explicit nine-candidate configuration and zero-valued fresh genesis."""
    from chipiron.environments.morpion.players.evaluators.neural_networks import (
        MorpionRegressorArgs,
        build_morpion_regressor,
        save_morpion_model_bundle,
    )

    from .scientific_publication_initialization import initialize_scientific_experiment

    root = Path(args.work_dir).resolve()
    require(
        sys.version_info[:2] == (3, 14),
        "Continual preparation requires the validated Python 3.14 environment.",
    )
    require(
        not root.is_symlink()
        and (
            not root.exists()
            or all(
                p.relative_to(root)
                in {Path("launcher_process_state.json"), Path("launcher.pid")}
                for p in root.rglob("*")
                if p.is_file() or p.is_symlink()
            )
        ),
        "Preparation requires a new workspace, never a historical run.",
    )
    inputs = root.with_name(root.name + "-inputs")
    inputs.mkdir(parents=True, exist_ok=False)
    args = replace(
        args,
        continual_learning=policy,
        pipeline_mode="artifact_pipeline",
        runtime_checkpoint_format="sharded",
        training_export_mode="sharded",
        tree_branch_limit=2_000_000_000,
        save_after_seconds=0.0,
        dataset_confidence_policy="selected_value_v2",
        # Ordinary restore attaches the committed evaluator without changing the
        # checkpoint. Atomic activation separately performs a complete refresh.
        evaluator_update_policy="future_only",
        min_available_ram_mb=policy.min_available_ram_mb,
        candidate_checkpoint_load_headroom_factor=0.0,
        candidate_checkpoint_load_min_headroom_mb=0,
        evaluator_family_preset=args.evaluator_family_preset
        or (
            None
            if args.evaluators_config is not None
            else CANONICAL_LINEAR_MLP_ENTITY_TRANSFORMER_SMALL_MORPION_EVALUATOR_FAMILY_PRESET
        ),
        skip_evaluator_diagnostics=True,
    )
    require(
        args.training_evaluator_names is None,
        "Preparation requires the full declared candidate lineup.",
    )
    config = bootstrap_config_from_args(args)
    save_bootstrap_config(config, inputs / "bootstrap_config.json")
    model_args = MorpionRegressorArgs(
        model_kind="linear", feature_subset_name="zero_moves", feature_names=("moves",)
    )
    with torch.random.fork_rng(devices=[]):
        model = build_morpion_regressor(model_args)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        save_morpion_model_bundle(
            model,
            inputs / "zero",
            model_args=model_args,
            metadata={
                "purpose": "initial_zero_search_evaluator",
                "certainty": "ESTIMATE",
                "candidate_lineage": False,
            },
        )
    return initialize_scientific_experiment(
        args, config, seed_model=inputs / "zero", initial_nodes=1
    )


def resource_limit_reason(
    policy: MorpionContinualLearningPolicy, sample: dict[str, Any], elapsed: float
) -> str | None:
    """Evaluate the declared research monitor limits without changing any budget."""
    require(
        math.isfinite(elapsed)
        and elapsed >= 0
        and all(
            isinstance(v, (int, float)) and math.isfinite(v) and v >= 0
            for v in sample.values()
        ),
        "Invalid or nonfinite resource measurement.",
    )
    checks = (
        (elapsed >= policy.max_elapsed_seconds, "maximum_elapsed_time"),
        (
            sample["available_ram_mb"] < policy.min_available_ram_mb,
            "minimum_available_ram",
        ),
        (sample["free_disk_mb"] < policy.min_free_disk_mb, "minimum_free_disk"),
        (sample["group_rss_mb"] > policy.max_group_rss_mb, "maximum_process_group_rss"),
        (sample["workspace_mb"] > policy.max_workspace_mb, "maximum_workspace_disk"),
        (sample["scratch_mb"] > policy.max_scratch_mb, "maximum_scratch_disk"),
        (
            sample["longest_worker_seconds"] > policy.worker_max_seconds,
            "maximum_worker_runtime",
        ),
    )
    return next((reason for reached, reason in checks if reached), None)


def run_monitored_continual_experiment(
    paths: MorpionBootstrapPaths, script: Path
) -> MorpionBootstrapRunState:
    """Run the existing headless supervisor with the existing resource sampler contract."""
    import anemone
    import psutil

    import chipiron

    from .derived.cli import package_fingerprint

    require(
        Path(chipiron.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()),
        "run-continuous requires the verified installed Chipiron wheel.",
    )

    policy = policy_from_metadata(
        load_bootstrap_config(paths.bootstrap_config_path).metadata
    )
    require(policy is not None, "This workspace has no continual-learning policy.")
    assert policy is not None
    state = load_bootstrap_run_state(paths.run_state_path)
    if state.metadata.get("experiment_termination") is not None:
        return state
    producer = publication_records(state.metadata)[-1]["scientific_identity"][
        "producer"
    ]["identity"]
    from platform import python_version

    require(
        sys.version_info[:2] == (3, 14)
        and producer.get("python_version") == python_version()
        and producer.get("torch_version") == str(torch.__version__),
        "Python or Torch runtime differs from the prepared experiment.",
    )
    require(
        producer["revision"] == package_fingerprint(Path(chipiron.__file__).parent)
        and producer["anemone_source_sha256"]
        == package_fingerprint(Path(anemone.__file__).parent),
        "Installed producer source identity differs; resume refused.",
    )
    from importlib import import_module

    for name in ("coral", "atomheart", "valanga", "parsley"):
        module = import_module(name)
        source_file = module.__file__
        require(source_file is not None, f"Missing {name} source file.")
        assert source_file is not None
        require(
            producer.get(name + "_source_sha256")
            == package_fingerprint(Path(source_file).parent),
            f"Installed {name} source identity differs; resume refused.",
        )
    script = script.resolve()
    require(script.is_file(), "The validated headless supervisor script is required.")
    monitor = paths.work_dir / "resource_monitor"
    monitor.mkdir(exist_ok=True)
    lock = (monitor / "owner.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = {"path": str(script), "sha256": file_digest(script)}
        recorded_script = monitor / "supervisor.json"
        if recorded_script.exists():
            require(
                read_object(recorded_script) == identity,
                "Supervisor source changed; resume refused.",
            )
        else:
            atomic_json(recorded_script, identity)
        with ExitStack() as resources:
            return _run_monitored(paths, script, policy, monitor, psutil, resources)
    finally:
        lock.close()


def _run_monitored(
    paths: MorpionBootstrapPaths,
    script: Path,
    policy: MorpionContinualLearningPolicy,
    monitor: Path,
    psutil: Any,
    resources: ExitStack,
) -> MorpionBootstrapRunState:
    usage_path = monitor / "usage.json"
    previous = (
        read_object(usage_path) if usage_path.exists() else {"elapsed_seconds": 0.0}
    )
    consumed = float(previous["elapsed_seconds"])
    require(math.isfinite(consumed) and consumed >= 0, "Invalid resource budget usage.")
    identifier = str(time.time_ns())
    session_path = monitor / f"session_{identifier}.json"
    scratch = Path(tempfile.mkdtemp(prefix="morpion-continuous-"))
    interrupted: list[str] = []
    old_handlers = {
        s: signal.signal(
            s, lambda number, frame: interrupted.append(signal.Signals(number).name)
        )
        for s in (signal.SIGTERM, signal.SIGINT)
    }
    process: subprocess.Popen[bytes] | None = None
    started = time.monotonic()
    reason = "supervisor_exit"
    peaks = {"group_rss_mb": 0.0, "workspace_mb": 0.0, "scratch_mb": 0.0}
    minimum = {"available_ram_mb": float("inf"), "free_disk_mb": float("inf")}
    owned_groups: set[int] = set()

    def sample() -> dict[str, Any]:
        processes: list[Any] = []
        if process is not None:
            with suppress(psutil.NoSuchProcess):
                processes = [
                    psutil.Process(process.pid),
                    *psutil.Process(process.pid).children(recursive=True),
                ]
        rss = 0
        longest = 0.0
        for child in processes:
            try:
                rss += child.memory_info().rss
                owned_groups.add(os.getpgid(child.pid))
                command = child.cmdline()
                if (
                    command
                    and "python" in Path(command[0]).name
                    and "--pipeline-stage" in command
                ):
                    longest = max(longest, time.time() - child.create_time())
            except psutil.NoSuchProcess, ProcessLookupError:
                continue
        ram = available_ram_mb()
        require(ram is not None, "Available RAM monitoring is unavailable.")
        assert ram is not None
        current: dict[str, float] = {
            "group_rss_mb": rss / 2**20,
            "available_ram_mb": ram,
            "free_disk_mb": shutil.disk_usage(paths.work_dir).free / 2**20,
            "workspace_mb": directory_disk_usage(paths.work_dir)["allocated_bytes"]
            / 2**20,
            "scratch_mb": directory_disk_usage(scratch)["allocated_bytes"] / 2**20,
            "longest_worker_seconds": longest,
        }
        for key in peaks:
            peaks[key] = max(peaks[key], current[key])
        for key in minimum:
            minimum[key] = min(minimum[key], current[key])
        return current

    try:
        reason = resource_limit_reason(policy, sample(), consumed) or ""
        if not reason:
            environment = dict(
                os.environ,
                PYTHON_BIN=sys.executable,
                MORPION_SCIENTIFIC="1",
                MORPION_USE_INSTALLED_PACKAGES="1",
                MORPION_CLUSTER_HEADLESS="1",
                MORPION_CLUSTER_WORKERS="all",
                MORPION_RESOURCE_MONITOR_PID=str(os.getpid()),
                MORPION_RESOURCE_MONITOR_START_TICKS=Path("/proc/self/stat")
                .read_text(encoding="utf-8")
                .rsplit(") ", 1)[1]
                .split()[19],
                MORPION_CLUSTER_MAX_SECONDS=str(
                    max(
                        1,
                        min(
                            int(policy.max_elapsed_seconds - consumed),
                            43200 - policy.shutdown_grace_seconds - 5,
                        ),
                    )
                ),
                MORPION_CLUSTER_STOP_GRACE_SECONDS=str(policy.shutdown_grace_seconds),
                MORPION_WORK_DIR=str(paths.work_dir),
                MORPION_SEED_MODELS_FROM_WORK_DIR="",
                MORPION_CLUSTER_OPEN_STATUS="0",
                MORPION_CLUSTER_SHOW_RECAP="0",
                MORPION_GROWTH_SHOW_RECAP="0",
                MORPION_GROWTH_WORKER_MAX_CYCLES="1",
                MORPION_MIN_AVAILABLE_RAM_MB=str(policy.min_available_ram_mb),
                TMPDIR=str(scratch),
                SQLITE_TMPDIR=str(scratch),
            )
            with (monitor / f"session_{identifier}.log").open("w") as output:
                process = resources.enter_context(
                    # ExitStack owns this process until monitored shutdown completes.
                    subprocess.Popen(  # pylint: disable=consider-using-with
                        [
                            "/usr/bin/time",
                            "-q",
                            "-f",
                            '{"wall_s":%e,"user_s":%U,"system_s":%S,"peak_rss_kib":%M}',
                            "-o",
                            str(monitor / f"session_{identifier}.native.json"),
                            "bash",
                            str(script),
                        ],
                        env=environment,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                )
                while process.poll() is None:
                    elapsed = consumed + time.monotonic() - started
                    current = sample()
                    state = load_bootstrap_run_state(paths.run_state_path)
                    termination = state.metadata.get("experiment_termination")
                    reason = (
                        "operator_" + interrupted[-1]
                        if interrupted
                        else termination["reason"]
                        if termination is not None
                        else resource_limit_reason(policy, current, elapsed)
                    ) or ""
                    atomic_json(
                        usage_path,
                        {
                            "elapsed_seconds": elapsed,
                            "session": identifier,
                            "reason": reason or "running",
                        },
                    )
                    if reason:
                        break
                    time.sleep(1)
                if not reason:
                    reason = "supervisor_exit"
    finally:
        shutdown_confirmed = True
        if process is not None:
            if process.poll() is None:
                # Signal the actual headless shell; GNU time remains its parent
                # and records complete child CPU usage after the shell reaps workers.
                try:
                    leaders = psutil.Process(process.pid).children()
                except psutil.NoSuchProcess:
                    leaders = []
                for leader in leaders:
                    with suppress(ProcessLookupError):
                        os.kill(leader.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=policy.shutdown_grace_seconds + 10)
                except subprocess.TimeoutExpired:
                    for group in owned_groups:
                        with suppress(ProcessLookupError):
                            os.killpg(group, signal.SIGKILL)
                    process.wait(timeout=5)
            # The existing supervisor confirms all groups. Independently retain
            # scratch if any sampled owned group still has a living member.
            live = []
            for child in psutil.process_iter(["pid", "status"]):
                try:
                    if os.getpgid(child.pid) in owned_groups and child.status() not in {
                        psutil.STATUS_ZOMBIE,
                        psutil.STATUS_DEAD,
                    }:
                        live.append(child.pid)
                except psutil.NoSuchProcess, ProcessLookupError:
                    continue
            shutdown_confirmed = not live
        elapsed = consumed + time.monotonic() - started
        atomic_json(
            usage_path,
            {"elapsed_seconds": elapsed, "session": identifier, "reason": reason},
        )
        atomic_json(
            session_path,
            {
                "policy": asdict(policy),
                "reason": reason,
                "elapsed_seconds": elapsed - consumed,
                "budget_elapsed_seconds": elapsed,
                "peaks": peaks,
                "minimum": minimum,
                "scratch": str(scratch),
                "supervisor_exit_code": None if process is None else process.returncode,
                "shutdown_confirmed": shutdown_confirmed,
            },
        )
        if shutdown_confirmed:
            shutil.rmtree(scratch)
        for number, handler in old_handlers.items():
            signal.signal(number, handler)
        require(
            shutdown_confirmed,
            "Worker group shutdown is unconfirmed; scratch retained.",
        )
    result = load_bootstrap_run_state(paths.run_state_path)
    print(
        json.dumps({
            "workspace": str(paths.work_dir),
            "reason": reason,
            "unique_nodes": result.tree_size_at_last_save,
            "completed_learning_rounds": sum(
                "accepted_training" in r for r in publication_records(result.metadata)
            ),
            "resource_usage": str(usage_path),
        })
    )
    return result

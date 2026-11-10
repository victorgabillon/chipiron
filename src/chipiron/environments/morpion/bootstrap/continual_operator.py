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
import socket
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
from .persistent_worker_state import (
    MAX_STATUS_BYTES,
    SupervisorShutdown,
    WorkerJobStatus,
    read_job_status,
    validate_shutdown_intent,
)
from .pipeline_memory import available_ram_mb, directory_disk_usage
from .run_state import load_bootstrap_run_state

if TYPE_CHECKING:
    from argparse import ArgumentParser

    from .bootstrap_args import MorpionBootstrapArgs
    from .bootstrap_paths import MorpionBootstrapPaths
    from .managed_sessions import AdditionalTimeRequest
    from .run_state import MorpionBootstrapRunState


def add_continual_arguments(parser: ArgumentParser) -> None:
    """Expose the explicit experiment policy through the existing launcher."""
    schedules = parser.add_mutually_exclusive_group()
    schedules.add_argument(
        "--growth-lookahead-cycles",
        type=int,
        default=None,
        help="Fresh opt-in bounded Growth window; full refresh remains owned.",
    )
    schedules.add_argument(
        "--ready-growth-max-cycles",
        type=int,
        default=None,
        help="Fresh ready-result policy; maximum ordinary cycles before waiting, with owned full refresh.",
    )
    defaults = MorpionContinualLearningPolicy()
    for name, default in asdict(defaults).items():
        if name == "transformer_cpu_threads":
            parser.add_argument(
                "--transformer-cpu-threads",
                type=int,
                choices=(1, 2),
                default=None,
                help="Opt-in transformer_v1 intra-op threads; explicit CPU, serial Training only.",
            )
            continue
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
    args: MorpionBootstrapArgs,
    policy: MorpionContinualLearningPolicy,
    *,
    managed_sessions: bool = False,
) -> MorpionBootstrapRunState:
    """Write an explicit nine-candidate configuration and zero-valued fresh genesis."""
    from .execution_profile import require_args_profile

    require_args_profile(
        replace(args, continual_learning=policy), stage="prepare-continuous"
    )
    from chipiron.environments.morpion.players.evaluators.neural_networks import (
        MorpionRegressorArgs,
        build_morpion_regressor,
        save_morpion_model_bundle,
    )

    from .scientific_publication_initialization import initialize_scientific_experiment

    if policy.transformer_cpu_threads is not None:
        from .candidate_resources import require_serial_cpu

        require_serial_cpu(args.training_device)
        require(
            "transformer_v1" in args.resolved_evaluators_config().evaluators,
            "Transformer CPU threads require a transformer_v1 candidate.",
        )
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
    if args.learning_schedule is None:
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
    from .learning_schedule import validate_supported

    validate_supported(args)
    if args.learning_schedule is not None:
        inputs.mkdir(parents=True, exist_ok=False)
    config = bootstrap_config_from_args(args)
    if managed_sessions:
        from .managed_sessions import KEY, capability

        require(
            1 <= policy.max_elapsed_seconds <= 43200,
            "Managed initial allowance must be in 1..43200 seconds.",
        )
        config = replace(config, metadata={**config.metadata, KEY: capability(root)})
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
    policy: MorpionContinualLearningPolicy,
    sample: dict[str, Any],
    elapsed: float,
    *,
    elapsed_ceiling_seconds: float | None = None,
) -> str | None:
    """Evaluate the declared research monitor limits without changing any budget."""
    require(
        isinstance(elapsed, (int, float))
        and not isinstance(elapsed, bool)
        and math.isfinite(elapsed)
        and elapsed >= 0
        and all(
            isinstance(v, (int, float))
            and not isinstance(v, bool)
            and math.isfinite(v)
            and v >= 0
            for v in sample.values()
        ),
        "Invalid or nonfinite resource measurement.",
    )
    require(
        elapsed_ceiling_seconds is None
        or (
            isinstance(elapsed_ceiling_seconds, (int, float))
            and not isinstance(elapsed_ceiling_seconds, bool)
            and math.isfinite(elapsed_ceiling_seconds)
            and elapsed_ceiling_seconds > 0
        ),
        "Invalid resolved elapsed ceiling.",
    )
    checks = (
        (
            elapsed
            >= (
                policy.max_elapsed_seconds
                if elapsed_ceiling_seconds is None
                else elapsed_ceiling_seconds
            ),
            "maximum_elapsed_time",
        ),
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
    paths: MorpionBootstrapPaths,
    script: Path,
    *,
    additional_time: AdditionalTimeRequest | None = None,
) -> MorpionBootstrapRunState:
    """Run the existing headless supervisor with the existing resource sampler contract."""
    from .execution_profile import require_workspace_profile

    require_workspace_profile(paths, require_device=True)
    import anemone
    import psutil

    import chipiron

    from .derived.cli import package_fingerprint

    require(
        Path(chipiron.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()),
        "run-continuous requires the verified installed Chipiron wheel.",
    )

    config = load_bootstrap_config(paths.bootstrap_config_path)
    policy = policy_from_metadata(config.metadata)
    require(policy is not None, "This workspace has no continual-learning policy.")
    assert policy is not None
    state = load_bootstrap_run_state(paths.run_state_path)
    from .learning_policy import validate_continual_context
    from .learning_schedule import WINDOW_KEY, schedule_from_metadata

    schedule = schedule_from_metadata(config.metadata)
    if (
        schedule is not None
        or WINDOW_KEY in state.metadata
        or state.metadata.get("continual_learning") != {"schema_version": 1}
    ):
        validate_continual_context(paths, state, policy, schedule)
    if state.metadata.get("experiment_termination") is not None:
        require(
            additional_time is None,
            "Additional time cannot override scientific termination.",
        )
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
    from . import managed_sessions as managed

    managed_data = managed.validate(paths, config, state)
    require(
        additional_time is None or managed_data is not None,
        "Additional time requires managed fresh enrollment.",
    )
    managed_summary = (
        managed.inspect(paths, config, state) if managed_data is not None else None
    )
    require(
        managed_summary is None or managed_summary["status"] != "blocked/uncertain",
        "Managed completion is uncertain or blocked; no monitor metadata was changed.",
    )
    if additional_time is not None:
        planned, _ = managed.plan_additional_time(
            paths,
            managed_summary,
            additional_time.seconds,
            additional_time.request_id,
        )
        require(
            planned == additional_time,
            "Additional-time predecessor changed; preview again.",
        )
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
        from .generation_publication_io import growth_receipt_ownership

        if managed_data is not None:
            managed.recover_close_owned(paths, config)
            with growth_receipt_ownership(paths, create=False):
                state = load_bootstrap_run_state(paths.run_state_path)
                state, work = managed.admit_additional_time_owned(
                    paths, config, state, additional_time
                )
            if not work:
                return state
            ceiling = state.metadata[managed.KEY]["authorized_seconds"]
        else:
            require(
                additional_time is None,
                "Additional time requires managed fresh enrollment.",
            )
            ceiling = None
        if not recorded_script.exists():
            atomic_json(recorded_script, identity)
        with ExitStack() as resources:
            return _run_monitored(
                paths,
                script,
                policy,
                monitor,
                psutil,
                resources,
                managed_ceiling=ceiling,
            )
    finally:
        lock.close()


def _run_monitored(
    paths: MorpionBootstrapPaths,
    script: Path,
    policy: MorpionContinualLearningPolicy,
    monitor: Path,
    psutil: Any,
    resources: ExitStack,
    *,
    managed_ceiling: float | None = None,
) -> MorpionBootstrapRunState:
    usage_path = monitor / "usage.json"
    previous = (
        read_object(usage_path) if usage_path.exists() else {"elapsed_seconds": 0.0}
    )
    require(
        isinstance(previous["elapsed_seconds"], (int, float))
        and not isinstance(previous["elapsed_seconds"], bool),
        "Invalid resource budget clock.",
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
    environment_owned_groups: set[int] = set()
    owned_process_starts: dict[int, str] = {}
    control: socket.socket | None = None
    control_buffer = bytearray()
    shutdown: SupervisorShutdown | None = None
    control_failure: str | None = None
    readings: list[tuple[int, WorkerJobStatus]] = []
    worker_failures: list[tuple[int, str]] = []
    decision: dict[str, Any] | None = None
    supervisor_limit = max(
        1,
        min(
            int(
                (
                    policy.max_elapsed_seconds
                    if managed_ceiling is None
                    else managed_ceiling
                )
                - consumed
            ),
            43200 - policy.shutdown_grace_seconds - 5,
        ),
    )
    managed_link = None
    launched = started
    ordinary_overrun = False
    ordinary_age = 0.0
    ordinary_pid: int | None = None
    current: dict[str, Any] = {}
    environment: dict[str, str] = {}

    def read_shutdown() -> None:
        nonlocal shutdown, control_failure
        if control is None or process is None or control_failure is not None:
            return
        try:
            while True:
                try:
                    message = control.recv(MAX_STATUS_BYTES + 1)
                except BlockingIOError:
                    return
                if not message:
                    require(
                        not control_buffer, "Incomplete supervisor shutdown message"
                    )
                    return
                require(shutdown is None, "Repeated supervisor shutdown intent")
                control_buffer.extend(message)
                require(
                    len(control_buffer) <= MAX_STATUS_BYTES, "Oversized shutdown intent"
                )
                if b"\n" not in control_buffer:
                    continue
                line, remainder = control_buffer.split(b"\n", 1)
                require(not remainder, "Multiple supervisor shutdown messages")
                record = json.loads(line)
                shutdown = validate_shutdown_intent(
                    record,
                    monitor_session=identifier,
                    parent_pid=process.pid,
                    limit_seconds=supervisor_limit,
                    grace_seconds=policy.shutdown_grace_seconds,
                    not_before=launched + supervisor_limit - 1,
                    now=time.monotonic(),
                )
                control_buffer.clear()
        except OSError, ValueError, KeyError, TypeError, IndexError, RecursionError:
            shutdown = None
            control_failure = "invalid_supervisor_shutdown"

    def limit_reason(current: dict[str, Any], now: float) -> str | None:
        """Keep native elapsed/resource priority and distinguish real job failures."""
        nonlocal worker_failures
        worker_failures = [
            (pid, failure)
            for pid, reading in readings
            if (failure := reading.shutdown_failure(shutdown, now)) is not None
        ]
        current["longest_worker_seconds"] = max(
            ordinary_age,
            max(
                (reading.active_age(shutdown, now) for _pid, reading in readings),
                default=0.0,
            ),
        )
        resource_reason = resource_limit_reason(
            policy,
            current,
            consumed + now - started,
            elapsed_ceiling_seconds=managed_ceiling,
        )
        if resource_reason not in {None, "maximum_worker_runtime"}:
            return resource_reason
        genuine_overrun = ordinary_overrun or any(
            reading.record is not None
            and (
                reading.record["phase"] in {"starting", "running", "cleanup"}
                or (
                    shutdown is not None
                    and reading.shutdown_failure(shutdown, now) is None
                )
            )
            and reading.active_age(shutdown, now) > policy.worker_max_seconds
            for _pid, reading in readings
        )
        failure = control_failure or next(
            (value for _pid, value in worker_failures), None
        )
        if failure is not None and not genuine_overrun:
            return failure
        if resource_reason:
            return resource_reason
        if shutdown is not None:
            return (
                "supervisor_shutdown_grace_expired"
                if now > shutdown.deadline
                else "supervisor_deadline"
            )
        return None

    def record_decision(current: dict[str, Any], now: float) -> None:
        nonlocal decision, control_failure
        if control_buffer:
            control_failure = "invalid_supervisor_shutdown"
        trigger = next(
            (pid for pid, failure in worker_failures if failure == reason),
            None,
        )
        if reason == "maximum_worker_runtime":
            trigger = max(
                [(ordinary_pid, ordinary_age)]
                + [
                    (pid, reading.active_age(shutdown, now))
                    for pid, reading in readings
                ],
                key=lambda item: item[1],
            )[0]
        decision = {
            "reason": reason,
            **({"managed_session": managed_link} if managed_link is not None else {}),
            "monotonic_s": now,
            "budget_elapsed_seconds": consumed + now - started,
            "sample": current,
            "triggering_worker_pid": trigger,
            "workers": [
                {
                    "pid": pid,
                    "status": reading.record,
                    "fallback_process_or_job_age_s": reading.age,
                    "accounted_job_age_s": reading.active_age(shutdown, now),
                }
                for pid, reading in readings
            ],
            "supervisor_shutdown": None if shutdown is None else shutdown.record,
            "control_failure": control_failure,
            "worker_failures": worker_failures,
            "ordinary_worker": {"pid": ordinary_pid, "age_s": ordinary_age},
            "supervisor_pid": None if process is None else process.pid,
            "supervisor_exit_code": None if process is None else process.returncode,
        }
        atomic_json(monitor / f"decision_{identifier}.json", decision)
        if control is not None and (
            shutdown is not None or control_failure is not None
        ):
            # Persist the decisive native clock/status BEFORE the supervisor can
            # signal workers. No cleanup-inclusive elapsed supplies this reason.
            with suppress(OSError):
                control.sendall(b"ok\n" if shutdown is not None else b"no\n")

    def sample() -> dict[str, Any]:
        nonlocal readings, worker_failures, ordinary_overrun, ordinary_age, ordinary_pid
        readings = []
        ordinary_pid = None
        ordinary_longest = 0.0
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
                    # Linux start ticks and BOOTTIME share a clock; Unix clock
                    # adjustments must not create or conceal a worker overrun.
                    fields = (
                        Path(f"/proc/{child.pid}/stat")
                        .read_text(encoding="utf-8")
                        .rsplit(") ", 1)[1]
                        .split()
                    )
                    elapsed = time.clock_gettime(time.CLOCK_BOOTTIME) - int(
                        fields[19]
                    ) / os.sysconf("SC_CLK_TCK")
                    if "--persistent-worker" in command:
                        reading = read_job_status(paths.work_dir, child.pid, elapsed)
                        readings.append((child.pid, reading))
                        elapsed = reading.age
                    else:
                        if elapsed > ordinary_longest:
                            ordinary_pid = child.pid
                        ordinary_longest = max(ordinary_longest, elapsed)
                    longest = max(longest, elapsed)
            except psutil.NoSuchProcess, ProcessLookupError, FileNotFoundError:
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
        # Read intent after status/disk reads: the supervisor may have published
        # its pre-signal record while sampling. A stopped record alone is inert.
        read_shutdown()
        now = time.monotonic()
        worker_failures = [
            (pid, failure)
            for pid, reading in readings
            if (failure := reading.shutdown_failure(shutdown, now)) is not None
        ]
        if readings:
            current["longest_worker_seconds"] = max(
                ordinary_longest,
                *(reading.active_age(shutdown, now) for _pid, reading in readings),
            )
        ordinary_overrun = ordinary_longest > policy.worker_max_seconds
        ordinary_age = ordinary_longest
        for key in peaks:
            peaks[key] = max(peaks[key], current[key])
        for key in minimum:
            minimum[key] = min(minimum[key], current[key])
        return current

    def remember_owned_groups() -> None:
        """Retain Linux ancestry independently of a possibly failing sampler."""
        if process is None:
            return
        pending = [process.pid] if process.poll() is None else []
        seen: set[int] = set()
        while pending:
            pid = pending.pop()
            if pid in seen:
                continue
            seen.add(pid)
            try:
                owned_groups.add(os.getpgid(pid))
                pending.extend(
                    int(value)
                    for value in Path(f"/proc/{pid}/task/{pid}/children")
                    .read_text(encoding="utf-8")
                    .split()
                )
            except FileNotFoundError, ProcessLookupError:
                continue
        # A killed shell can reparent setsid workers before our first sample.
        # Recover only this invocation's inherited kernel environment, never PID
        # claims from status files. This grants cleanup ownership, not acceptance.
        needles = {
            b"\0" + os.fsencode(f"{key}={environment[key]}") + b"\0"
            for key in (
                "MORPION_RESOURCE_MONITOR_PID",
                "MORPION_RESOURCE_MONITOR_START_TICKS",
                "MORPION_RESOURCE_MONITOR_SESSION",
                "MORPION_WORK_DIR",
            )
        }
        overlap = max(map(len, needles)) - 1
        for entry in Path("/proc").iterdir():
            if not entry.name.isdecimal():
                continue
            try:
                if entry.stat().st_uid != os.getuid():
                    continue
                before = (
                    (entry / "stat")
                    .read_text(encoding="utf-8")
                    .rsplit(") ", 1)[1]
                    .split()
                )
                found: set[bytes] = set()
                tail = b"\0"
                # exec bounds the complete argument/environment area. Stream it
                # with fixed memory; unrelated large variables cannot grow a buffer.
                remaining = max(MAX_STATUS_BYTES, os.sysconf("SC_ARG_MAX"))
                with (entry / "environ").open("rb") as stream:
                    while remaining > 0 and found != needles:
                        block = stream.read(min(MAX_STATUS_BYTES, remaining))
                        if not block:
                            break
                        remaining -= len(block)
                        data = tail + block
                        found.update(needle for needle in needles if needle in data)
                        tail = data[-overlap:]
                if found != needles:
                    continue
                after = (
                    (entry / "stat")
                    .read_text(encoding="utf-8")
                    .rsplit(") ", 1)[1]
                    .split()
                )
                if before[19] == after[19] and before[2] == after[2]:
                    owned_groups.add(int(after[2]))
                    environment_owned_groups.add(int(after[2]))
                    owned_process_starts[int(entry.name)] = after[19]
            except FileNotFoundError, ProcessLookupError, PermissionError:
                continue

    def live_owned_processes() -> list[int]:
        """Confirm KILL completion before retiring any attempt inputs."""
        live = []
        for child in psutil.process_iter(["pid", "status"]):
            try:
                group = os.getpgid(child.pid)
                if child.pid in owned_process_starts:
                    fields = (
                        Path(f"/proc/{child.pid}/stat")
                        .read_text(encoding="utf-8")
                        .rsplit(") ", 1)[1]
                        .split()
                    )
                    # A dispatched child can exec/setsid after discovery. Only
                    # its exact kernel PID/start identity grants the new group.
                    if fields[19] == owned_process_starts[child.pid]:
                        group = int(fields[2])
                        owned_groups.add(group)
                        environment_owned_groups.add(group)
                if group in owned_groups and child.status() not in {
                    psutil.STATUS_ZOMBIE,
                    psutil.STATUS_DEAD,
                }:
                    live.append(child.pid)
            except psutil.NoSuchProcess, ProcessLookupError, FileNotFoundError:
                continue
        return live

    try:
        if managed_ceiling is not None:
            from .managed_sessions import begin_owned

            managed_link = begin_owned(
                paths,
                load_bootstrap_config(paths.bootstrap_config_path),
                identifier,
                started,
                supervisor_limit,
            )
        current = sample()
        decisive_now = time.monotonic()
        elapsed = consumed + decisive_now - started
        reason = limit_reason(current, decisive_now) or ""
        if not reason:
            control, supervisor_control = socket.socketpair()
            resources.callback(control.close)
            resources.callback(supervisor_control.close)
            control.setblocking(False)
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
                MORPION_RESOURCE_MONITOR_SESSION=identifier,
                MORPION_SUPERVISOR_SHUTDOWN_FD=str(supervisor_control.fileno()),
                MORPION_CLUSTER_MAX_SECONDS=str(supervisor_limit),
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
                launched = time.monotonic()
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
                        pass_fds=(supervisor_control.fileno(),),
                    )
                )
                if process.poll() is None:
                    with suppress(ProcessLookupError):
                        owned_groups.add(os.getpgid(process.pid))
                supervisor_control.close()
                while process.poll() is None:
                    current = sample()
                    state = load_bootstrap_run_state(paths.run_state_path)
                    termination = state.metadata.get("experiment_termination")
                    # Sampling/status reads can cross the deadline. Decide from
                    # their completion time, before shutdown can alter the label.
                    decisive_now = time.monotonic()
                    elapsed = consumed + decisive_now - started
                    resource_reason = limit_reason(current, decisive_now)
                    reason = (
                        "operator_" + interrupted[-1]
                        if interrupted
                        else termination["reason"]
                        if termination is not None
                        else resource_reason
                    ) or ""
                    atomic_json(
                        usage_path,
                        {
                            "elapsed_seconds": elapsed,
                            **(
                                {"managed_session": managed_link}
                                if managed_link is not None
                                else {}
                            ),
                            "session": identifier,
                            "reason": reason or "running",
                        },
                    )
                    if reason:
                        record_decision(current, decisive_now)
                        break
                    time.sleep(1)
                if not reason:
                    read_shutdown()
                    decisive_now = time.monotonic()
                    elapsed = consumed + decisive_now - started
                    resource_reason = limit_reason(current, decisive_now)
                    reason = (
                        "operator_" + interrupted[-1]
                        if interrupted
                        else resource_reason
                    ) or "supervisor_exit"
        if decision is None:
            record_decision(current, decisive_now)
    except BaseException:
        if decision is None:
            reason = "operator_" + interrupted[-1] if interrupted else "monitor_error"
            record_decision(current, time.monotonic())
        raise
    finally:
        try:
            shutdown_confirmed = True
            shutdown_grace_expired = False
            forced_group_cleanup = False
            cleanup_confirmed_at: float | None = None
            kill_confirmation_deadline: float | None = None
            if process is not None:
                try:
                    remember_owned_groups()
                    if process.poll() is None:
                        # Signal the actual headless shell; GNU time remains its parent
                        # and records complete child CPU usage after the shell reaps workers.
                        try:
                            root = psutil.Process(process.pid)
                            leaders = root.children()
                            for child in root.children(recursive=True):
                                with suppress(ProcessLookupError):
                                    owned_groups.add(os.getpgid(child.pid))
                        except psutil.NoSuchProcess:
                            leaders = []
                        for leader in leaders:
                            with suppress(ProcessLookupError):
                                os.kill(leader.pid, signal.SIGTERM)
                        # Queue TERM first: otherwise a helper racing the final sample
                        # could turn native cancellation into a failed ACK/exit1.
                        if control is not None:
                            control.close()
                        try:
                            process.wait(
                                timeout=policy.shutdown_grace_seconds + 10
                                if shutdown is None
                                else max(0.0, shutdown.deadline - time.monotonic())
                            )
                        except subprocess.TimeoutExpired:
                            shutdown_grace_expired = True
                            forced_group_cleanup = True
                            kill_confirmation_deadline = time.monotonic() + 5
                            for group in owned_groups:
                                with suppress(ProcessLookupError):
                                    os.killpg(group, signal.SIGKILL)
                            process.wait(timeout=5)
                    # Dispatch can race the pre-TERM scan and Bash's PID registration.
                    # Once the supervisor is reaped, recover any final detached group.
                    remember_owned_groups()
                    # The existing supervisor confirms all groups. Independently retain
                    # scratch if any sampled owned group still has a living member.
                    live = live_owned_processes()
                    if live:
                        forced_group_cleanup = True
                        if kill_confirmation_deadline is None:
                            kill_confirmation_deadline = time.monotonic() + 5
                        while live and time.monotonic() < kill_confirmation_deadline:
                            for group in owned_groups:
                                with suppress(ProcessLookupError):
                                    os.killpg(group, signal.SIGKILL)
                            time.sleep(0.01)
                            live = live_owned_processes()
                    shutdown_confirmed = not live
                    cleanup_confirmed_at = time.monotonic()
                    if (
                        shutdown is not None
                        and cleanup_confirmed_at > shutdown.deadline
                    ):
                        shutdown_grace_expired = True
                    shutdown_grace_expired = (
                        shutdown_grace_expired or process.returncode == 124
                    )
                    if reason == "supervisor_deadline":
                        if shutdown_grace_expired or process.returncode == 124:
                            reason = "supervisor_shutdown_grace_expired"
                        elif forced_group_cleanup or process.returncode not in {0, 143}:
                            reason = "supervisor_failed_during_shutdown"
                except BaseException:
                    shutdown_confirmed = False
                    if process.poll() is None:
                        with suppress(ProcessLookupError):
                            os.killpg(process.pid, signal.SIGTERM)
                    if control is not None:
                        control.close()
                    try:
                        remember_owned_groups()
                    finally:
                        for group in owned_groups:
                            with suppress(ProcessLookupError):
                                os.killpg(group, signal.SIGKILL)
                        process.wait(timeout=5)
                    raise
            elapsed = consumed + time.monotonic() - started
            atomic_json(
                usage_path,
                {
                    "elapsed_seconds": elapsed,
                    "session": identifier,
                    "reason": reason,
                    **(
                        {"managed_session": managed_link}
                        if managed_link is not None
                        else {}
                    ),
                },
            )
            atomic_json(
                session_path,
                {
                    "policy": policy.to_metadata(),
                    **(
                        {
                            "managed_session": managed_link,
                            "managed_owned_groups": sorted(owned_groups),
                        }
                        if managed_link is not None
                        else {}
                    ),
                    "reason": reason,
                    "elapsed_seconds": elapsed - consumed,
                    "budget_elapsed_seconds": elapsed,
                    "peaks": peaks,
                    "minimum": minimum,
                    "scratch": str(scratch),
                    "supervisor_exit_code": None
                    if process is None
                    else process.returncode,
                    "shutdown_confirmed": shutdown_confirmed,
                    "decision": decision,
                    "shutdown_grace_expired": shutdown_grace_expired,
                    "cleanup_confirmed_monotonic_s": cleanup_confirmed_at,
                    "forced_group_cleanup": forced_group_cleanup,
                    "environment_owned_groups": sorted(environment_owned_groups),
                    "environment_owned_processes": owned_process_starts,
                },
            )
            if shutdown_confirmed:
                shutil.rmtree(scratch)
            require(
                shutdown_confirmed,
                "Worker group shutdown is unconfirmed; scratch retained.",
            )
        finally:
            if control is not None:
                control.close()
            for number, handler in old_handlers.items():
                signal.signal(number, handler)
    require(
        not shutdown_grace_expired
        and not forced_group_cleanup
        and not worker_failures
        and control_failure is None
        and reason
        not in {
            "persistent_worker_failed",
            "unexpected_persistent_worker_stop",
            "invalid_persistent_worker_status",
            "stale_persistent_worker_stop",
            "invalid_supervisor_shutdown",
            "supervisor_failed_during_shutdown",
            "supervisor_shutdown_grace_expired",
        },
        "Persistent worker group failed; all owned groups stopped before restart.",
    )
    require(
        process is None
        or process.returncode in {0, 143}
        or (reason == "operator_SIGINT" and process.returncode == 130),
        "Persistent worker group failed; all owned groups stopped before restart.",
    )
    if os.environ.get("MORPION_PERSISTENT_WORKERS") == "1" and reason in {
        "supervisor_exit",
        "supervisor_failed_during_shutdown",
        "supervisor_shutdown_grace_expired",
    }:
        require(
            process is not None and process.returncode == 0,
            "Persistent worker group failed; all owned groups stopped before restart.",
        )
    if managed_link is not None:
        from .managed_sessions import finalize_managed_session_owned

        finalize_managed_session_owned(
            paths, load_bootstrap_config(paths.bootstrap_config_path), identifier
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

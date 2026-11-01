"""Optional, bounded runtime evidence; never discover a supervisor by parent path."""
# The view model imports this adapter only inside load/classify functions.
# pylint: disable=cyclic-import

import os
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .performance_evidence import EvidenceReader

from .view_model import (
    WorkerSummary,
    age_label,
    mapping,
    number,
    read_artifact,
    read_history,
    timestamp,
)


def live_identity(identity: dict[str, Any], work_dir: Path | None = None) -> bool:
    """Verify boot and start ticks, rejecting stale/reused PIDs and remote evidence."""
    pid = identity.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    try:
        proc = Path("/proc") / str(pid)
        fields = (proc / "stat").read_text(encoding="utf-8").rsplit(") ", 1)[1].split()
        boot = (
            Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
        )
        if (
            fields[0] == "Z"
            or identity.get("start_ticks") != fields[19]
            or identity.get("boot", identity.get("boot_id")) != boot
        ):
            return False
        if "session" in identity and identity["session"] != int(fields[3]):
            return False
        if "pid_namespace" in identity and identity["pid_namespace"] != os.readlink(
            proc / "ns/pid"
        ):
            return False
        if identity.get("command") is not None:
            actual = (proc / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
            if actual != identity["command"]:
                return False
        if work_dir is not None:
            args = (proc / "cmdline").read_bytes().decode().split("\0")
            if "--work-dir" not in args or args[args.index("--work-dir") + 1] != str(
                work_dir
            ):
                return False
    except OSError, ValueError, UnicodeError, IndexError:
        return False
    return True


def load_runtime(
    work_dir: Path, root: Path | None, errors: list[str]
) -> dict[str, Any]:
    """Bind explicit B3 runtime evidence to its prepared workspace and run command."""
    if root is None:
        return {}
    root = root.resolve()
    ready = read_artifact(root / "LAUNCH_READY.json", errors)
    command = mapping(ready.get("commands")).get("run")
    command_matches = (
        isinstance(command, list)
        and "--work-dir" in command
        and command[command.index("--work-dir") + 1 :][:1] == [str(work_dir)]
    )
    if (
        ready.get("schema_version", 1) != 1
        or ready.get("scientific_work_dir", ready.get("work_dir")) != str(work_dir)
        or ready.get("runtime") != str(root)
        or not command_matches
    ):
        errors.append(
            "Runtime association refused: LAUNCH_READY workspace/runtime/run command mismatch or unsupported schema."
        )
        return {}
    result: dict[str, Any] = {
        "root": str(root),
        "origin": "Explicit B3 runtime; legacy unversioned receipt adapter",
        "ready": ready,
    }
    for key, relative in [
        ("status", "status.json"),
        ("owner", "owner.json"),
        ("receipt", "reports/run-receipt.json"),
        ("release", "reports/lease-release.json"),
    ]:
        artifact = read_artifact(root / relative, errors)
        if artifact.get("schema_version", 1) != 1:
            errors.append(f"{relative}: unsupported runtime schema version")
            return {}
        result[key] = artifact
    receipt_command = mapping(result["receipt"].get("commands")).get("run")
    if result["receipt"] and receipt_command != command:
        errors.append(
            "Runtime receipt command differs from associated prepared command."
        )
        return {}
    usage = read_artifact(work_dir / "resource_monitor/usage.json", errors)
    session = usage.get("session")
    result["usage"] = usage
    environments = read_history(root / "logs/workers.jsonl", errors)
    result["worker_environment"] = environments[-1] if environments else {}
    if isinstance(session, str) and re.fullmatch(r"[0-9]+", session):
        result["monitor"] = read_artifact(
            work_dir / f"resource_monitor/session_{session}.json", errors
        )
    return result


def runtime_lifecycle(
    runtime: dict[str, Any], state: dict[str, Any]
) -> tuple[str, str] | None:
    """Goal completion needs agreement and confirmed cleanup, not an exit-code guess."""
    if not runtime:
        return None
    status = mapping(runtime.get("status"))
    receipt = mapping(runtime.get("receipt"))
    owner = mapping(runtime.get("owner"))
    release = mapping(runtime.get("release"))
    if (
        not status
        and not receipt
        and not state
        and mapping(runtime.get("ready")).get("ready") is True
    ):
        return (
            "prepared",
            f"Associated preparation recorded at {runtime['root']}; no run/progress receipt.",
        )
    phase = status.get("phase")
    reason = receipt.get("reason", status.get("reason"))
    origin = f"{runtime['root']}/status.json; observed {age_label(status.get('updated_utc'))}"
    if not isinstance(phase, str) or (
        reason is not None and not isinstance(reason, str)
    ):
        return "unknown", f"Malformed runtime phase/reason. {origin}"
    alive = live_identity(owner)
    if alive:
        owner_command = owner.get("command")
        if (
            not isinstance(owner_command, list)
            or str(Path(runtime["root"]) / "scripts/controller.py") not in owner_command
            or "run" not in owner_command
        ):
            return (
                "unknown",
                f"Live owner command does not belong to the associated runtime. {origin}",
            )
        if phase in {"completed", "failed", "stopped"} or release:
            return "unknown", f"Contradictory live owner and final receipt. {origin}"
        return (
            "running",
            f"Runtime owner boot/PID/start identity verified; reported phase {phase}. {origin}",
        )
    if not receipt:
        if phase in {"ready", "prepared"} and not state:
            return "prepared", f"Associated preparation recorded. {origin}"
        return (
            "unknown",
            f"Runtime owner not verified live; final receipt absent. {origin}",
        )
    ownership = mapping(receipt.get("ownership"))
    started = timestamp(receipt.get("started_utc"))
    updated = timestamp(status.get("updated_utc"))
    released = timestamp(release.get("released_utc"))
    head = mapping(
        mapping(state.get("metadata")).get("durable_generation_publication_v1")
    ).get("head_id")
    observed_head = mapping(receipt.get("last_observation")).get("head")
    chronological = started is not None and updated is not None and updated >= started
    if (
        not chronological
        or (head is not None and observed_head != head)
        or status.get("reason") != reason
    ):
        return (
            "unknown",
            f"Contradictory or stale runtime receipt/workspace evidence. {origin}",
        )
    cleaned = (
        ownership.get("confirmed") is True
        and ownership.get("survivors") == []
        and ownership.get("remaining_children") == []
        and ownership.get("post_helpers_reaped") is True
        and release.get("children") == []
        and bool(owner)
        and release.get("owner") == owner
        and released is not None
        and updated is not None
        and released >= updated
    )
    if not cleaned:
        return (
            "unknown",
            f"{reason}: shutdown/lease release incomplete or unconfirmed. {origin}",
        )
    monitor = mapping(runtime.get("monitor"))
    if monitor.get("shutdown_confirmed") is False:
        return "unknown", f"Contradictory canonical monitor shutdown evidence. {origin}"
    bad_exit = monitor.get("supervisor_exit_code") not in (
        None,
        0,
        130,
        143,
        -2,
        -15,
    ) or receipt.get("monitor_exit_code") not in (None, 0, 130, 143, -2, -15)
    if receipt.get("errors") or status.get("errors") or phase == "failed" or bad_exit:
        return "failed", f"{reason}: recorded failure; shutdown confirmed. {origin}"
    if (
        phase == "completed"
        and reason == "early_goal"
        and mapping(receipt.get("last_observation")).get("early_goal") is True
    ):
        return (
            "completed",
            f"early_goal achieved; owned children reaped and lease released (worker SIGINT/143 is normal cleanup). {origin}",
        )
    if reason == "early_goal":
        return (
            "unknown",
            f"Declared early_goal lacks matching goal/completed evidence. {origin}",
        )
    if reason in {"operator_cancel", "operator_SIGINT", "operator_SIGTERM"}:
        return (
            "stopped",
            f"Intentional operator stop ({reason}); shutdown confirmed. {origin}",
        )
    if reason in {
        "startup_failure",
        "runtime_failure",
        "service_failure_or_forced_shutdown",
    }:
        return "failed", f"{reason}; shutdown confirmed. {origin}"
    return "stopped", f"Bounded run ended: {reason}; shutdown confirmed. {origin}"


def worker_summary(
    work_dir: Path,
    name: str,
    stage: str,
    lifecycle: str,
    runtime: dict[str, Any],
    errors: list[str],
    reader: EvidenceReader,
) -> WorkerSummary:
    """Consume I6.1 v1 without importing its dispatcher or treating idle as dead."""
    record = read_artifact(
        work_dir / f"pipeline/worker_processes/{stage}.json", errors, limit=16 * 1024
    )
    if lifecycle in {"completed", "stopped", "failed"} and runtime:
        if record and live_identity(mapping(record.get("identity")), work_dir):
            return WorkerSummary(
                name,
                "unknown",
                "Contradictory live worker after recorded supervisor shutdown.",
            )
        return WorkerSummary(
            name,
            "stopped",
            "Associated supervisor confirms shutdown; last job outcome is separate.",
        )
    if not record:
        return legacy_worker_summary(work_dir, name, stage, errors, reader)
    identity = mapping(record.get("identity"))
    if (
        record.get("schema_version") == 1
        and not isinstance(record.get("schema_version"), bool)
        and record.get("worker") == stage
        and record.get("phase") == "stopped"
    ):
        pid = identity.get("pid")
        code = record.get("exit_code")
        observed = number(record.get("updated_unix_s"))
        stopped_evidence = (
            isinstance(pid, int)
            and not isinstance(pid, bool)
            and pid > 0
            and not (Path("/proc") / str(pid)).exists()
            and isinstance(code, int)
            and not isinstance(code, bool)
            and observed is not None
        )
        if stopped_evidence:
            state = "stopped" if code in (0, 130, 143) else "failed"
            return WorkerSummary(
                name,
                state,
                f"Recorded worker exit {code}; observed Unix seconds {observed}. Saved job {record.get('job_id')}; no process at recorded PID, no live-progress claim.",
            )
    monitor = {
        "pid": identity.get("monitor_pid"),
        "start_ticks": identity.get("monitor_start_ticks"),
        "boot": identity.get("boot"),
    }
    now = time.monotonic()
    heartbeat = number(record.get("heartbeat_monotonic_s"))
    started = number(record.get("job_started_monotonic_s"))
    valid = (
        isinstance(record.get("schema_version"), int)
        and not isinstance(record.get("schema_version"), bool)
        and record.get("schema_version") == 1
        and record.get("worker") == stage
        and identity.get("pid_namespace") == os.readlink("/proc/self/ns/pid")
        and live_identity(identity, work_dir)
        and live_identity(monitor, work_dir)
        and heartbeat is not None
        and 0 <= heartbeat <= now
        and started is not None
        and 0 <= started <= now
    )
    if not valid:
        return WorkerSummary(
            name,
            "unknown",
            "Unsupported, stale or mismatched worker/monitor identity or clock.",
        )
    phase = record.get("phase")
    if not isinstance(phase, str):
        return WorkerSummary(name, "unknown", "Malformed worker phase.")
    state = "unknown"
    if phase == "idle" and heartbeat is not None and now - heartbeat <= 5:
        state = "idle"
    elif phase in {"starting", "running", "cleanup"}:
        state = "working"
    observed = record.get("updated_unix_s")
    detail = f"Job {record.get('job_id', 'unknown')} · {phase}; observation Unix seconds {observed}. "
    if state == "working" and started is not None:
        detail += f"Active job age {now - started:.0f}s (not process lifetime). "
    detail += "Admitted source/model and precise waiting reason not reported by v1."
    return WorkerSummary(name, state, detail)


def legacy_worker_summary(
    work_dir: Path, name: str, stage: str, errors: list[str], reader: EvidenceReader
) -> WorkerSummary:
    """Reuse J1's bounded reader for current wait cursors and dispatched invocations."""
    from itertools import islice

    records = []
    directory = work_dir / "pipeline/performance/runtime"
    wait_stage = {"dataset_worker": "dataset", "training_worker": "training"}.get(
        stage, stage
    )
    for pattern in (f"worker-{stage}-*.json", f"wait-current-{wait_stage}-*.json"):
        paths = list(islice(directory.glob(pattern), 257))
        if len(paths) > 256:
            errors.append(
                f"{pattern}: exceeds J1 retention bound; worker state unavailable"
            )
            return WorkerSummary(
                name, "unknown", "Runtime observation retention bound exceeded."
            )
        for path in paths:
            record = reader.read(path, limit=16 * 1024)
            if record.get("schema_version") == 1:
                records.append(record)
    records.sort(
        key=lambda r: (
            number(
                r.get(
                    "last_observed_unix_s",
                    r.get("finished_unix_s", r.get("started_unix_s")),
                )
            )
            or 0
        ),
        reverse=True,
    )
    for record in records:
        if record.get("stage") == "wait" and record.get("finished_unix_s") is None:
            stream = str(record.get("stream_id", "")).split(":")
            observed = number(record.get("last_observed_unix_s"))
            if (
                len(stream) == 3
                and stream[1].isdigit()
                and observed is not None
                and 0 <= time.time() - observed <= 5
            ):
                identity = {
                    "pid": int(stream[1]),
                    "start_ticks": stream[2],
                    "boot_id": record.get("boot_id"),
                }
                if live_identity(identity, work_dir):
                    reason = waiting_reason(str(record.get("reason")), name)
                    return WorkerSummary(
                        name,
                        "waiting",
                        f"{reason}; source generation {record.get('generation')}; observed Unix seconds {observed}. Sampled decision, not claim authority.",
                    )
        if (
            record.get("stage") == "worker_lifetime"
            and record.get("evidence_status") == "incomplete"
        ):
            identity_parts = str(record.get("process_identity", "")).split(":")
            if len(identity_parts) == 3 and identity_parts[1].isdigit():
                identity = {
                    "pid": int(identity_parts[1]),
                    "start_ticks": identity_parts[2],
                    "boot_id": identity_parts[0],
                }
                if live_identity(identity, work_dir):
                    return WorkerSummary(
                        name,
                        "working",
                        f"J1 dispatch identity verified; observed Unix seconds {record.get('started_unix_s')}. Current internal phase/source/model not reported.",
                    )
    last = records[0] if records else {}
    reason = waiting_reason(
        str(last.get("reason", last.get("outcome", "unavailable"))), name
    )
    observed = last.get(
        "last_observed_unix_s",
        last.get("finished_unix_s", last.get("started_unix_s", "unknown")),
    )
    return WorkerSummary(
        name,
        "unknown",
        f"Historical J1 observation: {reason}; observed Unix seconds {observed}. No verified live worker.",
    )


def waiting_reason(reason: str, worker: str) -> str:
    """Translate recorded decisions without inventing a reason from pipeline order."""
    return {
        "no_input": {
            "Dataset": "Waiting for snapshot",
            "Training": "Waiting for dataset",
            "Reevaluation": "Waiting for training",
        }.get(worker, "Waiting for input"),
        "claim_contention": "Waiting for ownership",
        "ownership_contention": "Waiting for ownership",
        "learning_boundary": "Waiting at learning gate",
        "await_learning_resolution": "Waiting at learning gate for training/activation resolution",
        "pending_patch": "Waiting for pending patch",
        "activation": "Waiting for activation/full refresh",
    }.get(reason, reason.replace("_", " "))

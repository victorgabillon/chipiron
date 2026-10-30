"""Conservative local process evidence for opt-in learning-claim replacement.

TTL expiry fences a result, not a running computation. Worker sessions are made
by the existing monitored cluster launcher; this module neither signals nor locks.
"""

import os
import socket
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

from .generation_publication_io import object_digest, require
from .learning_schedule import validate_source, window
from .pipeline_artifacts import load_pipeline_stage_claim

if TYPE_CHECKING:
    from .bootstrap_paths import MorpionBootstrapPaths
    from .run_state import MorpionBootstrapRunState


def _process(pid: int) -> tuple[int, int, str] | None:
    """Return parent, session and start ticks; only absent processes are dead."""
    try:
        fields = (
            Path(f"/proc/{pid}/stat")
            .read_text(encoding="utf-8")
            .rsplit(") ", 1)[1]
            .split()
        )
    except FileNotFoundError:
        return None
    return int(fields[1]), int(fields[3]), fields[19]


def monitored_identity() -> dict[str, Any]:
    """Require the still-live monitor in this process's actual ancestry."""
    monitor = os.environ.get("MORPION_RESOURCE_MONITOR_PID", "")
    ticks = os.environ.get("MORPION_RESOURCE_MONITOR_START_TICKS", "")
    require(
        monitor.isdecimal() and bool(ticks),
        "Bounded learning requires the resource monitor.",
    )
    monitor_pid = int(monitor)
    observed = _process(monitor_pid)
    require(
        observed is not None and observed[2] == ticks,
        "Resource monitor identity changed.",
    )
    pid = os.getpid()
    current = _process(pid)
    assert current is not None
    ancestor = pid
    for _ in range(64):
        if ancestor == monitor_pid:
            break
        parent = _process(ancestor)
        require(
            parent is not None and parent[0] > 0, "Resource monitor is not an ancestor."
        )
        assert parent is not None
        ancestor = parent[0]
    require(ancestor == monitor_pid, "Resource monitor ancestry is unknown.")
    return {
        "host": object_digest({
            "name": socket.gethostname(),
            "machine": Path("/etc/machine-id").read_text(encoding="utf-8").strip(),
        }),
        "boot": Path("/proc/sys/kernel/random/boot_id")
        .read_text(encoding="utf-8")
        .strip(),
        "pid_namespace": os.readlink("/proc/self/ns/pid"),
        "pid": pid,
        "start_ticks": current[2],
        "session": current[1],
        "monitor_pid": monitor_pid,
        "monitor_start_ticks": ticks,
    }


def require_quiescent(previous: object, current: dict[str, Any]) -> None:
    """Unknown or surviving sessions prevent retry, even after coordinator death."""
    require(isinstance(previous, dict), "Missing prior worker quiescence evidence.")
    assert isinstance(previous, dict)
    require(
        set(previous) == set(current) and previous["host"] == current["host"],
        "Unknown prior worker host/identity.",
    )
    require(
        isinstance(previous["boot"], str) and bool(previous["boot"]),
        "Missing prior boot identity.",
    )
    require(
        str(UUID(previous["boot"])) == previous["boot"], "Invalid prior boot identity."
    )
    require(
        all(
            isinstance(previous[k], int)
            and not isinstance(previous[k], bool)
            and previous[k] > 1
            for k in ("pid", "session", "monitor_pid")
        ),
        "Invalid prior worker process identity.",
    )
    require(
        all(
            isinstance(previous[k], str) and previous[k].isdecimal()
            for k in ("start_ticks", "monitor_start_ticks")
        ),
        "Invalid prior worker start identity.",
    )
    require(
        isinstance(previous["pid_namespace"], str)
        and previous["pid_namespace"].startswith("pid:[")
        and previous["pid_namespace"].endswith("]")
        and previous["pid_namespace"][5:-1].isdecimal(),
        "Invalid prior PID namespace.",
    )
    if previous["boot"] != current["boot"]:
        return  # Same host, earlier boot: no process from that boot can survive.
    require(
        previous["pid_namespace"] == current["pid_namespace"],
        "Prior worker PID namespace is unknown.",
    )
    observed = _process(previous["pid"])
    require(
        observed is None or observed[2] != previous["start_ticks"],
        "Prior learning worker session is not quiescent.",
    )
    session = previous["session"]
    require(
        isinstance(session, int) and not isinstance(session, bool) and session > 1,
        "Unknown prior worker session.",
    )
    for path in Path("/proc").iterdir():
        if path.name.isdecimal():
            process = _process(int(path.name))
            require(
                process is None or process[1] != session,
                "Prior learning worker session is not quiescent.",
            )


def claim_metadata(
    paths: MorpionBootstrapPaths,
    state: MorpionBootstrapRunState,
    generation: int,
    claim_path: Path,
) -> dict[str, Any] | None:
    """Capture worker identity under ownership before creating/replacing a claim."""
    if window(state) is None:
        return None
    validate_source(paths, state, generation)
    current = monitored_identity()
    if claim_path.exists():
        previous = load_pipeline_stage_claim(claim_path)
        require_quiescent(previous.metadata.get("learning_worker"), current)
    return {"learning_worker": current}

"""Explicit time authorization under the existing native and Growth owners.

Only durable run-state references admit records. Scientific publication, candidate
continuation and the native sampler remain authoritative for their own work.
"""

import json
import math
import os
import re
import socket
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from .derived.provenance import atomic_json
from .generation_publication_io import (
    growth_receipt_ownership,
    object_digest,
    publication_records,
    publish_directory_if_absent,
    require,
    sync_artifact,
)
from .run_state import load_bootstrap_run_state, save_bootstrap_run_state
from .worker_result_io import workspace_epoch

if TYPE_CHECKING:
    from .bootstrap_paths import MorpionBootstrapPaths
    from .config import MorpionBootstrapConfig
    from .lifecycle import Metadata
    from .run_state import MorpionBootstrapRunState

KEY = "managed_sessions_v1"
MAX_GRANTS = 256
MAX_SESSIONS = 1024
MAX_RECORD = 16 * 1024
TIME_ENDS = {"maximum_elapsed_time", "supervisor_deadline"}


@dataclass(frozen=True)
class AdditionalTimeRequest:
    """Carry one preview-bound intent unchanged across process replacement."""

    seconds: int
    request_id: str
    predecessor: dict[str, Any] | None


def positive_seconds(value: str) -> int:
    """Accept decimal seconds, without bool/float/sign coercion."""
    require(
        bool(re.fullmatch(r"[0-9]{1,5}", value)),
        "Additional time must be decimal seconds (1..43200).",
    )
    result = int(value)
    require(1 <= result <= 43200, "Additional time must be in 1..43200 seconds.")
    return result


def _number(value: Any) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        "Invalid managed clock.",
    )
    try:
        result = float(value)
    except OverflowError as exc:
        message = "Managed clock overflow."
        raise ValueError(message) from exc
    require(
        math.isfinite(result) and 0 <= result <= 2**53 - 1,
        "Nonfinite/backward/overflow managed clock.",
    )
    return result


def _integer(value: Any, maximum: int) -> int:
    require(
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= maximum,
        "Invalid managed integer.",
    )
    return int(value)


def capability(work_dir: Path) -> dict[str, Any]:
    """Create an explicit capability only while preparing a new normal run."""
    return {
        "schema_version": 1,
        "enrollment_id": uuid4().hex,
        "work_dir": str(work_dir.resolve()),
    }


def enroll(
    config: MorpionBootstrapConfig, state: MorpionBootstrapRunState
) -> MorpionBootstrapRunState:
    """Attach enrollment to the existing fresh genesis transaction, never an old run."""
    marker = config.metadata.get(KEY)
    if marker is None:
        return state
    from .config import bootstrap_config_sha256

    raw_policy = config.metadata["continual_learning"]
    require(isinstance(raw_policy, dict), "Missing continual policy for managed time.")
    assert isinstance(raw_policy, dict)
    initial = raw_policy["max_elapsed_seconds"]
    require(
        _integer(initial, 43200) >= 1,
        "Managed genesis requires a bounded initial allowance.",
    )
    require(
        state.generation == 1 and KEY not in state.metadata,
        "Managed enrollment requires fresh genesis.",
    )
    require(isinstance(marker, dict), "Malformed managed capability.")
    assert isinstance(marker, dict)
    data: dict[str, Any] = {
        **marker,
        "identity": workspace_epoch(state),
        "config_sha256": bootstrap_config_sha256(config),
        "initial_seconds": initial,
        "authorized_seconds": initial,
        "consumed_seconds": 0.0,
        "grants": [],
        "sessions": [],
        "active": None,
    }
    return replace(state, metadata={**state.metadata, KEY: data})


def _reader() -> Metadata:
    from .lifecycle import Metadata

    return Metadata()


def _record_path(paths: MorpionBootstrapPaths, identifier: str) -> Path:
    require(
        bool(
            re.fullmatch(
                r"(?:grant-[a-f0-9]{64}|(?:launch|close)-[0-9]{1,32})", identifier
            )
        ),
        "Invalid managed object identity.",
    )
    return (
        paths.work_dir
        / "resource_monitor/managed_v1/objects"
        / identifier
        / "record.json"
    )


def _read(
    paths: MorpionBootstrapPaths, ref: dict[str, Any], reader: Metadata
) -> dict[str, Any]:
    require(set(ref) == {"id", "sha256"}, "Malformed managed reference.")
    path = _record_path(paths, ref["id"])
    require(path.stat().st_size <= MAX_RECORD, "Oversized managed descriptor.")
    value = reader.read(path)
    require(object_digest(value) == ref["sha256"], "Managed descriptor changed.")
    return value


def _put(
    paths: MorpionBootstrapPaths, identifier: str, value: dict[str, Any]
) -> dict[str, Any]:
    require(
        len(json.dumps(value, allow_nan=False).encode()) <= MAX_RECORD,
        "Managed descriptor bound exceeded.",
    )
    path = _record_path(paths, identifier)
    ref = {"id": identifier, "sha256": object_digest(value)}
    if path.exists():
        require(_read(paths, ref, _reader()) == value, "Conflicting managed object.")
        return ref
    from .lifecycle import _entries

    maximum_objects = MAX_GRANTS + 2 * MAX_SESSIONS
    if path.parent.parent.exists():
        _entries(path.parent.parent, maximum_objects)
        require(
            len(_entries(path.parent.parent, maximum_objects)) < maximum_objects,
            "Managed object bound exceeded.",
        )
    staging = path.parent.parent.parent / "staging"
    if staging.exists():
        require(
            len(_entries(staging, maximum_objects)) < maximum_objects,
            "Managed staging bound exceeded.",
        )
    stage = staging / uuid4().hex
    stage.mkdir(parents=True)
    atomic_json(stage / "record.json", value)
    sync_artifact(stage, paths.work_dir)
    path.parent.parent.mkdir(parents=True, exist_ok=True)
    publish_directory_if_absent(stage, path.parent, paths.work_dir)
    return ref


def _commit(
    paths: MorpionBootstrapPaths, state: MorpionBootstrapRunState, data: dict[str, Any]
) -> MorpionBootstrapRunState:
    require(
        len(json.dumps(data, allow_nan=False).encode()) <= 1024**2,
        "Managed metadata bound exceeded.",
    )
    state = replace(state, metadata={**state.metadata, KEY: data})
    save_bootstrap_run_state(state, paths.run_state_path, durable=True)
    return state


def validate(
    paths: MorpionBootstrapPaths,
    config: MorpionBootstrapConfig,
    state: MorpionBootstrapRunState,
    reader: Metadata | None = None,
) -> dict[str, Any] | None:
    """Join bounded operational history to the existing workspace/config authority."""
    marker, data = config.metadata.get(KEY), state.metadata.get(KEY)
    if marker is None and data is None:
        return None
    from .config import bootstrap_config_sha256

    require(
        isinstance(marker, dict) and isinstance(data, dict),
        "Missing managed enrollment; no historical adoption.",
    )
    assert isinstance(marker, dict) and isinstance(data, dict)
    require(
        marker
        == {
            "schema_version": 1,
            "enrollment_id": data.get("enrollment_id"),
            "work_dir": str(paths.work_dir.resolve()),
        }
        and isinstance(marker["schema_version"], int)
        and not isinstance(marker["schema_version"], bool)
        and bool(re.fullmatch(r"[a-f0-9]{32}", str(data["enrollment_id"]))),
        "Unsupported managed enrollment.",
    )
    require(
        set(data)
        == {
            "schema_version",
            "enrollment_id",
            "identity",
            "work_dir",
            "config_sha256",
            "initial_seconds",
            "authorized_seconds",
            "consumed_seconds",
            "grants",
            "sessions",
            "active",
        },
        "Unsupported managed state.",
    )
    require(
        data["schema_version"] == 1 and not isinstance(data["schema_version"], bool),
        "Unsupported managed state version.",
    )
    raw_policy = config.metadata["continual_learning"]
    require(isinstance(raw_policy, dict), "Missing continual policy for managed time.")
    assert isinstance(raw_policy, dict)
    initial = raw_policy["max_elapsed_seconds"]
    require(
        data["work_dir"] == marker["work_dir"],
        "Managed workspace root differs from fresh enrollment.",
    )
    require(
        data["identity"] == workspace_epoch(state)
        and data["config_sha256"]
        == bootstrap_config_sha256(config)
        == state.metadata["bootstrap_config_hash"],
        "Managed workspace/config identity changed.",
    )
    require(
        data["initial_seconds"] == initial
        and _integer(data["initial_seconds"], 43200) >= 1,
        "Managed initial allowance changed.",
    )
    require(
        isinstance(data["grants"], list)
        and len(data["grants"]) <= MAX_GRANTS
        and isinstance(data["sessions"], list)
        and len(data["sessions"]) <= MAX_SESSIONS,
        "Managed history bound exceeded.",
    )
    reader = reader or _reader()
    require(
        len(json.dumps(data, allow_nan=False).encode()) <= 1024**2,
        "Managed metadata bound exceeded.",
    )
    total, ids = data["initial_seconds"], set()
    grants: list[dict[str, Any]] = []
    launches: list[dict[str, Any]] = []
    closes: list[dict[str, Any]] = []
    consumed = 0.0
    for ref in data["sessions"]:
        close = _read(paths, ref, reader)
        require(
            close["enrollment_id"] == data["enrollment_id"]
            and close["validated"] is True
            and close["previous_close"]
            == (data["sessions"][len(closes) - 1] if closes else None),
            "Managed close chain differs.",
        )
        launch = _read(paths, close["launch"], reader)
        require(
            launch["previous_close"] == close["previous_close"]
            and launch["enrollment_id"] == data["enrollment_id"]
            and _number(launch["consumed_seconds"]) == consumed,
            "Managed launch/close linkage differs.",
        )
        require(
            _number(close["consumed_seconds"]) >= consumed,
            "Managed cumulative usage moved backward.",
        )
        consumed = _number(close["consumed_seconds"])
        _native(paths, close, reader)
        launches.append(launch)
        closes.append(close)
    for ref in data["grants"]:
        grant = _read(paths, ref, reader)
        require(
            grant["enrollment_id"] == data["enrollment_id"]
            and grant["request_id"] not in ids
            and grant["previous_authorized"] == total
            and grant["previous_close"] in data["sessions"],
            "Managed grant chain differs.",
        )
        predecessor = closes[data["sessions"].index(grant["previous_close"])]
        require(
            grant["consumed_seconds"] == predecessor["consumed_seconds"]
            and predecessor["reason"] in TIME_ENDS
            and total - grant["consumed_seconds"] < 1,
            "Grant did not follow a clean spent endpoint.",
        )
        grants.append(grant)
        ids.add(grant["request_id"])
        seconds = _integer(grant["seconds"], 43200)
        require(seconds > 0, "Empty managed grant.")
        total += seconds
        require(grant["authorized_seconds"] == total, "Managed grant total differs.")
    for index, launch in enumerate(launches):
        allowed = data["initial_seconds"]
        for grant in grants:
            if data["sessions"].index(grant["previous_close"]) < index:
                allowed = grant["authorized_seconds"]
        require(
            launch["authorized_seconds"] == allowed,
            "Launch used an unadmitted authorization total.",
        )
    require(
        _integer(data["authorized_seconds"], 2**53 - 1) == total
        and _number(data["consumed_seconds"]) == consumed,
        "Managed totals do not match committed records.",
    )
    if data["active"] is not None:
        launch = _read(paths, data["active"], reader)
        require(
            launch["enrollment_id"] == data["enrollment_id"]
            and launch["authorized_seconds"] == total
            and launch["consumed_seconds"] == consumed
            and launch["previous_close"] == (data["sessions"][-1] if closes else None),
            "Active managed launch differs.",
        )
    return dict(data)


def preserve_capability(
    paths: MorpionBootstrapPaths,
    state: MorpionBootstrapRunState,
    persisted: MorpionBootstrapConfig,
    requested: MorpionBootstrapConfig,
) -> MorpionBootstrapConfig:
    """Keep fresh enrollment when existing Growth reconstructs canonical args/config."""
    from .config import bootstrap_config_sha256

    data = validate(paths, persisted, state)
    if data is None:
        return requested
    requested = replace(
        requested, metadata={**requested.metadata, KEY: persisted.metadata[KEY]}
    )
    require(
        bootstrap_config_sha256(requested) == bootstrap_config_sha256(persisted),
        "Managed Growth cannot change the persisted scientific recipe or resource caps.",
    )
    return requested


def _local_identity() -> dict[str, Any]:
    fields = (
        Path("/proc/self/stat").read_text(encoding="utf-8").rsplit(") ", 1)[1].split()
    )
    return {
        "host": object_digest({
            "name": socket.gethostname(),
            "machine": Path("/etc/machine-id").read_text(encoding="utf-8").strip(),
        }),
        "boot": Path("/proc/sys/kernel/random/boot_id")
        .read_text(encoding="utf-8")
        .strip(),
        "pid_namespace": os.readlink("/proc/self/ns/pid"),
        "pid": os.getpid(),
        "start_ticks": fields[19],
    }


def _same_scope(identity: dict[str, Any]) -> None:
    current = _local_identity()
    require(
        all(identity[k] == current[k] for k in ("host", "boot", "pid_namespace")),
        "Managed renewal supports the same host/boot/PID namespace only.",
    )


def _quiescent(
    launch: dict[str, Any], receipt: dict[str, Any], *, closing: bool = False
) -> None:
    identity = launch["owner"]
    _same_scope(identity)
    current = _local_identity()
    path = Path(f"/proc/{identity['pid']}/stat")
    if path.exists() and not (closing and identity == current):
        fields = path.read_text(encoding="utf-8").rsplit(") ", 1)[1].split()
        require(
            fields[19] != identity["start_ticks"],
            "Prior managed monitor is still alive.",
        )
    groups = receipt.get("managed_owned_groups")
    require(isinstance(groups, list), "Missing managed group-reaping evidence.")
    assert isinstance(groups, list)
    for group in groups:
        require(
            isinstance(group, int) and not isinstance(group, bool) and group > 1,
            "Invalid owned group identity.",
        )
    for proc in Path("/proc").iterdir():
        if proc.name.isdecimal():
            try:
                fields = (
                    (proc / "stat")
                    .read_text(encoding="utf-8")
                    .rsplit(") ", 1)[1]
                    .split()
                )
            except FileNotFoundError:
                continue
            require(int(fields[2]) not in groups, "Prior managed worker group remains.")


def _native(
    paths: MorpionBootstrapPaths, close: dict[str, Any], reader: Metadata
) -> tuple[dict[str, Any], dict[str, Any]]:
    identifier = close["native_session"]
    require(
        bool(re.fullmatch(r"[0-9]{1,32}", identifier)),
        "Invalid native session identity.",
    )
    monitor = paths.work_dir / "resource_monitor"
    for kind in ("decision", "session"):
        require(
            reader.digest(monitor / f"{kind}_{identifier}.json")
            == close["native_sha256"][kind],
            "Native managed receipt changed.",
        )
    receipt = reader.read(monitor / f"session_{identifier}.json")
    decision = reader.read(monitor / f"decision_{identifier}.json")
    require(
        receipt["decision"] == decision
        and receipt["budget_elapsed_seconds"] == close["consumed_seconds"]
        and decision["reason"] == close["reason"] == receipt["reason"],
        "Managed native decision/cleanup linkage differs.",
    )
    require(
        _number(decision["budget_elapsed_seconds"])
        <= _number(close["consumed_seconds"]),
        "Managed decisive/cleanup clocks conflict.",
    )
    launch = _read(paths, close["launch"], reader)
    _same_scope(launch["owner"])
    decisive = _number(decision["monotonic_s"])
    require(
        _number(launch["started_monotonic_s"]) <= decisive <= time.monotonic(),
        "Invalid future/backward native decision clock.",
    )
    require(
        math.isclose(
            _number(decision["budget_elapsed_seconds"]),
            launch["consumed_seconds"] + decisive - launch["started_monotonic_s"],
            rel_tol=0,
            abs_tol=1e-7,
        ),
        "Native cumulative/monotonic clocks differ.",
    )
    link = {
        "enrollment_id": launch["enrollment_id"],
        "launch": close["launch"],
        "authorized_seconds": launch["authorized_seconds"],
    }
    require(
        receipt.get("managed_session")
        == decision.get("managed_session")
        == close["usage"].get("managed_session")
        == link,
        "Native managed execution binding differs.",
    )
    return receipt, decision


def inspect(
    paths: MorpionBootstrapPaths,
    config: MorpionBootstrapConfig,
    state: MorpionBootstrapRunState,
    reader: Metadata | None = None,
) -> dict[str, Any] | None:
    """Return bounded native-accounting status without reserving authority."""
    reader = reader or _reader()
    data = validate(paths, config, state, reader)
    if data is None:
        return None
    monitor = paths.work_dir / "resource_monitor"
    last, endpoint = (data["sessions"][-1] if data["sessions"] else None), None
    if last is not None:
        close = _read(paths, last, reader)
        receipt, decision = _native(paths, close, reader)
        _quiescent(_read(paths, close["launch"], reader), receipt)
        endpoint = {
            "session": close["native_session"],
            "reason": close["reason"],
            "decision": decision,
            "cleanup_elapsed_seconds": close["consumed_seconds"],
        }
    if data["active"] is not None:
        launch = _read(paths, data["active"], reader)
        status = "blocked/uncertain"
        close_path = _record_path(paths, "close-" + launch["native_session"])
        if close_path.exists():
            close = reader.read(close_path)
            require(
                close["validated"] is True
                and close["launch"] == data["active"]
                and close["previous_close"] == launch["previous_close"],
                "Interrupted close linkage differs.",
            )
            receipt, _ = _native(paths, close, reader)
            _quiescent(launch, receipt)
            status = "accounting-finalization-required"
        decision_path = monitor / f"decision_{launch['native_session']}.json"
        pending = reader.read(decision_path) if decision_path.is_file() else None
        return {
            **data,
            "status": status,
            "last_endpoint": endpoint,
            "launch": launch,
            "pending_native_decision": pending,
        }
    # Every attempt must join a positively validated committed close. Sidecars share IDs.
    expected = {_read(paths, ref, reader)["native_session"] for ref in data["sessions"]}
    if monitor.exists():
        from .lifecycle import _entries

        observed = {
            p.name.removeprefix("session_").split(".", 1)[0]
            for p in _entries(monitor, MAX_SESSIONS * 4 + 16)
            if p.name.startswith("session_")
        }
        decisions = {
            p.stem.removeprefix("decision_")
            for p in _entries(monitor, MAX_SESSIONS * 4 + 16)
            if p.name.startswith("decision_") and p.suffix == ".json"
        }
        require(
            observed == decisions == expected,
            "Incomplete/unreferenced managed native attempt.",
        )
    if last is not None:
        usage = reader.read(monitor / "usage.json")
        close = _read(paths, last, reader)
        require(usage == close["usage"], "Managed finalized usage differs.")
    else:
        require(
            not (monitor / "usage.json").exists(),
            "Unmanaged usage cannot enroll in managed time.",
        )
    remaining = max(0.0, data["authorized_seconds"] - data["consumed_seconds"])
    status = (
        "scientific-terminal"
        if state.metadata.get("experiment_termination") is not None
        else "ready"
        if remaining >= 1
        else "clean-time-ended"
    )
    if (
        status != "scientific-terminal"
        and endpoint is not None
        and endpoint["reason"] not in TIME_ENDS | {"supervisor_exit"}
    ):
        status = "blocked/uncertain"
    if (
        status not in {"blocked/uncertain", "scientific-terminal"}
        and data["grants"]
        and _read(paths, data["grants"][-1], reader)["previous_close"] == last
    ):
        status = "admitted-not-launched"
    return {**data, "status": status, "last_endpoint": endpoint}


def latest_request(
    paths: MorpionBootstrapPaths,
    summary: dict[str, Any],
    reader: Metadata | None = None,
) -> dict[str, Any]:
    """Resolve the current admitted request from its committed reference."""
    require(bool(summary["grants"]), "Missing committed time request.")
    return _read(paths, summary["grants"][-1], reader or _reader())


def request_has_closed_successor(
    summary: dict[str, Any], grant: dict[str, Any]
) -> bool:
    """Report whether a request already produced a closed native session."""
    predecessor = grant["previous_close"]
    sessions: list[dict[str, Any]] = summary["sessions"]
    require(predecessor in sessions, "Managed request predecessor is missing.")
    return len(sessions) > sessions.index(predecessor) + 1


def plan_additional_time(
    paths: MorpionBootstrapPaths,
    summary: dict[str, Any] | None,
    seconds: int,
    request_id: str | None = None,
) -> tuple[AdditionalTimeRequest, bool]:
    """Bind a finite intent to one exact prior clean stop; duplicate IDs add nothing."""
    require(
        summary is not None,
        "Additional time requires fresh managed enrollment; no historical adoption.",
    )
    assert summary is not None
    require(
        _integer(seconds, 43200) >= 1, "Additional time must be in 1..43200 seconds."
    )
    require(
        summary["status"]
        not in {
            "blocked/uncertain",
            "accounting-finalization-required",
            "scientific-terminal",
        },
        "Managed run is blocked/uncertain or scientifically terminal.",
    )
    predecessor = summary["sessions"][-1] if summary["sessions"] else None
    identifier = request_id or object_digest({
        "enrollment": summary["enrollment_id"],
        "previous_close": predecessor,
        "action": "add-time",
    })
    require(
        bool(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", identifier)),
        "Request ID must contain 1..64 ASCII letters, digits, underscore or hyphen.",
    )
    for ref in summary["grants"]:
        grant = _read(paths, ref, _reader())
        if grant["request_id"] == identifier:
            require(
                grant["seconds"] == seconds,
                "Conflicting reuse of additional-time request ID.",
            )
            return AdditionalTimeRequest(
                seconds, identifier, grant["previous_close"]
            ), True
    require(len(summary["grants"]) < MAX_GRANTS, "Managed grant bound exceeded.")
    endpoint = summary["last_endpoint"]
    require(
        endpoint is not None
        and endpoint["reason"] in TIME_ENDS
        and summary["authorized_seconds"] - summary["consumed_seconds"] < 1,
        "Additional time requires a confirmed clean spent time endpoint.",
    )
    require(
        summary["authorized_seconds"] + seconds - summary["consumed_seconds"] >= 1,
        "Grant cannot cover overshoot plus one second; increase the explicit increment.",
    )
    return AdditionalTimeRequest(seconds, identifier, predecessor), False


def _anchor(state: MorpionBootstrapRunState) -> dict[str, Any]:
    records = publication_records(state.metadata)
    return {
        "head": {k: records[-1][k] for k in ("id", "descriptor_sha256")},
        "accepted_refs_sha256": object_digest([
            {k: r[k] for k in ("id", "accepted_dataset", "accepted_training") if k in r}
            for r in records
        ]),
    }


def admit_additional_time_owned(
    paths: MorpionBootstrapPaths,
    config: MorpionBootstrapConfig,
    state: MorpionBootstrapRunState,
    request: AdditionalTimeRequest | None,
) -> tuple[MorpionBootstrapRunState, bool]:
    """Commit one grant under existing owners; return false for an already spent request."""
    summary = inspect(paths, config, state)
    require(
        summary is not None
        and summary["status"] not in {"blocked/uncertain", "scientific-terminal"},
        "Managed work is blocked/uncertain or terminal.",
    )
    assert summary is not None
    if request is None:
        return state, summary["authorized_seconds"] - summary["consumed_seconds"] >= 1
    checked, duplicate = plan_additional_time(
        paths, summary, request.seconds, request.request_id
    )
    require(
        checked == request,
        "Additional-time predecessor changed before admission; preview again.",
    )
    if duplicate:
        # Old requests never launch work belonging to a later grant.
        latest = _read(paths, summary["grants"][-1], _reader())
        return state, (
            latest["request_id"] == request.request_id
            and not request_has_closed_successor(summary, latest)
            and summary["authorized_seconds"] - summary["consumed_seconds"] >= 1
        )
    data = {k: summary[k] for k in state.metadata[KEY]}
    value = {
        "enrollment_id": data["enrollment_id"],
        "request_id": request.request_id,
        "seconds": request.seconds,
        "previous_close": request.predecessor,
        "previous_authorized": data["authorized_seconds"],
        "consumed_seconds": data["consumed_seconds"],
        "authorized_seconds": data["authorized_seconds"] + request.seconds,
        **_anchor(state),
    }
    ref = _put(paths, "grant-" + object_digest(request.request_id), value)
    return _commit(
        paths,
        state,
        {
            **data,
            "grants": [*data["grants"], ref],
            "authorized_seconds": value["authorized_seconds"],
        },
    ), True


def begin_owned(
    paths: MorpionBootstrapPaths,
    config: MorpionBootstrapConfig,
    identifier: str,
    started: float,
    supervisor_limit: int,
) -> dict[str, Any]:
    """Durably reserve one launch before spawn, under monitor then Growth ownership."""
    with growth_receipt_ownership(paths, create=False):
        state = load_bootstrap_run_state(paths.run_state_path)
        data = validate(paths, config, state)
        require(
            data is not None
            and data["active"] is None
            and len(data["sessions"]) < MAX_SESSIONS,
            "Managed launch is active/uncertain or history full.",
        )
        assert data is not None
        require(_number(started) <= time.monotonic(), "Future managed launch clock.")
        value = {
            "enrollment_id": data["enrollment_id"],
            "native_session": identifier,
            "owner": _local_identity(),
            "started_monotonic_s": started,
            "authorized_seconds": data["authorized_seconds"],
            "consumed_seconds": data["consumed_seconds"],
            "previous_close": data["sessions"][-1] if data["sessions"] else None,
            "supervisor_limit_seconds": supervisor_limit,
            **_anchor(state),
        }
        ref = _put(paths, "launch-" + identifier, value)
        _commit(paths, state, {**data, "active": ref})
        return {
            "enrollment_id": data["enrollment_id"],
            "launch": ref,
            "authorized_seconds": data["authorized_seconds"],
        }


def finalize_managed_session_owned(
    paths: MorpionBootstrapPaths, config: MorpionBootstrapConfig, identifier: str
) -> None:
    """Publish positive completion only AFTER every native success/failure check passed."""
    with growth_receipt_ownership(paths, create=False):
        state = load_bootstrap_run_state(paths.run_state_path)
        data = validate(paths, config, state)
        require(
            data is not None and data["active"] is not None,
            "Missing managed launch during close.",
        )
        assert data is not None and isinstance(data["active"], dict)
        reader = _reader()
        launch = _read(paths, data["active"], reader)
        require(
            launch["native_session"] == identifier
            and launch["owner"] == _local_identity(),
            "Managed close belongs to another native owner.",
        )
        monitor = paths.work_dir / "resource_monitor"
        receipt = reader.read(monitor / f"session_{identifier}.json")
        decision = reader.read(monitor / f"decision_{identifier}.json")
        usage = reader.read(monitor / "usage.json")
        require(
            receipt["reason"] == decision["reason"] == usage["reason"]
            and receipt["decision"] == decision
            and usage["session"] == identifier
            and receipt["budget_elapsed_seconds"] == usage["elapsed_seconds"]
            and receipt["shutdown_confirmed"] is True
            and receipt["forced_group_cleanup"] is False
            and receipt["shutdown_grace_expired"] is False
            and isinstance(receipt["supervisor_exit_code"], int)
            and not isinstance(receipt["supervisor_exit_code"], bool)
            and receipt["supervisor_exit_code"] in (0, 143)
            and decision["worker_failures"] == []
            and decision["control_failure"] is None,
            "Managed native completion is not clean.",
        )
        _quiescent(launch, receipt, closing=True)
        consumed = _number(usage["elapsed_seconds"])
        require(
            data["consumed_seconds"]
            <= _number(decision["budget_elapsed_seconds"])
            <= consumed,
            "Managed clock moved backward.",
        )
        if decision["reason"] == "maximum_elapsed_time":
            require(
                decision["budget_elapsed_seconds"] >= data["authorized_seconds"],
                "Native elapsed endpoint predates its admitted deadline.",
            )
        if decision["reason"] == "supervisor_deadline":
            require(
                isinstance(decision.get("supervisor_shutdown"), dict),
                "Missing authenticated early supervisor deadline.",
            )
        close = {
            "enrollment_id": data["enrollment_id"],
            "launch": data["active"],
            "native_session": identifier,
            "previous_close": launch["previous_close"],
            "consumed_seconds": consumed,
            "reason": decision["reason"],
            "validated": True,
            "usage": usage,
            "native_sha256": {
                kind: reader.digest(monitor / f"{kind}_{identifier}.json")
                for kind in ("decision", "session")
            },
            **_anchor(state),
        }
        ref = _put(paths, "close-" + identifier, close)
        _commit(
            paths,
            state,
            {
                **data,
                "sessions": [*data["sessions"], ref],
                "consumed_seconds": consumed,
                "active": None,
            },
        )


def recover_close_owned(
    paths: MorpionBootstrapPaths, config: MorpionBootstrapConfig
) -> None:
    """Finish only a positively validated close whose final root commit was interrupted."""
    with growth_receipt_ownership(paths, create=False):
        state = load_bootstrap_run_state(paths.run_state_path)
        data = validate(paths, config, state)
        if data is None or data["active"] is None:
            return
        reader = _reader()
        launch = _read(paths, data["active"], reader)
        path = _record_path(paths, "close-" + launch["native_session"])
        require(
            path.is_file(),
            "Managed launch has no positively validated close; explicit recovery required.",
        )
        close = reader.read(path)
        ref = {
            "id": "close-" + launch["native_session"],
            "sha256": object_digest(close),
        }
        require(
            close["validated"] is True
            and close["launch"] == data["active"]
            and close["previous_close"] == launch["previous_close"]
            and _anchor(state)
            == {k: close[k] for k in ("head", "accepted_refs_sha256")},
            "Interrupted managed close differs from authority.",
        )
        receipt, _ = _native(paths, close, reader)
        _quiescent(launch, receipt)
        require(
            close["usage"]
            == reader.read(paths.work_dir / "resource_monitor/usage.json")
            and _number(close["consumed_seconds"]) >= data["consumed_seconds"],
            "Incomplete managed close usage.",
        )
        _commit(
            paths,
            state,
            {
                **data,
                "sessions": [*data["sessions"], ref],
                "consumed_seconds": close["consumed_seconds"],
                "active": None,
            },
        )


def require_managed_execution(
    paths: MorpionBootstrapPaths, state: MorpionBootstrapRunState | None = None
) -> None:
    """Fence managed writers with the actual live monitor ancestry and active launch."""
    from .config import load_bootstrap_config

    if not paths.bootstrap_config_path.is_file():
        return
    config = load_bootstrap_config(paths.bootstrap_config_path)
    if KEY not in config.metadata and (state is None or KEY not in state.metadata):
        return
    state = state or load_bootstrap_run_state(paths.run_state_path)
    data = validate(paths, config, state)
    require(
        data is not None
        and data["active"] is not None
        and state.metadata.get("experiment_termination") is None,
        "Managed scientific work requires an active nonterminal native session.",
    )
    assert data is not None and isinstance(data["active"], dict)
    from .persistent_worker_state import monitored_identity

    identity = monitored_identity()
    launch = _read(paths, data["active"], _reader())
    owner = launch["owner"]
    _same_scope(owner)
    require(
        identity["monitor_pid"] == owner["pid"]
        and identity["monitor_start_ticks"] == owner["start_ticks"]
        and os.environ.get("MORPION_RESOURCE_MONITOR_SESSION")
        == launch["native_session"],
        "Managed writer does not belong to the active native session.",
    )


def status_summary(paths: MorpionBootstrapPaths) -> dict[str, Any] | None:
    """Read normal managed status; never associate or alter an experiment wrapper."""
    from .config import bootstrap_config_from_dict
    from .run_state import _run_state_from_dict

    reader = _reader()
    if not paths.bootstrap_config_path.is_file():
        return None
    try:
        config = bootstrap_config_from_dict(reader.read(paths.bootstrap_config_path))
        state = _run_state_from_dict(reader.read(paths.run_state_path))
        if KEY not in config.metadata and KEY not in state.metadata:
            return None
        summary = inspect(paths, config, state, reader)
        assert summary is not None
        if summary["active"] is not None and summary["status"] == "blocked/uncertain":
            import fcntl

            launch = summary["launch"]
            owner = launch["owner"]
            _same_scope(owner)
            proc = Path(f"/proc/{owner['pid']}/stat")
            if (
                proc.exists()
                and proc.read_text(encoding="utf-8").rsplit(") ", 1)[1].split()[19]
                == owner["start_ticks"]
            ):
                with (paths.work_dir / "resource_monitor/owner.lock").open(
                    "rb"
                ) as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        pending = summary.get("pending_native_decision")
                        summary["status"] = (
                            "running"
                            if pending is None
                            else "blocked/uncertain"
                            if pending.get("worker_failures")
                            or pending.get("control_failure")
                            else "stopping"
                        )
                    else:
                        fcntl.flock(lock, fcntl.LOCK_UN)
        summary["remaining_seconds"] = max(
            0.0, summary["authorized_seconds"] - summary["consumed_seconds"]
        )
        summary["accounting"] = (
            "finalized native usage; live work is not refunded or finalized here"
        )
    except (ValueError, KeyError, TypeError, OSError) as exc:
        return {"status": "blocked/uncertain", "error": str(exc)}
    return summary

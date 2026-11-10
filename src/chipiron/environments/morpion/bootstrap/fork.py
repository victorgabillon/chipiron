"""Quiescent full-state fork: existing owners, independent copies, fresh genesis."""

import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .bootstrap_paths import MorpionBootstrapPaths
from .config import bootstrap_config_from_dict, bootstrap_config_to_dict
from .derived.provenance import atomic_json, inside
from .fork_origin import KEY
from .generation_publication_io import (
    artifact_digest,
    capsule_lease,
    capsule_path,
    file_digest,
    growth_receipt_ownership,
    object_digest,
    publication_records,
    publish_directory_if_absent,
    read_committed_descriptor,
    require,
    sync_artifact,
    validate_capsule,
)
from .learning_policy import policy_from_metadata, validate_continual_context
from .learning_schedule import schedule_from_metadata, window
from .lifecycle import _existing_lock, _references, _source_identity
from .lifecycle_metadata import Metadata
from .lifecycle_metadata import entries as _entries
from .managed_sessions import KEY as MANAGED
from .managed_sessions import _local_identity, inspect
from .run_state import _run_state_from_dict, load_bootstrap_run_state
from .worker_result_io import workspace_epoch

if TYPE_CHECKING:
    from collections.abc import Iterator


MAX_FILES = 16384


def inventory(root: Path) -> list[Path]:
    """Bound traversal before hashing/copy; reject links and special files."""
    files: list[Path] = []
    pending = [root]
    entries = 0
    while pending:
        path = pending.pop()
        require(not path.is_symlink(), "Symlinked fork closure.")
        entries += 1
        require(entries <= MAX_FILES, "Fork file-count ceiling exceeded.")
        if path.is_dir():
            pending.extend(_entries(path, MAX_FILES - entries))
        else:
            require(path.is_file(), "Nonregular fork input.")
            files.append(path)
    return sorted(files)


def _inventory_identity(root: Path) -> tuple[str, set[tuple[int, int]]]:
    files = inventory(root)
    counts = Counter((p.stat().st_dev, p.stat().st_ino) for p in files)
    for p in files:
        s = p.stat()
        require(
            counts[(s.st_dev, s.st_ino)] == s.st_nlink,
            "Unknown external fork hardlink ownership.",
        )
    return object_digest([
        [
            str(p.relative_to(root)),
            p.stat().st_size,
            p.stat().st_mtime_ns,
            p.stat().st_dev,
            p.stat().st_ino,
        ]
        for p in files
    ]), set(counts)


def _settled(paths: MorpionBootstrapPaths, state: Any, reader: Metadata) -> None:
    require(
        not paths.launcher_pid_path.exists(),
        "Launcher registration requires recovery before fork.",
    )
    require(
        not paths.pipeline_reevaluation_patch_path.exists(),
        "Pending Reevaluation patch prevents fork.",
    )
    data = window(state)
    require(
        data is None or data["active"] is None,
        "Reserved learning window prevents fork.",
    )
    records = publication_records(state.metadata)
    for entry in _entries(paths.pipeline_dir):
        if entry.is_dir() and entry.name.startswith("generation_"):
            require(
                not any(p.name.endswith("_claim.json") for p in _entries(entry)),
                "Any outstanding claim prevents fork, regardless of TTL.",
            )
    for namespace, accepted in (
        ("generation_publications_v3", None),
        ("dataset_publications_v1", "accepted_dataset"),
        ("training_publications_v1", "accepted_training"),
    ):
        root = paths.work_dir / namespace
        for name in ("staging", "trash"):
            require(
                not (root / name).exists() or not _entries(root / name),
                "Unresolved publication staging prevents fork.",
            )
        expected = (
            {r["id"] for r in records}
            if accepted is None
            else {r[accepted]["attempt_id"] for r in records if accepted in r}
        )
        actual = (
            {p.name for p in _entries(root / "objects")}
            if (root / "objects").exists()
            else set()
        )
        require(
            actual == expected,
            "Unaccepted or missing publication objects prevent fork.",
        )
    automatic = paths.pipeline_dir / "automatic_activation"
    if automatic.exists():
        from .model_activation import activation_completion

        completed = {
            r.get("activation_request_sha256"): r
            for r in records
            if "activation_request_sha256" in r
        }
        chains: dict[int, list[dict[str, Any]]] = {}
        for path in _entries(automatic):
            require(
                path.is_file() and path.suffix == ".json", "Unknown activation control."
            )
            request = reader.read(path)
            chains.setdefault(request["training_generation"], []).append(request)
        for requests in chains.values():
            matches = [
                completed[object_digest(r)]
                for r in requests
                if object_digest(r) in completed
            ]
            require(bool(matches), "Pending automatic activation prevents fork.")
            for record in matches:
                # Existing completion validation reads and hashes this descriptor.
                seal = inside(paths.work_dir, record["descriptor_path"])
                reader.data(seal)
                reader.data(seal)
                activation_completion(paths, state, record)


def _capture(
    root: Path, target: Path, seconds: int, nodes: int
) -> tuple[dict[str, Any], Any, Any]:
    """Caller holds existing source owners; adopt one complete metadata cut."""
    reader = Metadata()
    paths = MorpionBootstrapPaths.from_work_dir(root)
    config = bootstrap_config_from_dict(reader.read(paths.bootstrap_config_path))
    state = _run_state_from_dict(reader.read(paths.run_state_path))
    policy = policy_from_metadata(config.metadata)
    require(
        policy is not None
        and policy.training_policy == "continue_from_previous_accepted_weights"
        and policy.optimizer_state_policy == "continue_adam",
        "Unsupported fork continuation policy.",
    )
    assert policy is not None
    validate_continual_context(
        paths, state, policy, schedule_from_metadata(config.metadata)
    )
    from .checkpoint_patch_receipts import receipt_ledger
    from .execution_policy import policy_from_metadata as execution_policy

    receipt_ledger(state)
    require(
        execution_policy(config.metadata) == execution_policy(state.metadata),
        "Fork execution profile binding changed.",
    )
    require(
        MANAGED in config.metadata
        and state.metadata.get("durable_training") == {"schema_version": 1}
        and state.metadata.get("fenced_dataset") == {"schema_version": 1},
        "Only normal managed current scientific runs can fork.",
    )
    records = publication_records(state.metadata)
    require(
        0 < len(records) <= 1024
        and all(r.get("publication_version") == 3 for r in records),
        "Closed publication-v3 fork required.",
    )
    head = records[-1]
    require(
        (
            state.generation,
            state.latest_runtime_checkpoint_path,
            state.latest_tree_snapshot_path,
        )
        == (head["generation"], head["checkpoint_path"], head["tree_snapshot_path"]),
        "Fork source head changed.",
    )
    require("retired_payload_v1" not in head, "Retired fork head is unavailable.")
    descriptor_path = inside(root, head["descriptor_path"])
    require(
        reader.digest(descriptor_path) == head["descriptor_sha256"],
        "Fork source seal changed.",
    )
    reader.data(descriptor_path)
    reader.data(descriptor_path)
    descriptor = read_committed_descriptor(root, head)
    require(
        len(descriptor["files"]) <= 4096 and descriptor.get("evaluator_contract") == 1,
        "Unsupported fork head inventory/evaluator.",
    )
    _source_identity(head["scientific_identity"]["producer"]["identity"], reader)
    summary = inspect(paths, config, state, reader)
    require(
        summary is not None
        and summary["active"] is None
        and summary["last_endpoint"] is not None
        and summary["status"] in {"ready", "clean-time-ended", "scientific-terminal"},
        "Fork requires a positively closed, fully settled native parent.",
    )
    assert summary is not None
    endpoint = summary["last_endpoint"]
    assert isinstance(endpoint, dict)
    require(
        endpoint["reason"]
        in {
            "maximum_elapsed_time",
            "supervisor_deadline",
            "supervisor_exit",
            "maximum_tree_nodes",
        },
        "Unsupported failed/resource/goal fork endpoint.",
    )
    terminal = state.metadata.get("experiment_termination")
    require(
        terminal is None or terminal.get("reason") == "maximum_tree_nodes",
        "Unknown scientific goal completion cannot fork.",
    )
    require(
        nodes > state.tree_size_at_last_save,
        "New absolute node budget must exceed the retained tree.",
    )
    _settled(paths, state, reader)
    parents, datasets = _references(root, state, reader)
    from .fork_origin import seeds

    imported = seeds(paths, state, verify=False)
    names = list(config.evaluators.evaluators)
    require(
        len(names) <= 128
        and all(re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}", n) for n in names),
        "Unsupported candidate names for independent fork.",
    )
    parents = {n: parents.get(n, imported.get(n)) for n in names}
    capsule = capsule_path(root, head["id"], 3)
    pin = next(
        p
        for p in descriptor["model_pins"]
        if p["origin"] == descriptor["active_evaluator"]["path"]
        and p["sha256"] == descriptor["active_evaluator"]["sha256"]
    )
    active = {
        "identity": descriptor["active_evaluator"],
        "bundle": str((capsule / pin["path"]).relative_to(root)),
    }
    selected = {
        str((capsule / f["path"]).relative_to(root)) for f in descriptor["files"]
    }
    evidence = {"run_state.json", "bootstrap_config.json", head["descriptor_path"]}
    if KEY in config.metadata:
        origin_ref: Any = config.metadata[KEY]
        evidence.add(origin_ref["path"])
    for r in records:
        for kind, namespace in (
            ("accepted_dataset", "dataset_publications_v1"),
            ("accepted_training", "training_publications_v1"),
        ):
            if kind in r:
                evidence.add(
                    f"{namespace}/objects/{r[kind]['attempt_id']}/proposal.json"
                )
    last = summary["sessions"][-1]
    close_path = (
        root / "resource_monitor/managed_v1/objects" / last["id"] / "record.json"
    )
    close = reader.read(close_path)
    launch = close["launch"]
    evidence.update({
        str(close_path.relative_to(root)),
        f"resource_monitor/managed_v1/objects/{launch['id']}/record.json",
        f"resource_monitor/session_{close['native_session']}.json",
        f"resource_monitor/decision_{close['native_session']}.json",
        "resource_monitor/usage.json",
    })
    for parent in parents.values():
        if parent is not None:
            selected.update(
                str(p.relative_to(root))
                for p in inventory(inside(root, parent["model"]["path"]))
            )
    selected.update(evidence)
    require(len(selected) <= MAX_FILES, "Fork selected closure exceeds file ceiling.")
    sizes = {name: inside(root, name).stat().st_size for name in sorted(selected)}
    total = sum(sizes.values())
    retained = [
        p for p in _entries(target.parent) if p.name.startswith(f".{target.name}.fork-")
    ]
    require(len(retained) <= 2, "Fork retained-attempt ceiling exceeded.")
    retained_bytes = sum(
        p.stat().st_size for directory in retained for p in inventory(directory)
    )
    output_ceiling = 4 * total + 16 * 2**20
    require(
        total > 0
        and output_ceiling + retained_bytes < policy.max_workspace_mb * 2**20
        and 2 * total <= policy.max_scratch_mb * 2**20,
        "Insufficient inherited workspace/scratch allowance for fork coexistence.",
    )
    require(
        shutil.disk_usage(target.parent).free - output_ceiling
        >= policy.min_free_disk_mb * 2**20,
        "Insufficient fork output/free-space reservation.",
    )
    inventory_sha, _ = _inventory_identity(root)
    cut = {
        "root": str(root),
        "run_state_sha256": reader.digest(paths.run_state_path),
        "config_sha256": reader.digest(paths.bootstrap_config_path),
        "identity": workspace_epoch(state),
        "head": {k: head[k] for k in ("id", "descriptor_sha256")},
        "parents": parents,
        "dataset_origins": datasets,
        "endpoint": endpoint,
    }
    request = {
        "parent": cut,
        "target": str(target),
        "seconds": seconds,
        "nodes": nodes,
        "source_inventory_sha256": inventory_sha,
    }
    plan = {
        "action": "fork",
        "plan_sha256": object_digest(request),
        **request,
        "incumbent": active,
        "files": sizes,
        "evidence_files": sorted(evidence),
        "copy_bytes": total,
        "estimated_source_read_bytes": 8 * total + 64 * 2**20,
        "maximum_attempts": 2,
        "maximum_cumulative_source_read_bytes": 2 * (8 * total + 64 * 2**20),
        "reserved_output_bytes": output_ceiling,
        "dataset_policy": "regenerate_local",
        "preparation_limits": {
            "seconds": policy.worker_max_seconds,
            "rss_mb": policy.max_group_rss_mb,
            "workspace_mb": policy.max_workspace_mb,
            "scratch_mb": policy.max_scratch_mb,
            "free_mb": policy.min_free_disk_mb,
            "available_ram_mb": policy.min_available_ram_mb,
        },
    }
    require(
        len(json.dumps(plan).encode()) <= 4 * 2**20,
        "Fork plan metadata bound exceeded.",
    )
    return plan, config, state


def _arguments(
    parent: Path, target: Path, seconds: int, nodes: int
) -> tuple[Path, Path]:
    require(
        isinstance(seconds, int)
        and not isinstance(seconds, bool)
        and 1 <= seconds <= 43200
        and isinstance(nodes, int)
        and not isinstance(nodes, bool)
        and nodes > 0,
        "Fork requires finite positive explicit time/node budgets.",
    )
    source, destination = parent.expanduser().resolve(), target.expanduser().resolve()
    require(
        source != destination
        and source not in destination.parents
        and destination not in source.parents
        and not target.is_symlink(),
        "Fork source/destination cannot alias or nest.",
    )
    require(
        source.is_dir() and destination.parent.is_dir(),
        "Existing source and destination parent required.",
    )
    return source, destination


@contextmanager
def source_cut(
    parent: Path, target: Path, seconds: int, nodes: int
) -> Iterator[tuple[dict[str, Any], Any, Any]]:
    """No source write or repair; keep owners and existing lifetime pins throughout."""
    root, destination = _arguments(parent, target, seconds, nodes)
    paths = MorpionBootstrapPaths.from_work_dir(root)
    with (
        _existing_lock(root / "resource_monitor/owner.lock"),
        growth_receipt_ownership(paths, create=False),
        ExitStack() as pins,
    ):
        plan, config, state = _capture(root, destination, seconds, nodes)
        for record in sorted(
            publication_records(state.metadata), key=lambda r: r["id"]
        ):
            pins.enter_context(
                capsule_lease(
                    root, record["id"], version=3, create=False, nonblocking=True
                )
            )
        require(
            file_digest(paths.run_state_path) == plan["parent"]["run_state_sha256"],
            "Fork authority changed while adopting pins.",
        )
        yield plan, config, state


def plan_fork(parent: Path, target: Path, seconds: int, nodes: int) -> dict[str, Any]:
    """Return bounded read-only preview; this is no copy/launch reservation."""
    require(not target.exists(), "Fork destination already exists.")
    with source_cut(parent, target, seconds, nodes) as (plan, _, _):
        return plan


class PreparationBudget:
    """Finite one-attempt accounting, sampled through owned heavy initialization."""

    def __init__(
        self, plan: dict[str, Any], stage: Path, prior: dict[str, Any] | None = None
    ) -> None:
        """Start an attempt under the plan's inherited finite limits."""
        self.plan, self.stage = plan, stage
        now = time.monotonic()
        self.started = now if prior is None else prior["started_monotonic_s"]
        require(
            type(self.started) in {int, float}
            and math.isfinite(self.started)
            and 0 <= self.started <= now,
            "Invalid fork preparation clock.",
        )
        require(
            prior is None or prior["owner"]["boot"] == _local_identity()["boot"],
            "Cross-boot preparation retry is unsupported.",
        )
        # An interrupted attempt retains its entire read reservation. A second
        # attempt is the only permitted retry, within the original wall deadline.
        self.prior_read_reservation = (
            0 if prior is None else plan["estimated_source_read_bytes"]
        )
        self.read_bytes = 0
        self.written_bytes = 0

    def check(self, *, read: int = 0, written: int = 0, extra_rss: int = 0) -> None:
        """Charge IO and enforce live time, process, storage and reserve bounds."""
        import psutil

        self.read_bytes += read
        self.written_bytes += written
        limits = self.plan["preparation_limits"]
        require(
            time.monotonic() - self.started < limits["seconds"],
            "Fork preparation wall deadline reached.",
        )
        require(
            self.read_bytes <= self.plan["estimated_source_read_bytes"],
            "Fork original read ceiling reached.",
        )
        require(
            self.written_bytes <= self.plan["reserved_output_bytes"],
            "Fork copy write ceiling reached.",
        )
        require(
            psutil.Process().memory_info().rss + extra_rss <= limits["rss_mb"] * 2**20,
            "Fork preparation RSS ceiling reached.",
        )
        require(
            psutil.virtual_memory().available >= limits["available_ram_mb"] * 2**20
            and shutil.disk_usage(self.stage.parent).free >= limits["free_mb"] * 2**20,
            "Insufficient fork RAM/disk reserve.",
        )
        if self.stage.exists():
            total = sum(p.stat().st_size for p in inventory(self.stage))
            retained = [
                p
                for p in _entries(self.stage.parent)
                if p != self.stage
                and p.name.startswith(f".{Path(self.plan['target']).name}.fork-")
            ]
            combined = total + sum(
                p.stat().st_size for directory in retained for p in inventory(directory)
            )
            require(
                combined <= limits["workspace_mb"] * 2**20,
                "Fork attempts exceed aggregate inherited workspace ceiling.",
            )
            require(
                total <= self.plan["reserved_output_bytes"]
                and total <= limits["workspace_mb"] * 2**20,
                "Fork staged output ceiling reached.",
            )
            scratch = self.stage / "input"
            require(
                not scratch.exists()
                or sum(p.stat().st_size for p in inventory(scratch))
                <= limits["scratch_mb"] * 2**20,
                "Fork scratch ceiling reached.",
            )


def _copy(source: Path, target: Path, budget: PreparationBudget) -> None:
    """Write a new inode and verify exact bytes and stable original identity."""
    require(
        not target.exists() and not source.is_symlink(), "Fork copy collision/link."
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with source.open("rb") as src, target.open("xb") as dst:
        before = os.fstat(src.fileno())
        count = 0
        while chunk := src.read(2**20):
            count += len(chunk)
            require(count <= before.st_size, "Fork source grew during copy.")
            budget.check(read=len(chunk), written=len(chunk))
            digest.update(chunk)
            dst.write(chunk)
        dst.flush()
        os.fsync(dst.fileno())
        after = os.fstat(src.fileno())
    require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        "Original fork file changed during copy.",
    )
    require(
        digest.hexdigest() == file_digest(target)
        and (before.st_dev, before.st_ino)
        != (target.stat().st_dev, target.stat().st_ino),
        "Fork copy bytes/inode verification failed.",
    )


def _copy_closure(
    root: Path, stage: Path, plan: dict[str, Any], state: Any, budget: PreparationBudget
) -> tuple[Path, Path, dict[str, Any]]:
    copied = stage / "input/original"
    for name, size in plan["files"].items():
        source = inside(root, name)
        require(source.stat().st_size == size, "Fork input size changed after plan.")
        _copy(source, copied / name, budget)
    # Validate against original committed commitments only after fully independent copies.
    budget.check(read=plan["copy_bytes"])
    validate_capsule(root, publication_records(state.metadata)[-1])
    seeds: dict[str, Any] = {}
    for name, parent in plan["parent"]["parents"].items():
        if parent is None:
            seeds[name] = None
            continue
        source = copied / parent["model"]["path"]
        require(
            artifact_digest(source) == parent["model"]["sha256"],
            "Imported accepted candidate closure differs.",
        )
        destination = stage / f"genesis/fork/seeds/{name}"
        for p in inventory(source):
            _copy(p, destination / p.relative_to(source), budget)
        seeds[name] = {
            "kind": "fork_seed",
            "model": {
                "path": str(destination.relative_to(stage)),
                "sha256": artifact_digest(destination),
            },
            "weights_sha256": file_digest(destination / "param.pt"),
            "origin": {
                "candidate_name": name,
                "parent_head": plan["parent"]["head"],
                "source_reference": {
                    k: v
                    for k, v in parent.items()
                    if k
                    in {
                        "generation",
                        "accepted_training",
                        "kind",
                        "model",
                        "weights_sha256",
                    }
                },
            },
        }
    evidence = stage / "genesis/fork/evidence"
    for name in plan["evidence_files"]:
        _copy(copied / name, evidence / name, budget)
    checkpoint_source = copied / state.latest_runtime_checkpoint_path
    incumbent_source = copied / plan["incumbent"]["bundle"]
    require(
        artifact_digest(incumbent_source) == plan["incumbent"]["identity"]["sha256"],
        "Fork incumbent closure differs.",
    )
    checkpoint, incumbent = stage / "input/checkpoint", stage / "input/incumbent"
    for source, destination in (
        (checkpoint_source, checkpoint),
        (incumbent_source, incumbent),
    ):
        for p in inventory(source):
            _copy(p, destination / p.relative_to(source), budget)
    return checkpoint, incumbent, seeds


@contextmanager
def _stage_owner(stage: Path) -> Iterator[Any]:
    """Share only the destination lease with its writer; close releases the last fd.

    Explicit LOCK_UN would release the inherited open-file-description lock while
    a surviving writer still held it. Source ownership descriptors are never passed.
    """
    with (stage / "resource_monitor/owner.lock").open("rb") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield handle


def _worker(
    stage: Path,
    checkpoint: Path,
    model: Path,
    name: str,
    budget: PreparationBudget,
    owner_fd: int,
) -> None:
    """Own one no-descendant genesis process; hard-stop/reap before releasing source."""
    import psutil

    identity = _local_identity()
    command = [
        sys.executable,
        "-I",
        "-B",
        "-m",
        "chipiron.environments.morpion.bootstrap.fork_worker",
        str(stage),
        str(checkpoint.relative_to(stage)),
        str(model.relative_to(stage)),
        name,
        str(identity["pid"]),
        identity["start_ticks"],
        str(owner_fd),
    ]
    with (stage / "fork-worker.log").open("xb") as log:
        environment = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("MORPION_RESOURCE_MONITOR")
        }
        environment["MORPION_WORK_DIR"] = str(stage)
        # Explicit finally performs group cleanup while the child PID is still
        # unreaped; Popen.__exit__ would wait before that group cleanup.
        process = subprocess.Popen(  # pylint: disable=consider-using-with
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            pass_fds=(owner_fd,),
            env=environment,
        )
        try:
            finished = None
            while finished is None:
                finished = os.waitid(
                    os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT
                )
                if finished is not None:
                    break
                try:
                    observed = psutil.Process(process.pid)
                    children = observed.children(recursive=True)
                    require(
                        not children, "Fork genesis unexpectedly spawned descendants."
                    )
                    budget.check(extra_rss=observed.memory_info().rss)
                except psutil.NoSuchProcess:
                    pass
                time.sleep(0.05)
            require(
                finished.si_code == os.CLD_EXITED and finished.si_status == 0,
                "Fork genesis worker failed; inspect owned fork-worker.log.",
            )
        finally:
            # WNOWAIT keeps our child's PID unreaped until group kill, preventing
            # PID/PGID reuse. The new session contains only our owned helper.
            previous = signal.pthread_sigmask(
                signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}
            )
            try:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
                deadline = time.monotonic() + 5
                while any(
                    _group(p.info["pid"]) == process.pid
                    for p in psutil.process_iter(["pid"])
                ):
                    require(
                        time.monotonic() < deadline, "Fork helper group not reaped."
                    )
                    time.sleep(0.05)
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def _group(pid: int) -> int | None:
    try:
        return os.getpgid(pid)
    except ProcessLookupError:
        return None


def _validate_stage(
    stage: Path, target: Path, plan: dict[str, Any], *, completed: bool = False
) -> Any:
    state = load_bootstrap_run_state(stage / "run_state.json")
    config = bootstrap_config_from_dict(
        Metadata().read(stage / "bootstrap_config.json")
    )
    origin = Metadata().read(stage / "genesis/fork/origin.json")
    enrollment: Any = config.metadata[MANAGED]
    origin_ref: Any = config.metadata[KEY]
    require(
        state.generation == 1
        and config.metadata[KEY] == state.metadata[KEY]
        and origin["plan_sha256"] == plan["plan_sha256"]
        and origin["work_dir"] == str(target)
        and enrollment["work_dir"] == str(target),
        "Fork staged genesis identity differs.",
    )
    require(
        state.metadata[MANAGED]["active"] is None
        and state.metadata[MANAGED]["sessions"] == []
        and state.metadata[MANAGED]["grants"] == []
        and state.metadata[MANAGED]["consumed_seconds"] == 0,
        "Fork staged native authority is not fresh.",
    )
    records = publication_records(state.metadata)
    require(
        len(records) == 1
        and "accepted_dataset" not in records[0]
        and "accepted_training" not in records[0],
        "Foreign acceptances cannot enter fork genesis.",
    )
    validate_capsule(stage, records[0])
    require(
        file_digest(stage / origin_ref["path"]) == origin_ref["sha256"],
        "Staged origin changed.",
    )
    require(
        set(origin["candidate_seeds"]) == set(config.evaluators.evaluators),
        "Staged seed catalogue differs.",
    )
    for seed in origin["candidate_seeds"].values():
        if seed is not None:
            bundle = inside(stage, seed["model"]["path"])
            require(
                artifact_digest(bundle) == seed["model"]["sha256"]
                and file_digest(bundle / "param.pt") == seed["weights_sha256"],
                "Staged imported candidate changed.",
            )
    if completed:
        marker = Metadata().read(stage / "fork-preparation.json")
        require(
            marker.get("status") == "validated",
            "Incomplete fork preparation is not publishable.",
        )
        actual = {
            str(p.relative_to(stage)): file_digest(p)
            for p in inventory(stage)
            if p.relative_to(stage).as_posix()
            not in {"fork-preparation.json", "fork-failure.json"}
        }
        require(
            actual == marker["inventory"], "Completed staged fork inventory changed."
        )
    return state


def _existing(
    parent: Path, target: Path, seconds: int, nodes: int, expected: str | None
) -> dict[str, Any]:
    require(
        expected is not None,
        "Existing fork target requires the exact expected plan for idempotent acknowledgement.",
    )
    paths = MorpionBootstrapPaths.from_work_dir(target)
    state = load_bootstrap_run_state(paths.run_state_path)
    from .config import load_bootstrap_config
    from .fork_origin import validate

    origin = validate(paths, load_bootstrap_config(paths.bootstrap_config_path), state)
    require(
        origin is not None
        and origin["plan_sha256"] == expected
        and origin["parent"]["root"] == str(parent.expanduser().resolve())
        and origin["budget"] == {"seconds": seconds, "nodes": nodes},
        "Conflicting existing fork target.",
    )
    marker = Metadata().read(target / "fork-preparation.json")
    require(
        marker.get("status") == "validated" and marker.get("plan_sha256") == expected,
        "Existing destination lacks completed fork preparation.",
    )
    from .generation_publication_io import sync_directory

    sync_directory(target)
    sync_directory(target.parent)
    return {
        "action": "already-published",
        "work_dir": str(target),
        "plan_sha256": expected,
    }


def _execute_fork(
    parent: Path, target: Path, seconds: int, nodes: int, *, expected: str | None = None
) -> dict[str, Any]:
    """Prepare one new independent normal run; never launch scientific workers."""
    target = target.expanduser().absolute()
    require(not target.is_symlink(), "Symlinked fork target.")
    if target.exists():
        return _existing(parent, target.resolve(), seconds, nodes, expected)
    import chipiron

    require(
        Path(chipiron.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()),
        "Fork execution requires the verified installed candidate.",
    )
    with source_cut(parent, target, seconds, nodes) as (plan, config, state):
        require(
            expected is None or expected == plan["plan_sha256"],
            "Fork plan changed; inspect the new cut.",
        )
        root, target = Path(plan["parent"]["root"]), Path(plan["target"])
        stage = target.with_name(f".{target.name}.fork-{plan['plan_sha256']}")
        prior_marker = None
        if stage.exists():
            marker = Metadata().read(stage / "fork-preparation.json")
            require(
                marker["plan_sha256"] == plan["plan_sha256"]
                and marker["target"] == str(target),
                "Unknown fork staging ownership.",
            )
            current_owner = _local_identity()
            require(
                isinstance(marker.get("owner"), dict)
                and all(
                    marker["owner"].get(k) == current_owner[k]
                    for k in ("host", "boot", "pid_namespace")
                ),
                "Foreign preparation scope cannot be recovered.",
            )
            prior_started = marker.get("started_monotonic_s")
            require(
                isinstance(prior_started, (int, float))
                and not isinstance(prior_started, bool)
                and math.isfinite(prior_started)
                and 0 <= prior_started <= time.monotonic(),
                "Invalid prior preparation clock.",
            )
            assert isinstance(prior_started, (int, float))
            require(
                time.monotonic() - prior_started
                < plan["preparation_limits"]["seconds"],
                "Original fork preparation deadline has expired.",
            )
            with _stage_owner(stage):
                if (stage / "run_state.json").exists() and marker.get(
                    "status"
                ) == "validated":
                    require(
                        marker.get("prior_read_reservation", 0) == 0,
                        "Fork recovery has exhausted its original-read retry reservation.",
                    )
                    marker = {
                        **marker,
                        "prior_read_reservation": plan["estimated_source_read_bytes"],
                        "owner": _local_identity(),
                    }
                    atomic_json(stage / "fork-preparation.json", marker)
                    recovery_budget = PreparationBudget(plan, stage, marker)
                    recovery_budget.check()
                    _validate_stage(stage, target, plan, completed=True)
                    recovery_budget.check()
                    require(
                        _inventory_identity(root)[0] == plan["source_inventory_sha256"],
                        "Source changed before staged recovery publication.",
                    )
                    validate_capsule(root, publication_records(state.metadata)[-1])
                    for path in inventory(stage):
                        recovery_budget.check()
                        sync_artifact(path, stage)
                    recovery_budget.check()
                    publish_directory_if_absent(stage, target, target.parent)
                    return {
                        "action": "published",
                        "work_dir": str(target),
                        "plan_sha256": plan["plan_sha256"],
                    }
                quarantine = stage.with_name(stage.name + ".incomplete")
                require(
                    not quarantine.exists(),
                    "Fork retained-attempt ceiling reached; preserve evidence and reclaim owned temp explicitly.",
                )
                require(
                    marker.get("prior_read_reservation", 0) == 0,
                    "Only one incomplete fork retry is admitted.",
                )
                require(
                    marker.get("owner") is not None,
                    "Incomplete stage lacks owner identity.",
                )
                from .persistent_worker_state import read_process_identity

                prior = read_process_identity(marker["owner"]["pid"])
                require(
                    prior is None or prior[2] != marker["owner"]["start_ticks"],
                    "Prior fork preparer remains alive; no restart.",
                )
                prior_marker = marker
                publish_directory_if_absent(stage, quarantine, target.parent)
        # A prior retained partial cannot be bypassed by removing its current stage.
        quarantine = stage.with_name(stage.name + ".incomplete")
        if prior_marker is None and quarantine.exists():
            prior_marker = Metadata().read(quarantine / "fork-preparation.json")
            require(
                prior_marker["plan_sha256"] == plan["plan_sha256"],
                "Foreign retained fork partial.",
            )
        stage.mkdir(mode=0o700)
        (stage / "resource_monitor").mkdir()
        (stage / "resource_monitor/owner.lock").touch(exist_ok=False)
        budget = PreparationBudget(plan, stage, prior_marker)
        marker = {
            "schema_version": 1,
            "plan_sha256": plan["plan_sha256"],
            "target": str(target),
            "owner": _local_identity(),
            "started_monotonic_s": budget.started,
            "prior_read_reservation": budget.prior_read_reservation,
        }
        atomic_json(stage / "fork-preparation.json", marker)
        with _stage_owner(stage) as owner:
            try:
                budget.check()
                checkpoint, model, seeds = _copy_closure(
                    root, stage, plan, state, budget
                )
                from .managed_sessions import capability

                capability_data = capability(target)
                origin = {
                    "schema_version": 1,
                    "work_dir": str(target),
                    "enrollment_id": capability_data["enrollment_id"],
                    "plan_sha256": plan["plan_sha256"],
                    "parent": plan["parent"],
                    "candidate_seeds": seeds,
                    "incumbent": plan["incumbent"]["identity"],
                    "recipe_sha256": object_digest(bootstrap_config_to_dict(config)),
                    "origin_evidence": {
                        str(
                            p.relative_to(stage / "genesis/fork/evidence")
                        ): file_digest(p)
                        for p in inventory(stage / "genesis/fork/evidence")
                    },
                    "dataset_policy": "regenerate_local",
                    "budget": {"seconds": seconds, "nodes": nodes},
                }
                require(
                    len(json.dumps(origin).encode()) <= 4 * 2**20,
                    "Fork origin metadata ceiling exceeded.",
                )
                origin_path = stage / "genesis/fork/origin.json"
                atomic_json(origin_path, origin)
                inherited_policy = policy_from_metadata(config.metadata)
                assert inherited_policy is not None
                new_policy = replace(
                    inherited_policy,
                    max_elapsed_seconds=seconds,
                    max_tree_nodes=nodes,
                )
                new_config = replace(
                    config,
                    metadata={
                        **config.metadata,
                        "continual_learning": new_policy.to_metadata(),
                        MANAGED: capability_data,
                        KEY: {
                            "path": "genesis/fork/origin.json",
                            "sha256": file_digest(origin_path),
                        },
                    },
                )
                atomic_json(
                    stage / "bootstrap_config.json",
                    bootstrap_config_to_dict(new_config),
                )
                _worker(
                    stage,
                    checkpoint,
                    model,
                    plan["incumbent"]["identity"]["name"],
                    budget,
                    owner.fileno(),
                )
                _validate_stage(stage, target, plan)
                source_sha, source_inodes = _inventory_identity(root)
                require(
                    source_sha == plan["source_inventory_sha256"],
                    "Parent inventory changed during fork.",
                )
                require(
                    not any(
                        (p.stat().st_dev, p.stat().st_ino) in source_inodes
                        for p in inventory(stage)
                    ),
                    "Cross-root scientific inode alias.",
                )
                require(
                    file_digest(root / "run_state.json")
                    == plan["parent"]["run_state_sha256"],
                    "Parent authority changed during fork.",
                )
                budget.check(read=plan["copy_bytes"])
                validate_capsule(root, publication_records(state.metadata)[-1])
                for parent_ref in plan["parent"]["parents"].values():
                    if parent_ref is not None:
                        require(
                            artifact_digest(inside(root, parent_ref["model"]["path"]))
                            == parent_ref["model"]["sha256"],
                            "Parent candidate changed during fork.",
                        )
                budget.check()
                # Input copies are not referenced by the new sealed local genesis.
                shutil.rmtree(stage / "input")
                final_inventory = {
                    str(p.relative_to(stage)): file_digest(p)
                    for p in inventory(stage)
                    if p.relative_to(stage).as_posix()
                    not in {"fork-preparation.json", "fork-failure.json"}
                }
                atomic_json(
                    stage / "fork-preparation.json",
                    {
                        **marker,
                        "status": "validated",
                        "read_bytes": budget.read_bytes,
                        "copy_bytes": budget.written_bytes,
                        "inventory": final_inventory,
                    },
                )
                for path in inventory(stage):
                    budget.check()
                    sync_artifact(path, stage)
                budget.check()
                publish_directory_if_absent(stage, target, target.parent)
                return {
                    "action": "published",
                    "work_dir": str(target),
                    "plan_sha256": plan["plan_sha256"],
                    "preparation_seconds": time.monotonic() - budget.started,
                }
            except BaseException:
                if stage.exists():
                    atomic_json(
                        stage / "fork-failure.json",
                        {
                            "plan_sha256": plan["plan_sha256"],
                            "status": "failed-unpublished",
                            "read_bytes": budget.read_bytes,
                            "copy_bytes": budget.written_bytes,
                        },
                    )
                raise


def execute_fork(
    parent: Path, target: Path, seconds: int, nodes: int, *, expected: str | None = None
) -> dict[str, Any]:
    """Run one foreground preparation with normal cancellation and owned cleanup."""
    require(
        threading.current_thread() is threading.main_thread(),
        "Fork preparation requires its foreground coordinator.",
    )
    require(
        isinstance(seconds, int)
        and not isinstance(seconds, bool)
        and 1 <= seconds <= 43200
        and isinstance(nodes, int)
        and not isinstance(nodes, bool)
        and nodes > 0,
        "Fork requires finite positive explicit time/node budgets.",
    )

    def stop(number: int, _frame: Any) -> None:
        raise SystemExit(128 + number)

    with ExitStack() as handlers:
        for number in (signal.SIGTERM, signal.SIGHUP):
            previous = signal.signal(number, stop)
            handlers.callback(signal.signal, number, previous)
        return _execute_fork(parent, target, seconds, nodes, expected=expected)

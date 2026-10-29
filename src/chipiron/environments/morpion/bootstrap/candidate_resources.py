"""Operational CPU execution budgets, deliberately outside scientific identities."""

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import psutil

if TYPE_CHECKING:
    from .learning_policy import MorpionContinualLearningPolicy


@dataclass(frozen=True)
class CandidateExecutionPlan:
    """Resolved bounds; memory includes the coordinator and its owned children."""

    workers: int
    threads: int
    cpu_capacity: float
    cpu_budget: int
    reserved_cpus: int
    memory_mb: int
    worker_memory_mb: int
    min_available_mb: int
    timeout_s: int


def _integer(name: str, default: int) -> int:
    value = int(os.environ.get(f"CHIPIRON_CANDIDATE_{name}", default))
    if value <= 0:
        message = f"CHIPIRON_CANDIDATE_{name} must be positive"
        raise ValueError(message)
    return value


def cpu_capacity() -> float:
    """Honor affinity and all visible ancestor cgroup CPU quotas (v1 and v2)."""
    capacity = float(
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1)
    )
    groups = Path("/proc/self/cgroup")
    mounts = Path("/proc/self/mountinfo")
    if not groups.exists() or not mounts.exists():
        return capacity
    for group in groups.read_text(encoding="utf-8").splitlines():
        _, controllers, relative = group.split(":", 2)
        for mount in mounts.read_text(encoding="utf-8").splitlines():
            before, after = mount.split(" - ", 1)
            fields, fs = before.split(), after.split()
            if fs[0] not in {"cgroup", "cgroup2"}:
                continue
            if fs[0] == "cgroup" and (
                "cpu" not in controllers.split(",") or "cpu" not in fs[2].split(",")
            ):
                continue
            if fs[0] == "cgroup2" and controllers:
                continue
            root, location = Path(fields[3]), Path(fields[4])
            try:
                suffix = Path(relative).relative_to(root)
            except ValueError:
                continue
            directory = location / suffix
            while directory.is_relative_to(location):
                quota = directory / "cpu.max"
                if quota.exists():
                    amount, period = quota.read_text().split()
                    if amount != "max":
                        capacity = min(capacity, int(amount) / int(period))
                quota = directory / "cpu.cfs_quota_us"
                if quota.exists() and int(quota.read_text()) > 0:
                    capacity = min(
                        capacity,
                        int(quota.read_text())
                        / int((directory / "cpu.cfs_period_us").read_text()),
                    )
                if directory == location:
                    break
                directory = directory.parent
    return capacity


def resolve_candidate_plan(
    count: int, policy: MorpionContinualLearningPolicy | None, device: str
) -> CandidateExecutionPlan | None:
    """Keep legacy inline execution unless concurrency is explicitly requested."""
    requested = _integer("WORKERS", 1)
    if requested == 1:
        return None
    if policy is None or device.strip().lower() != "cpu":
        message = (
            "Parallel candidates require continual learning and explicit CPU training."
        )
        raise ValueError(message)
    capacity = cpu_capacity()
    reserved = _integer("RESERVED_CPUS", 2)
    budget = min(_integer("CPU_BUDGET", 4), math.floor(capacity) - reserved)
    threads = _integer("THREADS", 1)
    memory = min(
        _integer("MEMORY_MB", policy.max_group_rss_mb), policy.max_group_rss_mb
    )
    per_worker = _integer("WORKER_MEMORY_MB", 1024)
    available = psutil.virtual_memory().available // 2**20 - policy.min_available_ram_mb
    headroom = min(memory, available) - math.ceil(
        psutil.Process().memory_info().rss / 2**20
    )
    workers = min(requested, count, budget // threads, headroom // per_worker)
    if workers < 1:
        message = (
            "Candidate resource budget leaves no capacity after pipeline reserves."
        )
        raise ValueError(message)
    return CandidateExecutionPlan(
        workers,
        threads,
        capacity,
        budget,
        reserved,
        memory,
        per_worker,
        policy.min_available_ram_mb,
        policy.worker_max_seconds,
    )

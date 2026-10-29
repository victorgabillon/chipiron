"""Exact CPU candidate equivalence and real spawned-process failure boundaries."""

import json
import multiprocessing
from dataclasses import replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import pytest
import torch

from chipiron.environments.morpion.bootstrap.candidate_executor import (
    CandidateExecutionError,
    execute_candidates,
)
from chipiron.environments.morpion.bootstrap.candidate_job import (
    CandidateJob,
    run_candidate,
)
from chipiron.environments.morpion.bootstrap.candidate_resources import (
    CandidateExecutionPlan,
    resolve_candidate_plan,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    artifact_digest,
    file_digest,
)
from chipiron.environments.morpion.bootstrap.learning_policy import (
    MorpionContinualLearningPolicy,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.bundle import (
    load_morpion_model_bundle,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training import (
    MorpionStreamingTrainingArgs,
    MorpionTrainingArgs,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training.continuation import (
    MorpionTrainingContinuation,
    candidate_training_seed,
)
from tests.environments.test_morpion_continual_training import rows_file


def plan(workers: int) -> CandidateExecutionPlan:
    """Keep one CPU thread and explicit bounded memory for each test worker."""
    return CandidateExecutionPlan(workers, 1, 8, 4, 2, 4096, 1024, 1, 120)


def jobs_for(
    rows: Path, root: Path, generation: int, parents: Path | None = None
) -> list[CandidateJob]:
    """Exercise distinct seeds and every accepted lineage, including losing models."""
    jobs = []
    for index, kind in enumerate(("linear", "mlp", "mlp", "linear")):
        name = f"{kind}_{index}"
        parent = None if parents is None else parents / name
        jobs.append(
            CandidateJob(
                name,
                MorpionStreamingTrainingArgs(
                    MorpionTrainingArgs(
                        rows,
                        root / name,
                        model_kind=kind,
                        hidden_sizes=(4,) if kind == "mlp" else None,
                        batch_size=4,
                        num_epochs=2,
                        device="cpu",
                        stable_validation_split=True,
                        continuation=MorpionTrainingContinuation(
                            name,
                            candidate_training_seed(17, name, generation),
                            parent_bundle=None if parent is None else str(parent),
                            parent_bundle_sha256=None
                            if parent is None
                            else artifact_digest(parent),
                            parent_weights_sha256=None
                            if parent is None
                            else file_digest(parent / "param.pt"),
                        ),
                    ),
                    row_chunk_size=5,
                ),
            )
        )
    return jobs


def assert_state_equal(a: Any, b: Any) -> None:
    """Compare complete tensor/Adam structures with exact values, not archive bytes."""
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key, value in a.items():
            assert_state_equal(value, b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            assert_state_equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("workers", [2, 4])
def test_three_rounds_exact_serial_parallel(tmp_path: Path, workers: int) -> None:
    """Compare every tensor, Adam field, RNG observation, split, metric and selection."""
    torch.set_num_threads(1)
    rows = rows_file(tmp_path)
    previous_serial = previous_parallel = None
    for generation in range(1, 4):
        serial = tmp_path / f"serial-{generation}"
        parallel = tmp_path / f"parallel-{generation}"
        serial_jobs = jobs_for(rows, serial, generation, previous_serial)
        parallel_jobs = jobs_for(rows, parallel, generation, previous_parallel)
        serial_metrics = {job.name: run_candidate(job)[1] for job in serial_jobs}
        receipts = execute_candidates(
            parallel_jobs, plan(workers), tmp_path / f"receipts-{generation}"
        )
        assert list(receipts) == list(serial_metrics)
        for name, metrics in serial_metrics.items():
            for filename in ("param.pt", "optimizer.pt"):
                assert_state_equal(
                    torch.load(serial / name / filename, weights_only=True),
                    torch.load(parallel / name / filename, weights_only=True),
                )
            _, _, a = load_morpion_model_bundle(serial / name)
            _, _, b = load_morpion_model_bundle(parallel / name)
            assert a.metadata["state_split"] == b.metadata["state_split"]
            for key, value in a.metadata["continuation"].items():
                if key not in {
                    "parent_bundle_sha256",
                    "parent_weights_sha256",
                    "initial_bundle_sha256",
                    "optimizer_sha256",
                }:
                    assert value == b.metadata["continuation"][key], key
            for key, value in metrics.items():
                if (
                    not key.startswith("timing_")
                    and "per_second" not in key
                    and not key.endswith("_cache_rebuilt")
                ):
                    assert value == receipts[name]["metrics"][key], key
        assert min(
            serial_metrics, key=lambda n: (serial_metrics[n]["validation_loss"], n)
        ) == min(receipts, key=lambda n: (receipts[n]["metrics"]["validation_loss"], n))
        previous_serial, previous_parallel = serial, parallel
    assert not multiprocessing.active_children()


def test_failed_child_reaps_siblings(tmp_path: Path) -> None:
    """An invalid accepted parent fails inside the child; no partial success escapes."""
    rows = rows_file(tmp_path)
    jobs = jobs_for(rows, tmp_path / "outputs", 1)
    first = jobs[0]
    continuation = replace(
        first.training.training_args.continuation,
        parent_bundle=str(tmp_path / "missing"),
        parent_bundle_sha256="bad",
        parent_weights_sha256="bad",
    )
    jobs[0] = replace(
        first,
        training=replace(
            first.training,
            training_args=replace(
                first.training.training_args, continuation=continuation
            ),
        ),
    )
    with pytest.raises(CandidateExecutionError, match="exited"):
        execute_candidates(jobs, plan(2), tmp_path / "receipts")
    assert not multiprocessing.active_children()
    assert json.loads((tmp_path / "receipts/execution.json").read_text())[
        "children_reaped"
    ]


def test_resource_stop_reaps_children(tmp_path: Path) -> None:
    """Enforce a real child RSS budget and leave no owned writer alive."""
    rows = rows_file(tmp_path)
    with pytest.raises(CandidateExecutionError, match="memory"):
        execute_candidates(
            jobs_for(rows, tmp_path / "outputs", 1),
            replace(plan(2), worker_memory_mb=1),
            tmp_path / "receipts",
        )
    assert not multiprocessing.active_children()


def test_opt_in_and_cpu_reserve(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defaults preserve serial; quota/affinity reserve limits the requested count."""
    from chipiron.environments.morpion.bootstrap import candidate_resources

    monkeypatch.delenv("CHIPIRON_CANDIDATE_WORKERS", raising=False)
    assert resolve_candidate_plan(9, None, "auto") is None
    monkeypatch.setenv("CHIPIRON_CANDIDATE_WORKERS", "4")
    with pytest.raises(ValueError, match="explicit CPU"):
        resolve_candidate_plan(9, MorpionContinualLearningPolicy(), "auto")
    monkeypatch.setattr(candidate_resources, "cpu_capacity", lambda: 3.5)
    resolved = resolve_candidate_plan(
        9, MorpionContinualLearningPolicy(min_available_ram_mb=1), "cpu"
    )
    assert resolved is not None and resolved.workers == 1


@pytest.mark.parametrize("death_signal", ["SIGTERM", "SIGKILL"])
def test_coordinator_death_leaves_no_writers(tmp_path: Path, death_signal: str) -> None:
    """Kill a real coordinator during training and observe all writer PIDs stop."""
    import os
    import signal
    import subprocess
    import sys
    import time

    import psutil

    rows_file(tmp_path)
    script = tmp_path / "coordinator.py"
    script.write_text("""from pathlib import Path
from dataclasses import replace
from tests.environments.test_morpion_candidate_execution import jobs_for, plan
from chipiron.environments.morpion.bootstrap.candidate_executor import execute_candidates
if __name__ == '__main__':
    root = Path(__file__).parent
    jobs = jobs_for(root / 'rows.jsonl', root / 'outputs', 1)
    jobs = [replace(job, training=replace(job.training, training_args=replace(job.training.training_args, num_epochs=100000))) for job in jobs]
    execute_candidates(jobs, plan(2), root / 'receipts')
""")
    with (tmp_path / "coordinator.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, str(script)],
            stdout=log,
            stderr=log,
            env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        )
        children = []
        try:
            deadline = time.monotonic() + 30
            marker = tmp_path / "receipts/children.json"
            while time.monotonic() < deadline:
                if marker.exists():
                    children = json.loads(marker.read_text())
                    if len(children) == 2 and list(
                        (tmp_path / "outputs").glob("*/initial/param.pt")
                    ):
                        break
                assert process.poll() is None, (
                    tmp_path / "coordinator.log"
                ).read_text()
                time.sleep(0.1)
            assert len(children) == 2
            process.send_signal(getattr(signal, death_signal))
            process.wait(timeout=10)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                living = [
                    pid
                    for pid in children
                    if psutil.pid_exists(pid)
                    and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
                ]
                if not living:
                    break
                time.sleep(0.1)
            assert not living
            if death_signal == "SIGTERM":
                assert json.loads((tmp_path / "receipts/execution.json").read_text())[
                    "children_reaped"
                ]
            snapshots = {
                p: (p.stat().st_size, p.stat().st_mtime_ns)
                for p in (tmp_path / "outputs").rglob("*")
                if p.is_file()
            }
            time.sleep(0.2)
            assert snapshots == {
                p: (p.stat().st_size, p.stat().st_mtime_ns)
                for p in (tmp_path / "outputs").rglob("*")
                if p.is_file()
            }
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            for pid in children:
                if (
                    psutil.pid_exists(pid)
                    and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
                ):
                    psutil.Process(pid).kill()


def test_cpu_rng_evidence_does_not_read_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CUDA-capable installation must not initialize devices for CPU evidence."""
    rows = rows_file(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_rng_state_all",
        lambda: pytest.fail("CPU RNG evidence initialized CUDA"),
    )
    run_candidate(jobs_for(rows, tmp_path / "outputs", 1)[0])


@pytest.mark.parametrize(
    "kind", ["flat_tensor", "entity_token", "relational_entity_token"]
)
def test_read_only_cache_never_repairs_missing_input(tmp_path: Path, kind: str) -> None:
    """All cache families fail closed instead of racing a rebuild in a child."""
    from importlib import import_module

    module = import_module(
        f"chipiron.environments.morpion.players.evaluators.neural_networks.training.{kind}_cache"
    )
    rows = rows_file(tmp_path)
    arguments = {
        "rows_path": rows,
        "max_rows": None,
        "row_chunk_size": 5,
        "read_only": True,
    }
    if kind != "flat_tensor":
        arguments["entity_max_tokens"] = 1536
    with pytest.raises(ValueError, match="missing or stale"):
        getattr(module, f"load_or_materialize_{kind}_cache")(**arguments)
    assert not (tmp_path / "tensor_cache").exists()


@pytest.mark.parametrize("version", [1, 2])
def test_cgroup_cpu_and_memory_headroom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: int
) -> None:
    """Container quotas and usage cap host-wide affinity and available-memory reports."""
    from chipiron.environments.morpion.bootstrap import candidate_resources

    monkeypatch.setattr(
        candidate_resources, "_cgroup_directories", lambda controller: [tmp_path]
    )
    monkeypatch.setattr(
        candidate_resources.os, "sched_getaffinity", lambda pid: set(range(16))
    )
    if version == 2:
        (tmp_path / "cpu.max").write_text("250000 100000")
        (tmp_path / "memory.max").write_text(str(128 * 2**20))
        (tmp_path / "memory.current").write_text(str(112 * 2**20))
    else:
        (tmp_path / "cpu.cfs_quota_us").write_text("250000")
        (tmp_path / "cpu.cfs_period_us").write_text("100000")
        (tmp_path / "memory.limit_in_bytes").write_text(str(128 * 2**20))
        (tmp_path / "memory.usage_in_bytes").write_text(str(112 * 2**20))
    assert candidate_resources.cpu_capacity() == 2.5
    assert candidate_resources.available_memory_mb() <= 16

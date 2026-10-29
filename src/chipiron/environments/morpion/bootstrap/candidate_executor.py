"""Bounded spawned CPU jobs with fail-closed results and owned-child cleanup."""

import json
import multiprocessing
import os
import signal
import tempfile
import threading
import time
from contextlib import contextmanager, suppress
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Never

import psutil

from .candidate_job import CandidateJob, prepare_candidate_cache, run_candidate
from .candidate_resources import available_memory_mb

if TYPE_CHECKING:
    from collections.abc import Iterator
    from multiprocessing.connection import Connection

    from .candidate_resources import CandidateExecutionPlan


class CandidateExecutionError(RuntimeError):
    """No results may be accepted after a worker or resource failure."""


def _fail(reason: str) -> Never:
    """Reject an incomplete round."""
    raise CandidateExecutionError(reason)


def _parent_watch(lifeline: Connection) -> None:
    """Exit when the coordinator closes the sole writer, including after SIGKILL."""
    try:
        lifeline.recv_bytes()
    except EOFError, OSError:
        os._exit(125)


def _worker(
    job: CandidateJob, result: Path, lifeline: Connection, threads: int
) -> None:
    """Only private bundle/result/temp writes; no coordinator authority."""
    threading.Thread(target=_parent_watch, args=(lifeline,), daemon=True).start()
    import torch

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    from .performance import StageMeasurement

    with tempfile.TemporaryDirectory(
        prefix="candidate-", dir=result.parent
    ) as temporary:
        tempfile.tempdir = temporary
        os.environ["TMPDIR"] = temporary
        os.environ["SQLITE_TMPDIR"] = temporary
        measurement = StageMeasurement(cuda_device="cpu")
        model, metrics = run_candidate(job)
        del model
        if torch.cuda.is_initialized():  # type: ignore[no-untyped-call]
            _fail("CPU candidate initialized CUDA.")
        result.write_text(
            json.dumps(
                {"metrics": metrics, "performance": measurement.finish()},
                allow_nan=False,
            )
        )


@contextmanager
def _spawn_environment(threads: int) -> Iterator[None]:
    """Set inherited native thread/GPU limits before the spawned interpreter imports Torch."""
    values = {
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "NUMEXPR_NUM_THREADS": str(threads),
        "CUDA_VISIBLE_DEVICES": "",
        "HIP_VISIBLE_DEVICES": "",
        "ROCR_VISIBLE_DEVICES": "",
    }
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _interrupt(signum: int, frame: object) -> None:
    del frame
    message = f"Candidate coordinator received signal {signum}"
    raise KeyboardInterrupt(message)


def execute_candidates(
    jobs: list[CandidateJob], plan: CandidateExecutionPlan, directory: Path
) -> dict[str, dict[str, Any]]:
    """Return ordered receipts only after every writer exits; otherwise raise and reap."""
    if threading.current_thread() is not threading.main_thread():
        _fail("Candidate coordination requires the main thread.")
    if len({
        str(Path(job.training.training_args.output_dir).resolve()) for job in jobs
    }) != len(jobs):
        _fail("Candidate output directories must be distinct.")
    if len({job.name for job in jobs}) != len(jobs):
        _fail("Candidate names must be distinct.")
    if any(
        job.training.training_args.device != "cpu"
        or job.training.training_args.continuation is None
        for job in jobs
    ):
        _fail("Spawned jobs require explicit CPU and captured continuation identities.")
    directory.mkdir(parents=True, exist_ok=False)
    context = multiprocessing.get_context("spawn")
    owned: list[Any] = []
    pipes: list[Connection] = []
    active: dict[int, tuple[Any, Path]] = {}
    results: dict[str, dict[str, Any]] = {}
    started = time.monotonic()
    peak_mb = 0.0
    cache_s: float | None = None
    stop_monitor = threading.Event()
    resource_failure: list[str] = []

    def monitor_resources() -> None:
        nonlocal peak_mb
        while not stop_monitor.wait(0.05):
            rss = psutil.Process().memory_info().rss
            reason = None
            for process in owned:
                if process.pid is None or process.exitcode is not None:
                    continue
                with suppress(psutil.NoSuchProcess):
                    child_rss = psutil.Process(process.pid).memory_info().rss
                    rss += child_rss
                    if child_rss / 2**20 > plan.worker_memory_mb:
                        reason = "Candidate worker memory budget exceeded."
            peak_mb = max(peak_mb, rss / 2**20)
            if (
                peak_mb > plan.memory_mb
                or available_memory_mb() < plan.min_available_mb
            ):
                reason = "Candidate aggregate memory budget exceeded."
            if time.monotonic() - started > plan.timeout_s:
                reason = "Candidate round time budget exceeded."
            if reason is not None:
                resource_failure.append(reason)
                os.kill(os.getpid(), signal.SIGTERM)
                return

    def stop_execution(signum: int, frame: object) -> None:
        if resource_failure:
            _fail(resource_failure[0])
        _interrupt(signum, frame)

    previous_signal = signal.signal(signal.SIGTERM, stop_execution)
    monitor = threading.Thread(target=monitor_resources, daemon=True)
    monitor.start()
    try:
        cache_started = time.monotonic()
        for job in jobs:
            prepare_candidate_cache(job)
        cache_s = time.monotonic() - cache_started
        next_job = 0
        with _spawn_environment(plan.threads):
            while active or next_job < len(jobs):
                while next_job < len(jobs) and len(active) < plan.workers:
                    index = next_job
                    job = jobs[index]
                    job = replace(
                        job,
                        training=replace(
                            job.training, cache_read_only=True, progress_callback=None
                        ),
                    )
                    reader, writer = context.Pipe(duplex=False)
                    pipes.append(writer)
                    receipt = directory / f"{index}.json"
                    process = context.Process(
                        target=_worker,
                        args=(job, receipt, reader, plan.threads),
                        name=f"candidate-{index}",
                    )
                    owned.append(process)
                    try:
                        process.start()
                    finally:
                        reader.close()
                    active[index] = (process, receipt)
                    child_record = directory / "children.tmp"
                    child_record.write_text(json.dumps([child.pid for child in owned]))
                    child_record.replace(directory / "children.json")
                    next_job += 1
                for index, (process, receipt) in list(active.items()):
                    if process.exitcode is None:
                        continue
                    process.join()
                    if process.exitcode != 0:
                        _fail(
                            f"Candidate {jobs[index].name} exited with {process.exitcode}."
                        )
                    results[jobs[index].name] = json.loads(receipt.read_text())
                    del active[index]
                time.sleep(0.05)
        return {job.name: results[job.name] for job in jobs}
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        previous_int = signal.signal(signal.SIGINT, signal.SIG_IGN)
        stop_monitor.set()
        monitor.join()
        # No deletion or acceptance can happen until this barrier completes.
        for process in owned:
            if process.pid is not None and process.is_alive():
                process.terminate()
        for process in owned:
            if process.pid is not None:
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()
                process.join()
                process.close()
        for pipe in pipes:
            pipe.close()
        signal.signal(signal.SIGTERM, previous_signal)
        signal.signal(signal.SIGINT, previous_int)
        (directory / "execution.json").write_text(
            json.dumps({
                "plan": asdict(plan),
                "cache_preparation_s": cache_s,
                "wall_s": time.monotonic() - started,
                "aggregate_peak_rss_mb": peak_mb,
                "completed": [job.name for job in jobs if job.name in results],
                "children_reaped": True,
            })
        )

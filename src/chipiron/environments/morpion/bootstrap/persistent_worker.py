"""Reuse Python only; dispatch existing bounded scientific jobs with cold runtimes."""

import gc
import logging
import os
import shutil
import signal
import time
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from tempfile import mkdtemp
from typing import TYPE_CHECKING

from .bootstrap_paths import MorpionBootstrapPaths
from .execution_profile import launcher_execution
from .generation_publication_io import file_digest, require
from .learning_worker import monitored_identity
from .performance import StageMeasurement
from .persistent_worker_state import STAGES, WorkerJobState
from .process_control import (
    load_morpion_bootstrap_process_state,
    mark_current_launcher_process_stopped,
    register_current_launcher_process,
)
from .run_state import load_bootstrap_run_state
from .runtime_performance import WorkerObservation, observe_worker_boundary

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import FrameType

    from .launcher import MorpionBootstrapLauncherArgs

LOGGER = logging.getLogger(__name__)


def _validate(launcher: MorpionBootstrapLauncherArgs) -> None:
    """Keep initial qualification narrow and require the actual monitor ancestry."""
    args = launcher.bootstrap_args
    require(
        launcher.scientific
        and launcher.scientific_operation is None
        and launcher.pipeline_stage in STAGES
        and launcher.pipeline_generation is None,
        "Persistent jobs require a monitored scientific worker stage",
    )
    require(
        launcher.pipeline_stage != "growth" or launcher.max_cycles in {None, 1},
        "Persistent Growth jobs must remain one cycle",
    )
    require(
        args.execution_profile == "strict_math"
        and args.torch_intra_op_threads == args.torch_inter_op_threads == 1
        and args.training_device in {"cpu", "cuda:0"}
        and args.continual_learning is not None
        and os.environ.get("CHIPIRON_CANDIDATE_WORKERS", "1") == "1",
        "Persistent jobs require strict_math 1/1, explicit CPU/cuda:0 and serial continual candidates",
    )
    require(launcher.worker_poll_seconds > 0, "Invalid persistent poll interval")
    require(
        launcher.worker_max_jobs is None or launcher.worker_max_jobs > 0,
        "Invalid persistent job bound",
    )
    identity = monitored_identity()
    require(
        identity["session"] == identity["pid"],
        "Persistent worker must lead its supervised session",
    )


def _cleanup() -> None:
    """Discard per-invocation split memory and allocator slack, never disk artifacts."""
    import psutil
    import torch

    from chipiron.environments.morpion.players.evaluators.neural_networks.training.state_split import (
        _split,
    )
    from chipiron.environments.morpion.players.evaluators.neural_networks.training.state_split_cache import (
        _cached,
    )

    _split.cache_clear()
    _cached.cache_clear()
    gc.collect()
    is_initialized: Callable[[], bool] = torch.cuda.is_initialized
    if is_initialized():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    survivors = []
    for process in psutil.process_iter():
        try:
            if process.pid != os.getpid() and os.getsid(process.pid) == os.getsid(0):
                survivors.append(process.pid)
        except ProcessLookupError:
            continue
    require(
        not survivors and not psutil.Process().children(recursive=True),
        "Persistent job left child processes; session must stop before retry",
    )


def _interrupt(number: int, _frame: FrameType | None) -> None:
    """Unwind normal ownership; the supervisor retains bounded group kill fallback."""
    raise SystemExit(128 + number)


class _ReaderScope:
    """Keep only the supervisor's existing same-source on-disk reader scope."""

    def __init__(self, launcher: MorpionBootstrapLauncherArgs) -> None:
        self.launcher = launcher
        self.scope: Path | None = None
        self.source: str | None = None

    def next(self) -> Path | None:
        """Dataset rotates every job; Reevaluation rotates when its source changes."""
        if self.launcher.pipeline_stage not in {"dataset_worker", "reevaluation"}:
            return None
        paths = MorpionBootstrapPaths.from_work_dir(self.launcher.work_dir)
        state = load_bootstrap_run_state(paths.run_state_path)
        source = str(state.latest_tree_snapshot_path)
        if self.launcher.pipeline_stage == "dataset_worker" or source != self.source:
            self.close()
        if self.scope is None:
            parent = self.launcher.reader_cache_dir
            if parent is not None:
                parent.mkdir(parents=True, exist_ok=True)
            self.scope = Path(mkdtemp(prefix="persistent-reader-", dir=parent))
            self.source = source
        return self.scope

    def close(self) -> None:
        """Remove only this process's private scope after synchronous consumers return."""
        if self.scope is not None:
            shutil.rmtree(self.scope)
            self.scope = None


def run_persistent_worker(launcher: MorpionBootstrapLauncherArgs) -> int:
    """Run bounded existing jobs; any failure exits, preserving conservative claims."""
    from .launcher import _run_launcher_cli

    _validate(launcher)
    paths = MorpionBootstrapPaths.from_work_dir(launcher.work_dir)
    paths.ensure_directories()
    process_device = launcher.bootstrap_args.training_device
    state = WorkerJobState(paths.work_dir, launcher.pipeline_stage)
    register_current_launcher_process(paths)
    observation = WorkerObservation(paths.work_dir, launcher.pipeline_stage)
    reader = _ReaderScope(launcher)
    exit_code = 1
    with ExitStack() as resources:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous = signal.signal(sig, _interrupt)
            resources.callback(signal.signal, sig, previous)
        try:
            job_id = 0
            while launcher.worker_max_jobs is None or job_id < launcher.worker_max_jobs:
                job_id += 1
                state.update(
                    "running", job_id=job_id, job_started_monotonic_s=time.monotonic()
                )
                # Re-read configuration and state through the ordinary launcher;
                # neither admission nor permitted-source selection is duplicated.
                with launcher_execution(launcher) as configured:
                    _validate(configured)
                    require(
                        configured.bootstrap_args.training_device == process_device,
                        "Persistent process device changed; restart required",
                    )
                    state.update(
                        "running",
                        dispatch_config_sha256=file_digest(paths.bootstrap_config_path),
                        dispatch_state_sha256=file_digest(paths.run_state_path),
                        dispatch_identity_scope="observed at dispatch; existing admission records select permitted source/model",
                    )
                    configured = replace(configured, reader_cache_dir=reader.next())
                    _run_launcher_cli(configured, _job_id=job_id)
                state.update("cleanup")
                cleanup = StageMeasurement()
                _cleanup()
                if launcher.pipeline_stage == "dataset_worker":
                    reader.close()
                cleanup_record = cleanup.finish(evidence_status="completed")
                observe_worker_boundary(
                    paths.work_dir,
                    launcher.pipeline_stage,
                    job_id,
                    "worker_cleanup",
                    cleanup_record,
                )
                state.update("idle", cleanup=cleanup_record, completed_jobs=job_id)
                LOGGER.info(
                    "[persistent-worker] job=%s stage=%s idle; all job ownership released",
                    job_id,
                    launcher.pipeline_stage,
                )
                if (
                    launcher.worker_max_jobs is not None
                    and job_id >= launcher.worker_max_jobs
                ):
                    break
                waiting = StageMeasurement()
                completed_wait = False
                try:
                    deadline = time.monotonic() + launcher.worker_poll_seconds
                    while time.monotonic() < deadline:
                        state.update("idle")
                        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
                    completed_wait = True
                finally:
                    observe_worker_boundary(
                        paths.work_dir,
                        launcher.pipeline_stage,
                        job_id,
                        "worker_poll",
                        waiting.finish(
                            evidence_status="completed"
                            if completed_wait
                            else "incomplete"
                        ),
                    )
            exit_code = 0
            return 0
        finally:
            # On exceptions, never publish idle/reclaimability. The supervisor
            # reaps the complete session before deleting its outer cache root.
            try:
                if exit_code == 0:
                    reader.close()
            finally:
                observation.finish(exit_code)
                if load_morpion_bootstrap_process_state(paths).pid == os.getpid():
                    mark_current_launcher_process_stopped(paths, exit_code=exit_code)
                try:
                    state.update("stopped", exit_code=exit_code)
                finally:
                    state.close()

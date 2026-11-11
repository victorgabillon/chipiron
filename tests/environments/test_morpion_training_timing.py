"""Timing policy reaches real Morpion jobs without changing accepted science."""

import json
import multiprocessing
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest
import torch

from chipiron.environments.morpion.bootstrap.candidate_executor import (
    execute_candidates,
)
from chipiron.environments.morpion.bootstrap.candidate_job import run_candidate
from chipiron.environments.morpion.bootstrap.pipeline.stages import (
    run_pipeline_training_stage,
)
from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
    load_pipeline_training_status_file,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training import (
    MorpionStreamingTrainingArgs,
    MorpionTrainingArgs,
    service,
    train_morpion_regressor,
    train_morpion_regressor_streaming,
)
from tests.environments.morpion_durable_training_helpers import setup_training
from tests.environments.morpion_fenced_dataset_helpers import local_leaderboard
from tests.environments.test_morpion_candidate_execution import (
    assert_state_equal,
    jobs_for,
    plan,
)
from tests.environments.test_morpion_continual_training import rows_file

if TYPE_CHECKING:
    from pathlib import Path

    from chipiron.learning.training_timing import TrainingTimingMode


@pytest.mark.fast
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "kind",
    [
        "linear",
        "entity_token_transformer_value_net",
        "relation_biased_entity_token_transformer_value_net",
    ],
)
def test_complete_services_preserve_science_and_namespace(
    tmp_path: Path, streaming: bool, kind: str
) -> None:
    """Cover in-memory, flat, packed entity and relational train/evaluation paths."""
    rows = rows_file(tmp_path)
    results = []
    for mode in ("synchronized", "host_enqueue"):
        output = tmp_path / mode
        args = MorpionTrainingArgs(
            rows,
            output,
            model_kind=kind,
            device="cpu",
            batch_size=8,
            num_epochs=1,
            entity_d_model=8,
            entity_n_head=2,
            entity_n_layer=1,
            entity_dim_feedforward=16,
            entity_max_tokens=256,
            entity_relation_schema="morpion_entity_relations_v1"
            if "relation_biased" in kind
            else None,
            entity_relation_type_count=16 if "relation_biased" in kind else None,
        )
        typed_mode: TrainingTimingMode = (
            "synchronized" if mode == "synchronized" else "host_enqueue"
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(519)
            if streaming:
                model, metrics = train_morpion_regressor_streaming(
                    MorpionStreamingTrainingArgs(
                        args, row_chunk_size=8, timing_mode=typed_mode
                    )
                )
            else:
                model, metrics = train_morpion_regressor(args, timing_mode=typed_mode)
            rng = torch.get_rng_state()
        manifest = json.loads((output / "morpion_manifest.json").read_text())[
            "metadata"
        ]
        timing = manifest["timing"]
        assert metrics["timing_mode"] == timing["mode"] == mode
        assert timing["coarse_scope"] == "device_complete_host_elapsed"
        for scope in ("train", "train_metrics", "validation_metrics"):
            expected = "forward" if mode == "synchronized" else "host_enqueue_forward"
            assert expected in timing[scope]
            assert ("forward" in timing[scope]) == (mode == "synchronized")
        results.append((
            model.state_dict(),
            rng,
            {
                k: v
                for k, v in metrics.items()
                if not k.startswith("timing_") and not k.endswith("_cache_rebuilt")
            },
        ))
    assert_state_equal(results[0], results[1])


@pytest.mark.fast
def test_persistent_jobs_and_exception_do_not_leak_mode(tmp_path: Path) -> None:
    """Success/failure/success in one interpreter carries only each job's policy."""
    rows = rows_file(tmp_path)
    jobs = jobs_for(rows, tmp_path / "jobs", 1)
    first = replace(
        jobs[0], training=replace(jobs[0].training, timing_mode="host_enqueue")
    )
    assert run_candidate(first)[1]["timing_mode"] == "host_enqueue"
    broken = replace(
        first,
        training=replace(
            first.training,
            training_args=replace(
                first.training.training_args,
                dataset_file=tmp_path / "missing",
                output_dir=tmp_path / "bad",
            ),
        ),
    )
    with pytest.raises(FileNotFoundError):
        run_candidate(broken)
    assert run_candidate(jobs[1])[1]["timing_mode"] == "synchronized"


def test_spawn_captures_modes_and_exact_adam(tmp_path: Path) -> None:
    """Spawned children receive independent typed recipes, without GPU initialization."""
    rows = rows_file(tmp_path)
    serial = jobs_for(rows, tmp_path / "serial", 1)[:2]
    serial[0] = replace(
        serial[0], training=replace(serial[0].training, timing_mode="host_enqueue")
    )
    spawned = [
        replace(
            j,
            training=replace(
                j.training,
                training_args=replace(
                    j.training.training_args, output_dir=tmp_path / "spawn" / j.name
                ),
            ),
        )
        for j in serial
    ]
    expected = {j.name: run_candidate(j)[1] for j in serial}
    actual = execute_candidates(spawned, plan(2), tmp_path / "receipts")
    for job in spawned:
        assert (
            actual[job.name]["metrics"]["timing_mode"]
            == expected[job.name]["timing_mode"]
        )
        for name in ("param.pt", "optimizer.pt"):
            assert_state_equal(
                torch.load(tmp_path / "serial" / job.name / name, weights_only=True),
                torch.load(tmp_path / "spawn" / job.name / name, weights_only=True),
            )
    assert not multiprocessing.active_children()


@pytest.mark.fast
def test_bootstrap_retry_preserves_original_timing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real bootstrap propagation and strict serialized reuse ignore a new request."""
    local_leaderboard(monkeypatch, tmp_path)
    args, paths, _ = setup_training(tmp_path / "workspace")
    monkeypatch.setenv("CHIPIRON_TRAINING_TIMING_MODE", "host_enqueue")
    result = run_pipeline_training_stage(args, generation=1)
    status = load_pipeline_training_status_file(
        paths.pipeline_training_status_path_for_generation(1)
    )
    assert all(
        item.performance["training_timing_mode"] == "host_enqueue"
        for item in status.evaluator_results.values()
    )
    manifests = list(
        paths.work_dir.glob(
            "training_publications_v1/objects/*/models/**/morpion_manifest.json"
        )
    )
    assert manifests
    stored = {p: p.read_bytes() for p in manifests}
    assert all(
        json.loads(b)["metadata"]["timing"]["mode"] == "host_enqueue"
        for b in stored.values()
    )
    monkeypatch.setenv("CHIPIRON_TRAINING_TIMING_MODE", "invalid-new-request")
    assert run_pipeline_training_stage(args, generation=1) == result
    assert all(p.read_bytes() == b for p, b in stored.items())


@pytest.mark.fast
def test_raw_streaming_and_coarse_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fallback loops have the same mode; service completion is explicit, not .item()."""
    rows = rows_file(tmp_path)
    monkeypatch.setattr(
        service, "is_flat_morpion_training_model_kind", lambda _kind: False
    )
    waits: list[Any] = []
    monkeypatch.setattr(service, "synchronize_torch_device_if_needed", waits.append)
    _, metrics = train_morpion_regressor_streaming(
        MorpionStreamingTrainingArgs(
            MorpionTrainingArgs(
                rows, tmp_path / "raw", device="cpu", batch_size=8, num_epochs=1
            ),
            timing_mode="host_enqueue",
        )
    )
    assert metrics["timing_mode"] == "host_enqueue"
    assert "timing_train_host_enqueue_forward_s" in metrics
    assert waits == [torch.device("cpu")]


def test_actual_persistent_dispatch_uses_each_job_mode(tmp_path: Path) -> None:
    """Real dispatcher jobs own a dedicated session, including production cleanup."""
    import os
    import signal
    import subprocess
    import sys
    from contextlib import suppress
    from pathlib import Path

    assert not torch.cuda.is_initialized()
    environment = dict(os.environ)
    environment["MORPION_RESOURCE_MONITOR_PID"] = str(os.getpid())
    environment["MORPION_RESOURCE_MONITOR_START_TICKS"] = (
        Path("/proc/self/stat").read_text().rsplit(") ", 1)[1].split()[19]
    )
    code = (
        "import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
        "from tests.environments.test_morpion_training_timing import "
        "_persistent_dispatch_child; _persistent_dispatch_child(Path(sys.argv[2]))"
    )
    log = tmp_path / "dispatcher.log"
    with log.open("w") as output:
        child = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-B",
                "-c",
                code,
                str(Path(__file__).resolve().parents[2]),
                str(tmp_path),
            ],
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = child.wait(timeout=20)
        finally:
            if child.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    with suppress(ProcessLookupError):
                        os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=3)
            assert child.poll() is not None
    assert returncode == 0, log.read_text()
    receipt = json.loads((tmp_path / "dispatcher.json").read_text())
    assert receipt["pid"] == receipt["session"] == child.pid
    assert receipt["modes"] == ["host_enqueue", "synchronized"]
    assert receipt["gpu_initialized"] is False
    assert not torch.cuda.is_initialized()


def _persistent_dispatch_child(tmp_path: Path) -> None:
    """Keep real identity/descendant guards inside the session created by Popen."""
    import os
    from contextlib import nullcontext

    from chipiron.environments.morpion.bootstrap import launcher, persistent_worker
    from chipiron.learning.training_timing import bootstrap_training_timing_options
    from tests.environments.test_morpion_persistent_workers import (
        _launcher,
        _unmanaged_configuration,
    )

    assert os.getsid(0) == os.getpid()
    assert not torch.cuda.is_initialized()
    identity = persistent_worker.monitored_identity()
    assert identity["session"] == identity["pid"] == os.getpid()
    _unmanaged_configuration(tmp_path)
    (tmp_path / "run_state.json").write_text("{}")
    rows = rows_file(tmp_path)
    jobs = jobs_for(rows, tmp_path / "candidates", 1)
    observed = []

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(persistent_worker, "launcher_execution", nullcontext)

        def job(_args: Any, *, _job_id: int | None = None) -> int:
            assert _job_id is not None
            mode = "host_enqueue" if _job_id == 1 else "synchronized"
            monkeypatch.setenv("CHIPIRON_TRAINING_TIMING_MODE", mode)
            original = jobs[_job_id - 1]
            captured = replace(
                original,
                training=replace(
                    original.training, **bootstrap_training_timing_options()
                ),
            )
            observed.append(run_candidate(captured)[1]["timing_mode"])
            return 0

        monkeypatch.setattr(launcher, "_run_launcher_cli", job)
        assert persistent_worker.run_persistent_worker(_launcher(tmp_path)) == 0
    assert observed == ["host_enqueue", "synchronized"]
    assert not torch.cuda.is_initialized()
    (tmp_path / "dispatcher.json").write_text(
        json.dumps({
            "pid": os.getpid(),
            "session": os.getsid(0),
            "modes": observed,
            "gpu_initialized": torch.cuda.is_initialized(),
        })
    )

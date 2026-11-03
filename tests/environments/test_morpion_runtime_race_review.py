"""Independent process-overlap and failure-precedence acceptance for B3-R."""

import subprocess
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import pytest

from chipiron.environments.morpion.bootstrap import reevaluation_worker as worker
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    GrowthOwnershipBlockingIOError,
)
from chipiron.environments.morpion.bootstrap.model_activation import (
    activate_model,
    prepare_model_activation_request,
)
from chipiron.environments.morpion.bootstrap.pipeline.stages import (
    run_pipeline_training_stage,
)
from chipiron.environments.morpion.bootstrap.reevaluation_activation import (
    ReevaluationModelActivatedError,
    raise_for_committed_activation,
)
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from tests.environments.morpion_activation_helpers import setup_activation
from tests.environments.morpion_fenced_dataset_helpers import (
    local_leaderboard,
    run_dataset,
)
from tests.environments.test_morpion_model_activation_edges import propose

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


def test_real_process_publication_contender_releases_before_next_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An independent process holds the kernel lock; release, then publish normally."""
    local_leaderboard(monkeypatch, tmp_path)
    args, paths, _ = setup_activation(tmp_path / "work")
    before = paths.run_state_path.read_bytes()
    lock = paths.pipeline_dir / ".growth_patch_consumer.lock"
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import fcntl,sys; f=open(sys.argv[1],'a+'); "
                "fcntl.flock(f,fcntl.LOCK_EX); print('held',flush=True); sys.stdin.read(1)"
            ),
            str(lock),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert child.stdout is not None
    try:
        assert child.stdout.readline().strip() == "held"
        monkeypatch.setenv("MORPION_WORKER_OWNERSHIP_WAIT_SECONDS", "0.02")
        result = propose(args)
        assert result.reason == "ownership_contended" and not result.patch_written
        assert paths.run_state_path.read_bytes() == before
        assert not paths.pipeline_reevaluation_patch_path.exists()
        assert not paths.pipeline_reevaluation_cursor_path.exists()
    finally:
        child.communicate("x", timeout=10)
    assert child.returncode == 0
    assert propose(args).patch_written


@pytest.mark.parametrize("error", [OSError, GrowthOwnershipBlockingIOError])
def test_inference_error_stays_fatal_after_real_activation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: type[OSError]
) -> None:
    """A valid concurrent activation does not turn an inference failure into discard."""
    local_leaderboard(monkeypatch, tmp_path)
    args, paths, _ = setup_activation(tmp_path / "work")
    request = prepare_model_activation_request(paths, training_generation=1)
    authority = []
    evaluate = worker.MorpionActiveModelNodeReevaluationEvaluator.evaluate_node_window

    def fail(*positional: Any, **keywords: Any) -> Any:
        evaluate(*positional, **keywords)
        activate_model(paths, request)
        authority.append(paths.run_state_path.read_bytes())
        message = "independent inference failure after activation"
        raise error(message)

    monkeypatch.setattr(
        worker.MorpionActiveModelNodeReevaluationEvaluator, "evaluate_node_window", fail
    )
    with pytest.raises(error, match="independent inference failure"):
        propose(args)
    assert paths.run_state_path.read_bytes() == authority[0]
    assert not paths.pipeline_reevaluation_patch_path.exists()
    assert not paths.pipeline_reevaluation_cursor_path.exists()


def test_cleanup_error_overrides_benign_activation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ExitStack cleanup failure escapes even while unwinding a proven discard."""
    local_leaderboard(monkeypatch, tmp_path)
    args, paths, _ = setup_activation(tmp_path / "work")
    request = prepare_model_activation_request(paths, training_generation=1)
    pin = worker.pinned_generation_manifest
    proposal = worker.proposal_patch

    @contextmanager
    def broken_cleanup(**keywords: Any) -> Iterator[Any]:
        try:
            with pin(**keywords) as manifest:
                yield manifest
        finally:
            message = "independent cleanup failure"
            raise OSError(message)

    def overlap(*positional: Any, **keywords: Any) -> Any:
        result = proposal(*positional, **keywords)
        activate_model(paths, request)
        return result

    monkeypatch.setattr(worker, "pinned_generation_manifest", broken_cleanup)
    monkeypatch.setattr(worker, "proposal_patch", overlap)
    with pytest.raises(OSError, match="independent cleanup failure"):
        propose(args)
    assert not paths.pipeline_reevaluation_patch_path.exists()
    assert not paths.pipeline_reevaluation_cursor_path.exists()


def test_multiple_activations_require_every_transition_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify a real two-activation chain and refuse a damaged intermediate receipt."""
    local_leaderboard(monkeypatch, tmp_path)
    args, paths, before = setup_activation(tmp_path / "work")
    # Dataset/Training acceptance follows genesis, so capture current authority.
    before = load_bootstrap_run_state(paths.run_state_path)
    first = prepare_model_activation_request(paths, training_generation=1)
    middle = activate_model(paths, first)
    run_dataset(args, generation=middle.generation)
    run_pipeline_training_stage(args, generation=middle.generation)
    second = prepare_model_activation_request(paths, training_generation=2)
    after = activate_model(paths, second)
    authority = paths.run_state_path.read_bytes()
    with pytest.raises(ReevaluationModelActivatedError):
        raise_for_committed_activation(paths, before, after, boundary="review")
    assert paths.run_state_path.read_bytes() == authority
    from chipiron.environments.morpion.bootstrap.generation_publication_io import (
        publication_records,
    )

    records = publication_records(after.metadata)
    transition = next(r for r in records if "activation_request_sha256" in r)
    transition["activation_request_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="binding mismatch"):
        raise_for_committed_activation(paths, before, after, boundary="review")
    assert paths.run_state_path.read_bytes() == authority

"""Operational timing changes observation, never regression arithmetic or RNG."""

import annotationlib
import inspect
from typing import Any

import pytest
import torch

from chipiron.learning import training_timing
from chipiron.learning.supervised import TensorSupervisedBatch, regression
from chipiron.learning.timing import PhaseDurations


@pytest.mark.fast
def test_explicit_mode_defaults_and_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset operational settings do not create a new scientific recipe."""
    monkeypatch.delenv("CHIPIRON_TRAINING_TIMING_MODE", raising=False)
    assert training_timing.bootstrap_training_timing_options() == {}
    monkeypatch.setenv("CHIPIRON_TRAINING_TIMING_MODE", "host_enqueue")
    assert training_timing.bootstrap_training_timing_options() == {
        "timing_mode": "host_enqueue"
    }
    monkeypatch.setenv("CHIPIRON_TRAINING_TIMING_MODE", "invalid")
    with pytest.raises(ValueError, match="timing mode"):
        training_timing.bootstrap_training_timing_options()
    with pytest.raises(ValueError, match="CPU and CUDA"):
        training_timing.synchronize_training_phases("host_enqueue", torch.device("mps"))
    assert training_timing.synchronize_training_phases(
        "synchronized", torch.device("mps")
    )
    for fn in (regression.train_regression_batch, regression.evaluate_regression_batch):
        signature = inspect.signature(fn, annotation_format=annotationlib.Format.STRING)
        assert signature.parameters["synchronize_cuda"].default is True


@pytest.mark.fast
@pytest.mark.parametrize("synchronize", [True, False])
@pytest.mark.parametrize("timed", [True, False])
def test_phase_sync_counts_and_truthful_keys(
    monkeypatch: pytest.MonkeyPatch, synchronize: bool, timed: bool
) -> None:
    """Training has seven and evaluation three fine scopes; None has no timers."""
    waits: list[torch.device] = []
    monkeypatch.setattr(regression, "synchronize_torch_device_if_needed", waits.append)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters())
    batch = TensorSupervisedBatch(torch.ones(2, 2), torch.zeros(2, 1))
    durations = PhaseDurations() if timed else None
    regression.train_regression_batch(
        model=model,
        optimizer=optimizer,
        criterion=torch.nn.MSELoss(),
        batch=batch,
        device=torch.device("cpu"),
        timings=durations,
        synchronize_cuda=synchronize,
    )
    assert len(waits) == (14 if timed and synchronize else 0)
    regression.evaluate_regression_batch(
        model=model,
        batch=batch,
        device=torch.device("cpu"),
        timings=durations,
        synchronize_cuda=synchronize,
    )
    assert len(waits) == (20 if timed and synchronize else 0)
    if durations is not None:
        assert all(
            k.startswith("host_enqueue_") != synchronize for k in durations.as_dict()
        )
        assert len(durations.as_dict()) == 7


@pytest.mark.fast
def test_coarse_scope_completes_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Outstanding device work cannot escape the service's exception boundary."""
    waits: list[torch.device] = []
    monkeypatch.setattr(
        training_timing, "synchronize_torch_device_if_needed", waits.append
    )
    device = torch.device("cpu")
    with (
        pytest.raises(RuntimeError, match="injected"),
        training_timing.device_complete_scope(device),
    ):
        assert waits == [device]
        raise RuntimeError("injected")
    assert waits == [device, device]


@pytest.mark.fast
def test_cpu_adam_rng_exact_with_timing_modes() -> None:
    """Independent initializations retain dropout RNG, every update and Adam moments."""
    results: list[Any] = []
    with torch.random.fork_rng(devices=[]):
        for synchronize in (True, False):
            torch.manual_seed(731)
            model = torch.nn.Sequential(
                torch.nn.Linear(2, 4), torch.nn.Dropout(0.2), torch.nn.Linear(4, 1)
            )
            optimizer = torch.optim.Adam(model.parameters())
            batch = TensorSupervisedBatch(torch.ones(3, 2), torch.zeros(3, 1))
            stats = [
                regression.train_regression_batch(
                    model=model,
                    optimizer=optimizer,
                    criterion=torch.nn.MSELoss(),
                    batch=batch,
                    device=torch.device("cpu"),
                    timings=PhaseDurations(),
                    synchronize_cuda=synchronize,
                )
                for _ in range(3)
            ]
            results.append((
                model.state_dict(),
                optimizer.state_dict(),
                torch.get_rng_state(),
                stats,
            ))
    torch.testing.assert_close(results[0][:3], results[1][:3], rtol=0, atol=0)
    assert results[0][3] == results[1][3]

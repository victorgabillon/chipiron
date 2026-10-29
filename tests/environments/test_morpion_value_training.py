"""Training restart, unchanged bundle defaults, and canonical-cache contracts."""

import json
import pickle
import subprocess
import sys
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
import torch

from chipiron.environments.morpion.players.evaluators.datasets.state_cache import (
    file_sha256,
    prepare_state_cache,
)
from chipiron.environments.morpion.players.evaluators.neural_networks import (
    MorpionRegressorArgs,
    build_morpion_regressor,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.bundle import (
    load_morpion_model_bundle,
    save_morpion_model_bundle,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training import (
    value_training,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training.value_training import (
    EntityValueTrainingConfig,
    learning_rate_at_step,
    train_entity_value,
)
from chipiron.learning.torch_runtime import TorchDeviceUnavailableError
from tests.environments.morpion_nn_test_helpers import tiny_entity_model_args
from tests.environments.test_morpion_entity_token_cache import _build_jsonl_rows_file

if TYPE_CHECKING:
    from pathlib import Path


def test_canonical_cache_keeps_source_targets_and_reopens(tmp_path: Path) -> None:
    """Cached states preserve raw targets and survive worker serialization."""
    path, rows = _build_jsonl_rows_file(
        tmp_path, target_values=tuple(float(i) for i in range(10))
    )
    before = file_sha256(path)
    cache = prepare_state_cache(path, tmp_path / "states.sqlite", max_rows=10)
    assert len(cache) == len(rows)
    for index, row in enumerate(rows):
        assert cache[index][1] == row.target_value
    reopened = pickle.loads(pickle.dumps(cache))
    assert reopened[3] == cache[3]
    assert (
        prepare_state_cache(path, tmp_path / "states.sqlite", max_rows=10)[3]
        == cache[3]
    )
    assert file_sha256(path) == before
    subset = prepare_state_cache(
        path, tmp_path / "subset.sqlite", max_rows=10, row_indices=(1, 4, 9)
    )
    assert len(subset) == 3
    assert [subset[i][1] for i in range(3)] == [rows[i].target_value for i in (1, 4, 9)]
    assert (
        prepare_state_cache(
            path, tmp_path / "subset.sqlite", max_rows=10, row_indices=(1, 4, 9)
        )[1]
        == cache[4]
    )
    with pytest.raises(ValueError, match="differs"):
        prepare_state_cache(path, tmp_path / "states.sqlite", max_rows=5)


@pytest.mark.parametrize(
    ("warmup_fraction", "stop_after_steps"),
    [(0.05, 3), (0.25, 1), (0.25, 2), (0.25, 3)],
)
def test_mid_epoch_resume_matches_uninterrupted_with_dropout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    warmup_fraction: float,
    stop_after_steps: int,
) -> None:
    """Restart preserves sample order, augmentation, optimizer schedule and RNG."""
    path, _ = _build_jsonl_rows_file(
        tmp_path, target_values=tuple(float(i) for i in range(10))
    )
    states = prepare_state_cache(path, tmp_path / "states.sqlite", max_rows=10)
    config = EntityValueTrainingConfig(
        model=replace(
            tiny_entity_model_args(relational=True),
            entity_n_layer=1,
            entity_dropout_ratio=0.1,
            relation_bias_scale=0.25,
        ),
        num_epochs=2,
        batch_size=2,
        workers=0,
        device="cpu",
        d4_augmentation=True,
        symmetry_diagnostic_rows=2,
        warmup_fraction=warmup_fraction,
    )
    expected_rates = []
    actual_rates = []
    shared_update = value_training.backward_and_step

    def legacy_update(loss: torch.Tensor, optimizer: torch.optim.Optimizer) -> None:
        expected_rates.append(optimizer.param_groups[0]["lr"])
        torch.autograd.backward(loss)
        optimizer.step()

    def observed_update(loss: torch.Tensor, optimizer: torch.optim.Optimizer) -> None:
        actual_rates.append(optimizer.param_groups[0]["lr"])
        shared_update(loss, optimizer)

    monkeypatch.setattr(value_training, "backward_and_step", legacy_update)
    uninterrupted = train_entity_value(config, states, tmp_path / "full", provenance={})
    monkeypatch.setattr(value_training, "backward_and_step", observed_update)
    partial = train_entity_value(
        config,
        states,
        tmp_path / "resume",
        provenance={},
        stop_after_steps=stop_after_steps,
    )
    assert partial["status"] == "interrupted"
    assert not (tmp_path / "resume/result.json").exists()
    resumed = train_entity_value(config, states, tmp_path / "resume", provenance={})
    assert resumed["status"] == uninterrupted["status"] == "complete"
    first, _, _ = load_morpion_model_bundle(tmp_path / "full/bundle")
    second, _, _ = load_morpion_model_bundle(tmp_path / "resume/bundle")
    for name, weight in first.state_dict().items():
        assert torch.equal(weight, second.state_dict()[name]), name
    assert resumed["metrics"] == uninterrupted["metrics"]
    assert resumed["symmetry_consistency"] == uninterrupted["symmetry_consistency"]
    assert actual_rates == expected_rates
    assert len(actual_rates) == 8
    if warmup_fraction == 0.25:
        assert actual_rates[:3] == [
            config.learning_rate / 2,
            config.learning_rate,
            config.learning_rate,
        ]
    checkpoints = [
        torch.load(tmp_path / name / "checkpoint.pt", weights_only=False)
        for name in ("full", "resume")
    ]
    for key in ("model", "optimizer", "step", "torch_rng", "cuda_rng"):
        torch.testing.assert_close(
            checkpoints[0][key], checkpoints[1][key], rtol=0, atol=0
        )
    with pytest.raises(ValueError, match="differs"):
        train_entity_value(
            replace(config, seed=1), states, tmp_path / "resume", provenance={}
        )


def test_nonfinite_loss_is_rejected_before_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The standalone recipe retains its own finite-loss rejection boundary."""
    path, _ = _build_jsonl_rows_file(
        tmp_path, target_values=tuple(float(i) for i in range(10))
    )
    cached = prepare_state_cache(path, tmp_path / "states.sqlite", max_rows=10)
    states = tuple((state, float("nan")) for state, _ in cached)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("a non-finite loss reached the update")

    monkeypatch.setattr(value_training, "backward_and_step", forbidden)
    config = EntityValueTrainingConfig(
        model=tiny_entity_model_args(relational=True),
        num_epochs=1,
        batch_size=2,
        workers=0,
        device="cpu",
        symmetry_diagnostic_rows=0,
    )
    with pytest.raises(ValueError, match="Non-finite training loss"):
        train_entity_value(config, states, tmp_path / "run", provenance={})


@pytest.mark.parametrize("device", ["cuda", "mps", "xpu"])
def test_value_training_uses_shared_availability_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    """Reject unavailable devices at the resolver, before model construction."""
    path, _ = _build_jsonl_rows_file(
        tmp_path, target_values=tuple(float(i) for i in range(10))
    )
    states = prepare_state_cache(path, tmp_path / "states.sqlite", max_rows=10)
    config = EntityValueTrainingConfig(
        model=tiny_entity_model_args(relational=True), device=device
    )
    for api in (torch.cuda, torch.backends.mps, torch.xpu):
        monkeypatch.setattr(api, "is_available", lambda: False)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("unavailable device reached model construction")

    monkeypatch.setattr(value_training, "build_morpion_regressor", forbidden)
    with pytest.raises(TorchDeviceUnavailableError, match="unavailable"):
        train_entity_value(config, states, tmp_path / "run", provenance={})


def test_old_scale_default_and_new_scale_bundle_roundtrip(tmp_path: Path) -> None:
    """Absent scale metadata retains the old scale-one behavior."""
    args = tiny_entity_model_args(relational=True)
    model = build_morpion_regressor(args)
    save_morpion_model_bundle(model=model, model_args=args, output_dir=tmp_path / "old")
    path = tmp_path / "old/morpion_regressor_args.json"
    payload = json.loads(path.read_text())
    payload.pop("relation_bias_scale")
    payload.update(
        target_transform_enabled=False, target_mean=0.0, target_standard_deviation=1.0
    )
    path.write_text(json.dumps(payload))
    loaded, loaded_args, _ = load_morpion_model_bundle(tmp_path / "old")
    assert loaded_args.relation_bias_scale == 1.0
    for name, weight in model.state_dict().items():
        assert torch.equal(weight, loaded.state_dict()[name])
    scaled = replace(args, relation_bias_scale=0.25)
    save_morpion_model_bundle(
        model=build_morpion_regressor(scaled),
        model_args=scaled,
        output_dir=tmp_path / "new",
    )
    assert load_morpion_model_bundle(tmp_path / "new")[1].relation_bias_scale == 0.25
    with pytest.raises(ValueError):
        MorpionRegressorArgs(relation_bias_scale=float("nan"))


def test_historical_learning_rate_endpoints() -> None:
    """The first update, warmup boundary, and final update match the old recipe."""
    config = EntityValueTrainingConfig(model=tiny_entity_model_args(relational=True))
    assert learning_rate_at_step(config, 0, 100000) == pytest.approx(2e-7)
    assert learning_rate_at_step(config, 4999, 100000) == pytest.approx(0.001)
    assert learning_rate_at_step(config, 5000, 100000) == pytest.approx(0.001)
    assert learning_rate_at_step(config, 99999, 100000) == pytest.approx(1e-5)


def _check_validation_with_low_file_limit(tmp_path: Path) -> None:
    """Exercise real spawned workers with more batches than available handles."""
    import resource

    torch.set_num_threads(1)
    _, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, hard_limit))
    torch.multiprocessing.set_sharing_strategy("file_descriptor")
    values = tuple(float(index % 7) for index in range(800))
    path, _ = _build_jsonl_rows_file(tmp_path, target_values=values)
    states = prepare_state_cache(path, tmp_path / "states.sqlite", max_rows=800)
    config = EntityValueTrainingConfig(
        model=tiny_entity_model_args(relational=True),
        num_epochs=1,
        batch_size=1,
        workers=2,
        device="cpu",
        symmetry_diagnostic_rows=0,
    )
    result = train_entity_value(config, states, tmp_path / "run", provenance={})
    assert result["status"] == "complete"
    validation = torch.load(tmp_path / "run/validation.pt", weights_only=True)
    assert torch.equal(validation["row_indices"], torch.arange(4, 800, 5))
    assert torch.equal(validation["targets"].flatten(), torch.tensor(values)[4::5])
    assert torch.isfinite(validation["predictions"]).all()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux file-descriptor regression")
def test_validation_workers_do_not_exhaust_file_descriptors(tmp_path: Path) -> None:
    """A low descriptor limit must not prevent complete canonical validation."""
    code = (
        "import atomheart, anemone; import sys; from pathlib import Path; "
        "from tests.environments.test_morpion_value_training import "
        "_check_validation_with_low_file_limit; "
        "_check_validation_with_low_file_limit(Path(sys.argv[1]))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

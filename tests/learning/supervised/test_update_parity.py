"""Independent pre-refactor update oracles for the shared tensor mechanics."""

import copy
from dataclasses import replace

import pytest
import torch
from torch import nn

from chipiron.environments.morpion.players.evaluators.neural_networks import (
    MorpionRelationalEntityTokenConverter,
    build_morpion_regressor,
)
from chipiron.learning.optimization import backward_and_step, forward_tensors
from chipiron.learning.supervised import (
    RegressionBatchTrainStats,
    TensorSupervisedBatch,
    train_regression_batch,
)
from chipiron.learning.timing import PhaseDurations
from tests.environments.morpion_nn_test_helpers import tiny_entity_model_args
from tests.environments.test_morpion_entity_tokens import _make_one_step_state


def _legacy_batch_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    batch: TensorSupervisedBatch,
) -> RegressionBatchTrainStats:
    """Keep the previous CPU algorithm independent of the new primitives."""
    optimizer.zero_grad()
    predictions = model(*batch.get_model_input_tensors())
    targets = batch.get_target_value()
    loss = criterion(predictions, targets)
    loss.backward()
    optimizer.step()
    errors = predictions.detach() - targets
    return RegressionBatchTrainStats(
        loss=float(loss.detach().cpu().item()),
        squared_error_sum=float(torch.sum(errors * errors).item()),
        absolute_error_sum=float(torch.sum(torch.abs(errors)).item()),
        sample_count=int(batch.input_tensor.shape[0]),
        target_count=int(targets.numel()),
    )


def _case(recipe: str) -> tuple[nn.Module, nn.Module, list[TensorSupervisedBatch]]:
    """Use real relational inputs and dropout, plus ordinary multi-output batches."""
    if recipe == "relational_adamw":
        model = build_morpion_regressor(
            replace(
                tiny_entity_model_args(relational=True),
                entity_n_layer=1,
                entity_dropout_ratio=0.2,
                relation_bias_scale=0.25,
            )
        )
        tensors = MorpionRelationalEntityTokenConverter(max_tokens=64).state_to_tensors(
            _make_one_step_state()
        )
        batches = [
            TensorSupervisedBatch(
                input_tensor=tensors.token_tensor.unsqueeze(0).repeat(size, 1, 1),
                auxiliary_input_tensors=(
                    tensors.relation_triples.unsqueeze(0).repeat(size, 1, 1),
                ),
                target_tensor=torch.linspace(-0.5, 0.7, size).unsqueeze(1),
            )
            for size in (3, 2, 1, 3)
        ]
        return model, nn.MSELoss(), batches
    model = nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Dropout(0.2), nn.Linear(5, 2))
    batches = [
        TensorSupervisedBatch(torch.randn(size, 3), torch.randn(size, 2))
        for size in (3, 2, 1, 3)
    ]
    return model, nn.L1Loss() if recipe == "l1_sgd" else nn.MSELoss(), batches


def _optimizer(recipe: str, model: nn.Module) -> torch.optim.Optimizer:
    if recipe == "l1_sgd":
        return torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.7)
    if recipe == "mse_adam":
        return torch.optim.Adam(model.parameters(), lr=0.01)
    return torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.03)


@pytest.mark.parametrize("recipe", ["l1_sgd", "mse_adam", "relational_adamw"])
def test_multiple_updates_match_legacy_parameters_state_metrics_and_rng(
    recipe: str,
) -> None:
    """Every update retains predictions, optimizer moments, timing phases and RNG."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(817)
        actual, criterion, batches = _case(recipe)
        reference = copy.deepcopy(actual)
        actual_optimizer = _optimizer(recipe, actual)
        reference_optimizer = _optimizer(recipe, reference)
        for batch in batches:
            before_rng = torch.get_rng_state()
            expected = _legacy_batch_step(
                reference, reference_optimizer, criterion, batch
            )
            expected_rng = torch.get_rng_state()
            torch.set_rng_state(before_rng)
            timings = PhaseDurations()
            observed = train_regression_batch(
                model=actual,
                optimizer=actual_optimizer,
                criterion=criterion,
                batch=batch,
                device=torch.device("cpu"),
                timings=timings,
                synchronize_cuda=False,
            )
            assert observed == expected
            assert torch.equal(torch.get_rng_state(), expected_rng)
            torch.testing.assert_close(
                actual.state_dict(), reference.state_dict(), rtol=0, atol=0
            )
            torch.testing.assert_close(
                actual_optimizer.state_dict(),
                reference_optimizer.state_dict(),
                rtol=0,
                atol=0,
            )
            assert actual.training and reference.training
            assert list(timings.as_dict()) == [
                "batch_transfer",
                "zero_grad",
                "forward",
                "loss",
                "backward",
                "optimizer_step",
                "metric_accumulation",
            ]


def test_update_does_not_clear_gradients_extract_scalars_or_synchronize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The low-level update has no hidden metric or gradient-clearing policy."""
    parameter = nn.Parameter(torch.tensor(2.0))
    parameter.grad = torch.tensor(3.0)
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    loss = parameter.square()

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail(
            "unexpected scalar extraction, synchronization or gradient clearing"
        )

    with monkeypatch.context() as context:
        context.setattr(torch.Tensor, "item", forbidden)
        context.setattr(torch.Tensor, "cpu", forbidden)
        context.setattr(torch.cuda, "synchronize", forbidden)
        context.setattr(optimizer, "zero_grad", forbidden)
        assert backward_and_step(loss, optimizer) is None
    assert parameter.grad == 7.0
    assert float(parameter.detach()) == pytest.approx(1.3)


def test_update_leaves_finite_loss_policy_to_caller() -> None:
    """Only recipes that already reject non-finite loss should gain that policy."""
    parameter = nn.Parameter(torch.tensor(2.0))
    backward_and_step(parameter * float("nan"), torch.optim.SGD([parameter], lr=0.1))
    assert torch.isnan(parameter)


def test_forward_preserves_eval_mode_and_rng() -> None:
    """Forwarding must not enable dropout or consume extra random numbers."""
    model = nn.Sequential(nn.Dropout(0.8)).eval()
    inputs = torch.ones(3, 2)
    before = torch.get_rng_state()
    assert torch.equal(forward_tensors(model, (inputs,)), inputs)
    assert not model.training
    assert torch.equal(before, torch.get_rng_state())

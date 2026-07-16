"""Tests for Morpion neural-network diagnostics prediction adapters."""
# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHIPIRON_PACKAGE_ROOT = _REPO_ROOT / "src" / "chipiron"
_ATOMHEART_PACKAGE_ROOT = _REPO_ROOT.parent / "atomheart" / "src" / "atomheart"
_ANEMONE_PACKAGE_ROOT = _REPO_ROOT.parent / "anemone" / "src" / "anemone"
_MORPION_EVALUATORS_PACKAGE_ROOT = (
    _REPO_ROOT
    / "src"
    / "chipiron"
    / "environments"
    / "morpion"
    / "players"
    / "evaluators"
)

if "chipiron" not in sys.modules:
    _chipiron_stub = ModuleType("chipiron")
    _chipiron_stub.__path__ = [str(_CHIPIRON_PACKAGE_ROOT)]
    sys.modules["chipiron"] = _chipiron_stub

if "chipiron.environments.morpion.players.evaluators" not in sys.modules:
    _evaluators_stub = ModuleType("chipiron.environments.morpion.players.evaluators")
    _evaluators_stub.__path__ = [str(_MORPION_EVALUATORS_PACKAGE_ROOT)]
    sys.modules["chipiron.environments.morpion.players.evaluators"] = _evaluators_stub

if "atomheart" not in sys.modules:
    _atomheart_stub = ModuleType("atomheart")
    _atomheart_stub.__path__ = [str(_ATOMHEART_PACKAGE_ROOT)]
    sys.modules["atomheart"] = _atomheart_stub

if "anemone" not in sys.modules:
    _anemone_stub = ModuleType("anemone")
    _anemone_stub.__path__ = [str(_ANEMONE_PACKAGE_ROOT)]
    sys.modules["anemone"] = _anemone_stub

from atomheart.games.morpion import MorpionDynamics as AtomMorpionDynamics
from atomheart.games.morpion import initial_state as morpion_initial_state
from atomheart.games.morpion.checkpoints import MorpionStateCheckpointCodec

from chipiron.environments.morpion.learning import MorpionSupervisedRow
from chipiron.environments.morpion.players.evaluators.neural_networks.entity_tokens import (
    MORPION_ENTITY_TOKEN_MODEL_KIND,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.model import (
    MorpionRegressor,
    MorpionRegressorArgs,
    build_morpion_regressor,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training import (
    UnsupportedMorpionDiagnosticInputFormatError,
    predict_morpion_rows_for_diagnostics,
    try_predict_morpion_rows_for_diagnostics,
)


def _make_morpion_payload() -> dict[str, object]:
    """Build one real Morpion state payload for diagnostics tests."""
    dynamics = AtomMorpionDynamics()
    start_state = morpion_initial_state()
    first_action = dynamics.all_legal_actions(start_state)[0]
    next_state = dynamics.step(start_state, first_action).next_state
    codec = MorpionStateCheckpointCodec()
    return cast("dict[str, object]", codec.dump_state_ref(next_state))


def _rows(count: int = 2) -> tuple[MorpionSupervisedRow, ...]:
    """Build real Morpion supervised rows for diagnostics tests."""
    payload = _make_morpion_payload()
    return tuple(
        MorpionSupervisedRow(
            node_id=f"node-{index}",
            state_ref_payload=payload,
            target_value=float(index) / 2.0,
            is_terminal=True,
            is_exact=True,
            depth=index + 1,
            metadata={"source": "training-diagnostics-test"},
        )
        for index in range(count)
    )


def _constant_linear_regressor(value: float) -> MorpionRegressor:
    """Return one linear regressor with deterministic constant predictions."""
    model = MorpionRegressor(MorpionRegressorArgs(model_kind="linear"))
    linear = model.net
    with torch.no_grad():
        linear.weight.zero_()
        linear.bias.fill_(value)
    model.eval()
    return model


class _UnsupportedDiagnosticModel(torch.nn.Module):
    """Fake model family that diagnostics should skip cleanly."""

    args: SimpleNamespace

    def __init__(self) -> None:
        super().__init__()
        self.args = SimpleNamespace(model_kind="unknown_model_kind")

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Raise if diagnostics incorrectly reaches model forward."""
        raise AssertionError


class _ShapeMismatchDiagnosticModel(torch.nn.Module):
    """Fake model whose forward reports a clear input-shape mismatch."""

    args: SimpleNamespace

    def __init__(self) -> None:
        super().__init__()
        self.args = SimpleNamespace(model_kind="linear")

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Raise a shape-related ValueError like a model input adapter would."""
        raise ValueError(  # noqa: TRY003
            f"expected entity-token input shape, got {tuple(input_tensor.shape)}"
        )


def test_flat_model_diagnostics_return_predictions() -> None:
    """Flat linear diagnostics should produce one float prediction per row."""
    predictions = predict_morpion_rows_for_diagnostics(
        _constant_linear_regressor(0.75),
        _rows(3),
    )

    assert predictions == [0.75, 0.75, 0.75]
    assert all(isinstance(prediction, float) for prediction in predictions)


def test_safe_diagnostics_returns_skip_for_unknown_model_kind() -> None:
    """Safe diagnostics should skip unsupported model families cleanly."""
    result = try_predict_morpion_rows_for_diagnostics(
        _UnsupportedDiagnosticModel(),
        _rows(2),
    )

    assert result.skipped is True
    assert result.reason == UnsupportedMorpionDiagnosticInputFormatError.reason
    assert result.predictions == []
    assert result.detail is not None


def test_safe_diagnostics_returns_skip_for_input_shape_value_error() -> None:
    """Safe diagnostics should convert clear input-format ValueErrors to skips."""
    result = try_predict_morpion_rows_for_diagnostics(
        _ShapeMismatchDiagnosticModel(),
        _rows(1),
    )

    assert result.skipped is True
    assert result.reason == UnsupportedMorpionDiagnosticInputFormatError.reason
    assert result.predictions == []
    assert result.detail is not None


def test_entity_token_diagnostics_do_not_crash() -> None:
    """Entity-token diagnostics should either predict or skip without traceback."""
    model = build_morpion_regressor(
        MorpionRegressorArgs(
            model_kind=MORPION_ENTITY_TOKEN_MODEL_KIND,
            entity_max_tokens=128,
            entity_d_model=16,
            entity_n_head=4,
            entity_n_layer=1,
            entity_dim_feedforward=32,
        )
    )

    result = try_predict_morpion_rows_for_diagnostics(model, _rows(2))

    if result.skipped:
        assert result.reason == UnsupportedMorpionDiagnosticInputFormatError.reason
        assert result.predictions == []
    else:
        assert len(result.predictions) == 2
        assert all(isinstance(prediction, float) for prediction in result.predictions)

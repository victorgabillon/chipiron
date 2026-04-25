"""Tests for persisted Morpion bootstrap evaluator diagnostics."""
# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

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

from chipiron.environments.morpion.bootstrap.evaluator_diagnostics import (
    MorpionEvaluatorDiagnosticExample,
    MorpionEvaluatorTrainingDiagnostics,
    build_evaluator_training_diagnostics,
    diagnostics_generation_dir,
    diagnostics_history_path,
    diagnostics_path,
    load_evaluator_training_diagnostics,
    load_latest_evaluator_training_diagnostics,
    representative_row_indexes,
    save_evaluator_training_diagnostics,
)
from chipiron.environments.morpion.learning import (
    MorpionSupervisedRow,
    MorpionSupervisedRows,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.model import (
    MorpionRegressor,
    MorpionRegressorArgs,
)


def _make_morpion_payload() -> dict[str, object]:
    """Build one real Morpion state payload for diagnostics tests."""
    dynamics = AtomMorpionDynamics()
    start_state = morpion_initial_state()
    first_action = dynamics.all_legal_actions(start_state)[0]
    next_state = dynamics.step(start_state, first_action).next_state
    codec = MorpionStateCheckpointCodec()
    return dict(codec.dump_state_ref(next_state))


def _rows_bundle(target_values: tuple[float, ...]) -> MorpionSupervisedRows:
    """Build a small rows bundle with real Morpion state payloads."""
    payload = _make_morpion_payload()
    return MorpionSupervisedRows(
        rows=tuple(
            MorpionSupervisedRow(
                node_id=f"node-{index}",
                state_ref_payload=payload,
                target_value=target_value,
                is_terminal=True,
                is_exact=True,
                depth=index + 1,
                metadata={"source": "diagnostics-test"},
            )
            for index, target_value in enumerate(target_values)
        ),
        metadata={"bootstrap_generation": 7},
    )


def _constant_regressor(value: float) -> MorpionRegressor:
    """Return one linear regressor with deterministic constant predictions."""
    model = MorpionRegressor(MorpionRegressorArgs(model_kind="linear"))
    linear = model.net
    with torch.no_grad():
        linear.weight.zero_()
        linear.bias.fill_(value)
    model.eval()
    return model


def test_representative_row_indexes_cover_small_medium_and_large_datasets() -> None:
    """Representative row selection should follow deterministic scale windows."""
    assert representative_row_indexes(0) == ()
    assert representative_row_indexes(5) == (0, 1, 2, 3, 4)
    assert representative_row_indexes(25) == tuple(range(20))
    assert representative_row_indexes(125) == tuple(
        list(range(10)) + list(range(10, 20)) + list(range(100, 110))
    )


def test_diagnostics_json_round_trip_and_latest_loader(tmp_path: Path) -> None:
    """Saved diagnostics should load back exactly and appear in latest lookup."""
    diagnostics = MorpionEvaluatorTrainingDiagnostics(
        generation=48,
        evaluator_name="mlp_20",
        dataset_size=123,
        created_at="2026-04-24T09:00:00Z",
        representative_examples=[
            MorpionEvaluatorDiagnosticExample(
                row_index=0,
                node_id="node-0",
                state_tag=17,
                depth=2,
                target_value=1.5,
                prediction_before=None,
                prediction_after=1.25,
                abs_error_before=None,
                abs_error_after=0.25,
            )
        ],
        worst_examples=[],
        mae_before=None,
        mae_after=0.25,
        max_abs_error_before=None,
        max_abs_error_after=0.25,
    )
    target = diagnostics_path(tmp_path, 48, "mlp_20")
    save_evaluator_training_diagnostics(diagnostics, target)

    loaded = load_evaluator_training_diagnostics(target)
    latest = load_latest_evaluator_training_diagnostics(tmp_path)

    assert target.parent == diagnostics_generation_dir(tmp_path, 48)
    assert loaded == diagnostics
    assert latest == {"mlp_20": diagnostics}
    assert diagnostics_history_path(tmp_path, "mlp_20").name == "mlp_20_history.jsonl"


def test_worst_examples_are_sorted_by_after_training_abs_error() -> None:
    """Worst examples should rank rows by descending post-training absolute error."""
    diagnostics = build_evaluator_training_diagnostics(
        generation=7,
        evaluator_name="linear",
        rows=_rows_bundle((1.0, -3.0, 0.5)),
        created_at="2026-04-24T09:30:00Z",
        model_before=_constant_regressor(1.0),
        model_after=_constant_regressor(0.0),
    )

    assert [example.row_index for example in diagnostics.worst_examples] == [1, 0, 2]
    assert diagnostics.worst_examples[0].abs_error_after == 3.0
    assert diagnostics.mae_after is not None
    assert diagnostics.mae_before is not None

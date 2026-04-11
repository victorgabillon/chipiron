"""Tests for the Morpion supervised PyTorch dataset layer."""
# ruff: noqa: E402

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest
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

from anemone.training_export import TrainingNodeSnapshot, TrainingTreeSnapshot
from atomheart.games.morpion import MorpionDynamics as AtomMorpionDynamics
from atomheart.games.morpion import initial_state as morpion_initial_state
from atomheart.games.morpion.checkpoints import MorpionStateCheckpointCodec

from chipiron.environments.morpion.learning import (
    MalformedMorpionSupervisedRowsError,
    save_morpion_supervised_rows,
    training_tree_snapshot_to_morpion_supervised_rows,
)
from chipiron.environments.morpion.players.evaluators.datasets import (
    MorpionSupervisedDataset,
    MorpionSupervisedDatasetArgs,
    load_morpion_supervised_dataset,
)
from chipiron.environments.morpion.players.evaluators.neural_networks import (
    morpion_feature_names,
    morpion_input_dim,
    morpion_state_to_tensor,
)
from chipiron.environments.morpion.types import MorpionDynamics


def _make_morpion_payload() -> dict[str, object]:
    """Build one real Morpion checkpoint payload from a one-step state."""
    dynamics = AtomMorpionDynamics()
    start_state = morpion_initial_state()
    first_action = dynamics.all_legal_actions(start_state)[0]
    next_state = dynamics.step(start_state, first_action).next_state
    codec = MorpionStateCheckpointCodec()
    return cast("dict[str, object]", codec.dump_state_ref(next_state))


def _make_training_node(
    *,
    node_id: str,
    payload: dict[str, object],
    target_value: float,
) -> TrainingNodeSnapshot:
    """Build one export node that PR 5 will convert into a raw Morpion row."""
    return TrainingNodeSnapshot(
        node_id=node_id,
        parent_ids=(),
        child_ids=(),
        depth=2,
        state_ref_payload=payload,
        direct_value_scalar=target_value / 2.0,
        backed_up_value_scalar=target_value,
        is_terminal=False,
        is_exact=True,
        over_event_label=None,
        visit_count=5,
        metadata={"source": "dataset-test"},
    )


def _build_rows_file(
    tmp_path: Path,
    *,
    target_values: tuple[float, ...] = (1.25, -0.5),
) -> tuple[Path, tuple[TrainingNodeSnapshot, ...]]:
    """Build and persist a raw Morpion supervised-row artifact for tests."""
    payload = _make_morpion_payload()
    nodes = tuple(
        _make_training_node(
            node_id=f"node-{index}",
            payload=payload,
            target_value=target_value,
        )
        for index, target_value in enumerate(target_values)
    )
    snapshot = TrainingTreeSnapshot(
        root_node_id="node-0" if nodes else None,
        nodes=nodes,
        metadata={"format_kind": "training_tree_snapshot", "format_version": 1},
    )
    rows = training_tree_snapshot_to_morpion_supervised_rows(snapshot)
    path = tmp_path / "morpion_supervised_rows.json"
    save_morpion_supervised_rows(rows, path)
    return path, nodes


def test_dataset_loads_persisted_rows_and_exposes_correct_length(
    tmp_path: Path,
) -> None:
    """The dataset should eagerly load the persisted raw-row artifact."""
    path, _ = _build_rows_file(tmp_path)

    dataset = load_morpion_supervised_dataset(
        MorpionSupervisedDatasetArgs(file_name=path)
    )

    assert isinstance(dataset, MorpionSupervisedDataset)
    assert len(dataset) == 2


def test_one_sample_has_correct_tensor_types_and_shapes(tmp_path: Path) -> None:
    """One dataset sample should expose float32 tensors with stable shapes."""
    path, _ = _build_rows_file(tmp_path)
    dataset = MorpionSupervisedDataset(MorpionSupervisedDatasetArgs(file_name=path))

    input_tensor, target_tensor = dataset[0]

    assert isinstance(input_tensor, torch.Tensor)
    assert isinstance(target_tensor, torch.Tensor)
    assert input_tensor.dtype == torch.float32
    assert target_tensor.dtype == torch.float32
    assert input_tensor.ndim == 1
    assert input_tensor.shape == (morpion_input_dim(),)
    assert target_tensor.shape == (1,)


def test_input_tensor_matches_morpion_converter_directly(tmp_path: Path) -> None:
    """Dataset inputs should match the direct Morpion tensor-conversion path."""
    path, nodes = _build_rows_file(tmp_path, target_values=(1.25,))
    dataset = MorpionSupervisedDataset(MorpionSupervisedDatasetArgs(file_name=path))
    sample_input, _ = dataset[0]

    dynamics = MorpionDynamics()
    atom_state = MorpionStateCheckpointCodec().load_state_ref(nodes[0].state_ref_payload)
    chipiron_state = dynamics.wrap_atomheart_state(atom_state)
    expected_input = morpion_state_to_tensor(chipiron_state, dynamics=dynamics)

    torch.testing.assert_close(sample_input, expected_input)


def test_target_tensor_matches_row_target_value(tmp_path: Path) -> None:
    """Dataset targets should preserve the persisted raw row target value."""
    path, _ = _build_rows_file(tmp_path, target_values=(1.25,))
    dataset = MorpionSupervisedDataset(MorpionSupervisedDatasetArgs(file_name=path))

    _, target_tensor = dataset[0]

    torch.testing.assert_close(target_tensor, torch.tensor([1.25], dtype=torch.float32))


def test_feature_names_helper_is_stable(tmp_path: Path) -> None:
    """Dataset feature metadata should match the Morpion converter source of truth."""
    path, _ = _build_rows_file(tmp_path)
    dataset = MorpionSupervisedDataset(MorpionSupervisedDatasetArgs(file_name=path))

    assert dataset.feature_names() == morpion_feature_names()
    assert len(dataset.feature_names()) == dataset.input_dim


def test_repeated_indexing_is_deterministic(tmp_path: Path) -> None:
    """Repeated indexing should return equal tensors for the same sample."""
    path, _ = _build_rows_file(tmp_path, target_values=(1.25,))
    dataset = MorpionSupervisedDataset(MorpionSupervisedDatasetArgs(file_name=path))

    first_input, first_target = dataset[0]
    second_input, second_target = dataset[0]

    torch.testing.assert_close(first_input, second_input)
    torch.testing.assert_close(first_target, second_target)


def test_dataset_preserves_row_order(tmp_path: Path) -> None:
    """Distinct target values should come back in persisted row order."""
    path, _ = _build_rows_file(tmp_path, target_values=(1.25, -0.5))
    dataset = MorpionSupervisedDataset(MorpionSupervisedDatasetArgs(file_name=path))

    _, first_target = dataset[0]
    _, second_target = dataset[1]

    torch.testing.assert_close(first_target, torch.tensor([1.25], dtype=torch.float32))
    torch.testing.assert_close(second_target, torch.tensor([-0.5], dtype=torch.float32))


def test_malformed_persisted_rows_fail_loudly(tmp_path: Path) -> None:
    """Malformed persisted raw rows should raise through the dataset loader path."""
    path = tmp_path / "malformed_rows.json"
    path.write_text(json.dumps({"rows": {}}), encoding="utf-8")

    with pytest.raises(MalformedMorpionSupervisedRowsError):
        load_morpion_supervised_dataset(MorpionSupervisedDatasetArgs(file_name=path))

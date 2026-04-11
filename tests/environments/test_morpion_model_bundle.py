"""Tests for Morpion model definition, bundle IO, and minimal training."""
# ruff: noqa: E402

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest
import torch
from torch.utils.data import DataLoader

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
    save_morpion_supervised_rows,
    training_tree_snapshot_to_morpion_supervised_rows,
)
from chipiron.environments.morpion.players.evaluators.datasets import (
    MorpionSupervisedDataset,
    MorpionSupervisedDatasetArgs,
)
from chipiron.environments.morpion.players.evaluators.neural_networks import (
    MORPION_FEATURE_SCHEMA,
    MORPION_INPUT_DIM,
    MORPION_MANIFEST_FILE_NAME,
    MORPION_MODEL_ARGS_FILE_NAME,
    MORPION_MODEL_WEIGHTS_FILE_NAME,
    IncompatibleMorpionModelBundleError,
    MorpionRegressorArgs,
    build_morpion_regressor,
    load_morpion_model_bundle,
    load_morpion_regressor_for_inference,
    save_morpion_model_bundle,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.train import (
    MorpionTrainingArgs,
    train_morpion_regressor,
)


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
        metadata={"source": "model-bundle-test"},
    )


def _build_rows_file(
    tmp_path: Path,
    *,
    target_values: tuple[float, ...] = (1.25, -0.5),
) -> Path:
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
    return path


def test_linear_model_builds_with_correct_output_shape() -> None:
    """The default Morpion regressor should map batches to one scalar output."""
    model = build_morpion_regressor()
    dummy_batch = torch.randn(3, MORPION_INPUT_DIM)

    batch_output = model(dummy_batch)
    single_output = model(torch.randn(MORPION_INPUT_DIM))

    assert batch_output.shape == (3, 1)
    assert single_output.shape == (1, 1)


def test_mlp_model_builds_with_multiple_hidden_layers() -> None:
    """An MLP regressor should support multiple configured hidden layers."""
    model = build_morpion_regressor(
        MorpionRegressorArgs(model_kind="mlp", hidden_sizes=(8, 4))
    )

    output = model(torch.randn(2, MORPION_INPUT_DIM))

    assert output.shape == (2, 1)


def test_save_load_bundle_round_trip(tmp_path: Path) -> None:
    """Saving and loading a Morpion bundle should preserve args, manifest, and weights."""
    args = MorpionRegressorArgs(model_kind="linear")
    model = build_morpion_regressor(args)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.25)

    bundle_dir = tmp_path / "bundle"
    save_morpion_model_bundle(model, bundle_dir, model_args=args, metadata={"epoch": 1})
    loaded_model, loaded_args, loaded_manifest = load_morpion_model_bundle(bundle_dir)

    assert loaded_args == args
    assert loaded_manifest.game_kind == "morpion"
    assert loaded_manifest.input_dim == MORPION_INPUT_DIM
    assert loaded_manifest.feature_schema == MORPION_FEATURE_SCHEMA
    assert loaded_manifest.metadata["epoch"] == 1

    for original_parameter, loaded_parameter in zip(
        model.parameters(), loaded_model.parameters(), strict=True
    ):
        torch.testing.assert_close(original_parameter, loaded_parameter)


def test_manifest_incompatibility_fails_clearly(tmp_path: Path) -> None:
    """Loading should reject a Morpion bundle whose manifest no longer matches."""
    args = MorpionRegressorArgs(model_kind="linear")
    model = build_morpion_regressor(args)
    bundle_dir = tmp_path / "bundle"
    save_morpion_model_bundle(model, bundle_dir, model_args=args)

    manifest_path = bundle_dir / MORPION_MANIFEST_FILE_NAME
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_data["feature_schema"] = "wrong_schema"
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

    with pytest.raises(IncompatibleMorpionModelBundleError):
        load_morpion_model_bundle(bundle_dir)


def test_minimal_training_helper_runs_end_to_end(tmp_path: Path) -> None:
    """The first Morpion training helper should train, save, and report metrics."""
    dataset_file = _build_rows_file(tmp_path, target_values=(1.25, -0.5))
    output_dir = tmp_path / "trained_bundle"

    model, metrics = train_morpion_regressor(
        MorpionTrainingArgs(
            dataset_file=dataset_file,
            output_dir=output_dir,
            batch_size=2,
            num_epochs=1,
            learning_rate=1e-3,
            shuffle=False,
        )
    )

    assert output_dir.is_dir()
    assert (output_dir / MORPION_MODEL_WEIGHTS_FILE_NAME).is_file()
    assert (output_dir / MORPION_MODEL_ARGS_FILE_NAME).is_file()
    assert (output_dir / MORPION_MANIFEST_FILE_NAME).is_file()
    assert "final_loss" in metrics
    assert metrics["num_samples"] == 2.0
    assert metrics["num_epochs"] == 1.0

    dataset = MorpionSupervisedDataset(
        MorpionSupervisedDatasetArgs(file_name=dataset_file)
    )
    sample_input, _ = dataset[0]
    output = model(sample_input)
    assert output.shape == (1, 1)


def test_dataloader_batch_collation_shapes_are_stable(tmp_path: Path) -> None:
    """Default DataLoader collation should batch Morpion samples as expected."""
    dataset_file = _build_rows_file(tmp_path, target_values=(1.25, -0.5))
    dataset = MorpionSupervisedDataset(
        MorpionSupervisedDatasetArgs(file_name=dataset_file)
    )
    data_loader = DataLoader(dataset, batch_size=2, shuffle=False)

    batch = next(iter(data_loader))

    assert batch.get_input_layer().shape == (2, MORPION_INPUT_DIM)
    assert batch.get_target_value().shape == (2, 1)


def test_loaded_trained_model_works_for_inference(tmp_path: Path) -> None:
    """A saved trained Morpion bundle should load back for inference cleanly."""
    dataset_file = _build_rows_file(tmp_path, target_values=(1.25, -0.5))
    output_dir = tmp_path / "trained_bundle"
    train_morpion_regressor(
        MorpionTrainingArgs(
            dataset_file=dataset_file,
            output_dir=output_dir,
            batch_size=2,
            num_epochs=1,
            learning_rate=1e-3,
            shuffle=False,
        )
    )

    model = load_morpion_regressor_for_inference(output_dir)
    dataset = MorpionSupervisedDataset(
        MorpionSupervisedDatasetArgs(file_name=dataset_file)
    )
    sample_input, _ = dataset[0]
    output = model(sample_input)

    assert model.training is False
    assert output.shape == (1, 1)


def test_load_bundle_accepts_legacy_hidden_dim_payload(tmp_path: Path) -> None:
    """Loading should migrate older saved model args that used `hidden_dim`."""
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    args_path = bundle_dir / MORPION_MODEL_ARGS_FILE_NAME
    args_path.write_text(
        json.dumps(
            {
                "model_kind": "mlp",
                "input_dim": MORPION_INPUT_DIM,
                "hidden_dim": 8,
            }
        ),
        encoding="utf-8",
    )

    model = build_morpion_regressor(
        MorpionRegressorArgs(model_kind="mlp", hidden_sizes=(8,))
    )
    save_morpion_model_bundle(
        model,
        bundle_dir,
        model_args=MorpionRegressorArgs(model_kind="mlp", hidden_sizes=(8,)),
    )
    args_path.write_text(
        json.dumps(
            {
                "model_kind": "mlp",
                "input_dim": MORPION_INPUT_DIM,
                "hidden_dim": 8,
            }
        ),
        encoding="utf-8",
    )

    _loaded_model, loaded_args, _loaded_manifest = load_morpion_model_bundle(bundle_dir)

    assert loaded_args.hidden_sizes == (8,)

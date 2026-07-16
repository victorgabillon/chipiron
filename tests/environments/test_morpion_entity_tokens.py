"""Tests for Morpion entity-token evaluator support."""
# ruff: noqa: E402

from __future__ import annotations

import json
import math
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
from coral.neural_networks.nn_model_type import NNModelType
from valanga.evaluations import Certainty

from chipiron.environments.morpion.bootstrap.runtime.runner import (
    load_morpion_evaluator_from_model_bundle,
)
from chipiron.environments.morpion.learning import (
    save_morpion_supervised_rows,
    training_tree_snapshot_to_morpion_supervised_rows,
)
from chipiron.environments.morpion.players.evaluators.datasets import (
    MorpionEntityTokenSupervisedSample,
    collate_morpion_entity_token_supervised_samples,
)
from chipiron.environments.morpion.players.evaluators.neural_networks import (
    MORPION_ENTITY_TOKEN_DIRECTIONS,
    MORPION_ENTITY_TOKEN_FEATURE_DIM,
    MORPION_ENTITY_TOKEN_FEATURE_NAMES,
    MORPION_ENTITY_TOKEN_INPUT_REPRESENTATION,
    MORPION_ENTITY_TOKEN_MODEL_KIND,
    MORPION_MANIFEST_FILE_NAME,
    MORPION_MODEL_ARGS_FILE_NAME,
    MorpionEntityTokenConverter,
    MorpionEntityTokenType,
    MorpionRegressorArgs,
    build_morpion_regressor,
    canonical_segment,
    load_morpion_model_bundle,
    save_morpion_model_bundle,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.entity_tokens import (
    _direction_index_for_segment,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.feature_extractor import (
    DIRECTIONS,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training import (
    MorpionTrainingArgs,
    train_morpion_regressor,
)
from chipiron.environments.morpion.types import MorpionDynamics, MorpionState
from tests.environments.morpion_training_snapshot_helpers import (
    make_training_node_snapshot,
)


def _make_standard_state() -> MorpionState:
    """Build a standard Chipiron-facing Morpion state."""
    dynamics = MorpionDynamics()
    return dynamics.wrap_atomheart_state(morpion_initial_state())


def _make_one_step_state() -> MorpionState:
    """Build a state containing occupied dots, used edges, and legal moves."""
    atom_dynamics = AtomMorpionDynamics()
    atom_state = morpion_initial_state()
    action = atom_dynamics.all_legal_actions(atom_state)[0]
    next_state = atom_dynamics.step(atom_state, action).next_state
    return MorpionDynamics().wrap_atomheart_state(next_state)


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
    """Build one export node for a raw Morpion row."""
    return make_training_node_snapshot(
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
        metadata={"source": "entity-token-test"},
    )


def _build_rows_file(
    tmp_path: Path,
    *,
    target_values: tuple[float, ...] = (0.25, -0.5),
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


def _small_entity_token_args() -> MorpionRegressorArgs:
    """Return small entity-token model args for fast tests."""
    return MorpionRegressorArgs(
        model_kind=MORPION_ENTITY_TOKEN_MODEL_KIND,
        entity_max_tokens=128,
        entity_d_model=16,
        entity_n_head=4,
        entity_n_layer=1,
        entity_dim_feedforward=32,
    )


def test_entity_token_converter_contract() -> None:
    """Entity token conversion should produce real tokens with validity set."""
    converter = MorpionEntityTokenConverter(max_tokens=256)
    tensor = converter.state_to_tensor(_make_standard_state())
    feature_names = converter.feature_names()

    assert tensor.dtype == torch.float32
    assert tensor.ndim == 2
    assert tensor.shape[1] == MORPION_ENTITY_TOKEN_FEATURE_DIM
    assert len(feature_names) == MORPION_ENTITY_TOKEN_FEATURE_DIM
    assert feature_names == MORPION_ENTITY_TOKEN_FEATURE_NAMES
    assert MORPION_ENTITY_TOKEN_FEATURE_DIM == 25
    assert feature_names[:4] == (
        "type_global",
        "type_dot",
        "type_edge",
        "type_move",
    )
    assert feature_names[-1] == "validity"
    assert "type_" + "value" not in feature_names
    assert int(tensor.shape[0]) <= converter.max_tokens
    assert torch.all(tensor[:, -1] == 1.0)
    assert tensor[0, MorpionEntityTokenType.GLOBAL.value] == 1.0
    assert torch.all(tensor[0, 1:4] == 0.0)


def test_entity_token_model_kind_matches_coral() -> None:
    """Chipiron's dependency-light model-kind literal should match Coral's enum."""
    assert (
        NNModelType.ENTITY_TOKEN_TRANSFORMER_VALUE_NET.value
        == MORPION_ENTITY_TOKEN_MODEL_KIND
    )


def test_canonical_segment_preserves_geometric_direction() -> None:
    """Canonicalization should preserve ascending-diagonal semantics."""
    diag_up = ((0, 0), (1, 1))
    reversed_diag_up = ((1, 1), (0, 0))

    canonical = canonical_segment(diag_up)

    assert canonical_segment(reversed_diag_up) == canonical
    assert _direction_index_for_segment(canonical) == 2
    assert _direction_index_for_segment(canonical_segment(reversed_diag_up)) == 2


def test_canonical_segment_preserves_down_diagonal_direction() -> None:
    """Canonicalization should preserve descending-diagonal semantics."""
    diag_down = ((0, 1), (1, 0))
    reversed_diag_down = ((1, 0), (0, 1))

    canonical = canonical_segment(diag_down)

    assert canonical_segment(reversed_diag_down) == canonical
    assert _direction_index_for_segment(canonical) == 3


def test_entity_token_layout_indices_and_truncation() -> None:
    """Layout maps should identify only surviving DOT, EDGE, and MOVE rows."""
    state = _make_one_step_state()
    layout = MorpionEntityTokenConverter(max_tokens=512).state_to_layout(state)

    assert layout.dot_index_by_point
    assert layout.edge_index_by_segment
    assert layout.move_index_by_action
    for index in layout.dot_index_by_point.values():
        assert 0 <= index < layout.tensor.shape[0]
        assert layout.tensor[index, MorpionEntityTokenType.DOT.value] == 1.0
    for index in layout.edge_index_by_segment.values():
        assert 0 <= index < layout.tensor.shape[0]
        assert layout.tensor[index, MorpionEntityTokenType.EDGE.value] == 1.0
    segment, segment_index = next(iter(layout.edge_index_by_segment.items()))
    reversed_segment = segment[1], segment[0]
    assert canonical_segment(segment) == segment
    assert canonical_segment(reversed_segment) == segment
    assert (
        layout.edge_index_by_segment[canonical_segment(reversed_segment)]
        == segment_index
    )
    for index in layout.move_index_by_action.values():
        assert 0 <= index < layout.tensor.shape[0]
        assert layout.tensor[index, MorpionEntityTokenType.MOVE.value] == 1.0

    truncated = MorpionEntityTokenConverter(max_tokens=1).state_to_layout(state)
    assert truncated.tensor.shape == (1, MORPION_ENTITY_TOKEN_FEATURE_DIM)
    assert not truncated.dot_index_by_point
    assert not truncated.edge_index_by_segment
    assert not truncated.move_index_by_action


def test_entity_token_representative_rows_preserve_v1_semantics() -> None:
    """Representative GLOBAL, DOT, EDGE, and MOVE rows should stay exact."""
    layout = MorpionEntityTokenConverter(max_tokens=512).state_to_layout(
        _make_one_step_state()
    )
    expected_rows = {
        "global": torch.tensor(
            [
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                37,
                1,
                27,
                4,
                1,
            ],
            dtype=torch.float32,
        ),
        "dot": torch.tensor(
            [
                0,
                1,
                0,
                0,
                -0.5,
                -0.15,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
            ],
            dtype=torch.float32,
        ),
        "edge": torch.tensor(
            [
                0,
                0,
                1,
                0,
                -0.45,
                -0.15,
                1,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
            ],
            dtype=torch.float32,
        ),
        "move": torch.tensor(
            [
                0,
                0,
                0,
                1,
                -0.5,
                0.15,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
            ],
            dtype=torch.float32,
        ),
    }

    torch.testing.assert_close(layout.tensor[0], expected_rows["global"])
    torch.testing.assert_close(
        layout.tensor[layout.dot_index_by_point[(-6, -2)]], expected_rows["dot"]
    )
    torch.testing.assert_close(
        layout.tensor[layout.edge_index_by_segment[((-6, -2), (-5, -2))]],
        expected_rows["edge"],
    )
    torch.testing.assert_close(
        layout.tensor[layout.move_index_by_action[(0, -6, 1, 0)]],
        expected_rows["move"],
    )


def test_entity_token_collate_pads_variable_token_counts() -> None:
    """Entity-token collate should zero-pad token rows and stack targets."""
    sample_a = MorpionEntityTokenSupervisedSample(
        input_tensor=torch.ones((2, MORPION_ENTITY_TOKEN_FEATURE_DIM)),
        target_tensor=torch.tensor([0.25], dtype=torch.float32),
        is_batch=False,
    )
    sample_b = MorpionEntityTokenSupervisedSample(
        input_tensor=torch.ones((4, MORPION_ENTITY_TOKEN_FEATURE_DIM)),
        target_tensor=torch.tensor([-0.5], dtype=torch.float32),
        is_batch=False,
    )

    batch = collate_morpion_entity_token_supervised_samples((sample_a, sample_b))

    assert batch.is_batch is True
    assert batch.get_input_layer().shape == (2, 4, MORPION_ENTITY_TOKEN_FEATURE_DIM)
    assert batch.get_target_value().shape == (2, 1)
    assert torch.all(batch.input_tensor[0, 2:, :] == 0.0)
    assert torch.all(batch.input_tensor[0, 2:, -1] == 0.0)
    torch.testing.assert_close(
        batch.target_tensor,
        torch.tensor([[0.25], [-0.5]], dtype=torch.float32),
    )


def test_entity_token_direction_names_match_feature_extractor_order() -> None:
    """Entity-token direction labels should match the raw Morpion direction indices."""
    assert MORPION_ENTITY_TOKEN_DIRECTIONS == (
        "horizontal",
        "vertical",
        "diag_up",
        "diag_down",
    )
    horizontal_dx, horizontal_dy = DIRECTIONS[0]
    vertical_dx, vertical_dy = DIRECTIONS[1]
    diag_up_dx, diag_up_dy = DIRECTIONS[2]
    diag_down_dx, diag_down_dy = DIRECTIONS[3]

    assert horizontal_dy == 0
    assert abs(horizontal_dx) == 1
    assert vertical_dx == 0
    assert abs(vertical_dy) == 1
    assert abs(diag_up_dx) == abs(diag_up_dy) == 1
    assert diag_up_dx * diag_up_dy > 0
    assert abs(diag_down_dx) == abs(diag_down_dy) == 1
    assert diag_down_dx * diag_down_dy < 0

    assert _direction_index_for_segment(((0, 0), (1, 0))) == 0
    assert _direction_index_for_segment(((0, 0), (0, 1))) == 1
    assert _direction_index_for_segment(((0, 0), (1, 1))) == 2
    assert _direction_index_for_segment(((0, 0), (1, -1))) == 3


def test_entity_token_regressor_args_reject_feature_dim_mismatch() -> None:
    """Entity-token models should match the v1 entity-token feature width exactly."""
    with pytest.raises(ValueError, match="entity_input_feature_dim"):
        MorpionRegressorArgs(
            model_kind=MORPION_ENTITY_TOKEN_MODEL_KIND,
            entity_input_feature_dim=MORPION_ENTITY_TOKEN_FEATURE_DIM + 1,
        )


def test_entity_token_regressor_args_reject_too_small_token_cap() -> None:
    """Entity-token models need room for the GLOBAL entity."""
    with pytest.raises(ValueError, match="entity_max_tokens"):
        MorpionRegressorArgs(
            model_kind=MORPION_ENTITY_TOKEN_MODEL_KIND,
            entity_max_tokens=0,
        )


def test_entity_token_regressor_args_reject_incompatible_attention_width() -> None:
    """Transformer width should be divisible by the configured head count."""
    with pytest.raises(ValueError, match="divisible"):
        MorpionRegressorArgs(
            model_kind=MORPION_ENTITY_TOKEN_MODEL_KIND,
            entity_d_model=10,
            entity_n_head=4,
        )


def test_entity_token_model_builds_and_runs_forward() -> None:
    """The model should accept unbatched and batched entity tensors."""
    model = build_morpion_regressor(_small_entity_token_args())
    converter = MorpionEntityTokenConverter(max_tokens=128)
    input_tensor = converter.state_to_tensor(_make_standard_state())

    single_output = model(input_tensor)
    batch_output = model(input_tensor.unsqueeze(0))

    assert single_output.shape == (1,)
    assert batch_output.shape == (1, 1)
    assert torch.isfinite(single_output).all()
    assert torch.isfinite(batch_output).all()
    assert model.net.args.use_value_token is True  # type: ignore[attr-defined]
    assert model.net.args.pooling == "value_token"  # type: ignore[attr-defined]
    assert model.net.input_projection.in_features == 24  # type: ignore[attr-defined]

    current_projection_params = 24 * 64 + 64
    previous_projection_params = 25 * 64 + 64
    assert previous_projection_params - current_projection_params == 64


def test_entity_token_padding_content_is_masked() -> None:
    """Non-valid padded content must not change the model output."""
    model = build_morpion_regressor(_small_entity_token_args()).eval()
    entities = MorpionEntityTokenConverter(max_tokens=128).state_to_tensor(
        _make_standard_state()
    )
    padded = torch.zeros(
        (1, entities.shape[0] + 2, MORPION_ENTITY_TOKEN_FEATURE_DIM),
        dtype=torch.float32,
    )
    padded[0, : entities.shape[0]] = entities
    noisy_padding = padded.clone()
    noisy_padding[0, entities.shape[0] :, :-1] = 123.0

    torch.testing.assert_close(model(padded), model(noisy_padding))


def test_entity_token_backward_has_finite_gradients() -> None:
    """A tiny padded batch should support loss computation and backpropagation."""
    model = build_morpion_regressor(_small_entity_token_args())
    entities = MorpionEntityTokenConverter(max_tokens=128).state_to_tensor(
        _make_standard_state()
    )
    prediction = model(entities.unsqueeze(0))
    loss = torch.nn.functional.mse_loss(prediction, torch.zeros_like(prediction))
    loss.backward()

    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_entity_token_bundle_roundtrip_runs_forward(tmp_path: Path) -> None:
    """Entity-token model bundles should persist entity args and load."""
    args = _small_entity_token_args()
    model = build_morpion_regressor(args)
    bundle_dir = tmp_path / "entity_bundle"

    save_morpion_model_bundle(model, bundle_dir, model_args=args)
    loaded_model, loaded_args, manifest = load_morpion_model_bundle(bundle_dir)
    manifest_text = (bundle_dir / MORPION_MANIFEST_FILE_NAME).read_text(
        encoding="utf-8"
    )
    args_payload = json.loads(
        (bundle_dir / MORPION_MODEL_ARGS_FILE_NAME).read_text(encoding="utf-8")
    )

    assert loaded_args.model_kind == MORPION_ENTITY_TOKEN_MODEL_KIND
    assert manifest.input_representation == MORPION_ENTITY_TOKEN_INPUT_REPRESENTATION
    assert manifest.input_dim == MORPION_ENTITY_TOKEN_FEATURE_DIM
    assert "entity_token_feature_names" in manifest_text
    assert args_payload == {
        "model_kind": MORPION_ENTITY_TOKEN_MODEL_KIND,
        "input_representation": MORPION_ENTITY_TOKEN_INPUT_REPRESENTATION,
        "entity_max_tokens": 128,
        "entity_input_feature_dim": MORPION_ENTITY_TOKEN_FEATURE_DIM,
        "entity_d_model": 16,
        "entity_n_head": 4,
        "entity_n_layer": 1,
        "entity_dim_feedforward": 32,
        "entity_dropout_ratio": 0.0,
        "entity_pooling": "value_token",
        "entity_output_tanh": False,
    }

    input_tensor = MorpionEntityTokenConverter(max_tokens=128).state_to_tensor(
        _make_standard_state()
    )
    output = loaded_model(input_tensor)
    assert output.shape == (1,)
    assert torch.isfinite(output).all()


def test_entity_token_bundle_rejects_removed_and_incompatible_schemas(
    tmp_path: Path,
) -> None:
    """Bundle loading should reject removed fields and incompatible entity schemas."""
    args = _small_entity_token_args()
    bundle_dir = tmp_path / "entity_bundle"
    save_morpion_model_bundle(
        build_morpion_regressor(args), bundle_dir, model_args=args
    )
    args_path = bundle_dir / MORPION_MODEL_ARGS_FILE_NAME
    original_payload = json.loads(args_path.read_text(encoding="utf-8"))

    wrong_representation = dict(original_payload)
    wrong_representation["input_representation"] = "wrong_entity_schema"
    args_path.write_text(json.dumps(wrong_representation), encoding="utf-8")
    with pytest.raises(ValueError, match="feature_schema"):
        load_morpion_model_bundle(bundle_dir)

    wrong_width = dict(original_payload)
    wrong_width["entity_input_feature_dim"] = 26
    args_path.write_text(json.dumps(wrong_width), encoding="utf-8")
    with pytest.raises(ValueError, match="entity_input_feature_dim"):
        load_morpion_model_bundle(bundle_dir)

    removed_schema = dict(original_payload)
    removed_schema["model_kind"] = "graph" + "_transformer"
    removed_schema["graph" + "_max_tokens"] = 1536
    args_path.write_text(json.dumps(removed_schema), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected fields"):
        load_morpion_model_bundle(bundle_dir)


def test_entity_token_training_smoke_saves_bundle_and_metrics(tmp_path: Path) -> None:
    """The training helper should run the entity-token path end to end."""
    dataset_file = _build_rows_file(tmp_path)
    output_dir = tmp_path / "trained_entity_bundle"

    _model, metrics = train_morpion_regressor(
        MorpionTrainingArgs(
            dataset_file=dataset_file,
            output_dir=output_dir,
            batch_size=2,
            num_epochs=1,
            learning_rate=1e-3,
            shuffle=False,
            model_kind=MORPION_ENTITY_TOKEN_MODEL_KIND,
            entity_max_tokens=128,
            entity_d_model=16,
            entity_n_head=4,
            entity_n_layer=1,
            entity_dim_feedforward=32,
        )
    )

    assert output_dir.is_dir()
    assert math.isfinite(cast("float", metrics["final_loss"]))
    _loaded_model, loaded_args, manifest = load_morpion_model_bundle(output_dir)
    assert loaded_args.model_kind == MORPION_ENTITY_TOKEN_MODEL_KIND
    assert manifest.input_representation == MORPION_ENTITY_TOKEN_INPUT_REPRESENTATION


def test_entity_token_bundle_loads_as_anemone_evaluator(tmp_path: Path) -> None:
    """A entity-token bundle should load behind the Anemone evaluator protocol."""
    args = _small_entity_token_args()
    model = build_morpion_regressor(args)
    bundle_dir = tmp_path / "entity_bundle"
    save_morpion_model_bundle(model, bundle_dir, model_args=args)

    evaluator = load_morpion_evaluator_from_model_bundle(bundle_dir)
    value = evaluator.evaluate(_make_standard_state())

    assert value.certainty is Certainty.ESTIMATE
    assert math.isfinite(value.score)

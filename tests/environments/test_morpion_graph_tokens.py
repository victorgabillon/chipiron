"""Tests for Morpion graph-token evaluator support."""
# ruff: noqa: E402

from __future__ import annotations

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
_CORAL_SRC_ROOT = _REPO_ROOT.parent / "coral" / "src"
_MORPION_EVALUATORS_PACKAGE_ROOT = (
    _REPO_ROOT
    / "src"
    / "chipiron"
    / "environments"
    / "morpion"
    / "players"
    / "evaluators"
)

if str(_CORAL_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_CORAL_SRC_ROOT))

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
from valanga.evaluations import Certainty

from chipiron.environments.morpion.bootstrap.runtime.runner import (
    load_morpion_evaluator_from_model_bundle,
)
from chipiron.environments.morpion.learning import (
    save_morpion_supervised_rows,
    training_tree_snapshot_to_morpion_supervised_rows,
)
from chipiron.environments.morpion.players.evaluators.datasets import (
    MorpionGraphSupervisedSample,
    collate_morpion_graph_supervised_samples,
)
from chipiron.environments.morpion.players.evaluators.neural_networks import (
    MORPION_GRAPH_DIRECTIONS,
    MORPION_GRAPH_INPUT_REPRESENTATION,
    MORPION_GRAPH_MODEL_KIND,
    MORPION_GRAPH_TOKEN_FEATURE_DIM,
    MORPION_GRAPH_TOKEN_FEATURE_NAMES,
    MORPION_MANIFEST_FILE_NAME,
    MorpionGraphTokenConverter,
    MorpionGraphTokenType,
    MorpionRegressorArgs,
    build_morpion_regressor,
    load_morpion_model_bundle,
    save_morpion_model_bundle,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.feature_extractor import (
    DIRECTIONS,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.graph_tokens import (
    _direction_index_for_segment,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.train import (
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
        metadata={"source": "graph-token-test"},
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


def _small_graph_args() -> MorpionRegressorArgs:
    """Return small graph model args for fast tests."""
    return MorpionRegressorArgs(
        model_kind=MORPION_GRAPH_MODEL_KIND,
        graph_max_tokens=128,
        graph_d_model=16,
        graph_n_head=4,
        graph_n_layer=1,
        graph_dim_feedforward=32,
    )


def test_graph_token_converter_contract() -> None:
    """Graph token conversion should produce real tokens with validity set."""
    converter = MorpionGraphTokenConverter(max_tokens=256)
    tensor = converter.state_to_tensor(_make_standard_state())
    feature_names = converter.feature_names()

    assert tensor.dtype == torch.float32
    assert tensor.ndim == 2
    assert tensor.shape[1] == MORPION_GRAPH_TOKEN_FEATURE_DIM
    assert len(feature_names) == MORPION_GRAPH_TOKEN_FEATURE_DIM
    assert feature_names == MORPION_GRAPH_TOKEN_FEATURE_NAMES
    assert feature_names[-1] == "validity"
    assert int(tensor.shape[0]) <= converter.max_tokens
    assert torch.all(tensor[:, -1] == 1.0)
    assert tensor[0, MorpionGraphTokenType.VALUE.value] == 1.0
    assert tensor[1, MorpionGraphTokenType.GLOBAL.value] == 1.0


def test_graph_collate_pads_variable_token_counts() -> None:
    """Graph collate should zero-pad token rows and stack targets."""
    sample_a = MorpionGraphSupervisedSample(
        input_tensor=torch.ones((2, MORPION_GRAPH_TOKEN_FEATURE_DIM)),
        target_tensor=torch.tensor([0.25], dtype=torch.float32),
    )
    sample_b = MorpionGraphSupervisedSample(
        input_tensor=torch.ones((4, MORPION_GRAPH_TOKEN_FEATURE_DIM)),
        target_tensor=torch.tensor([-0.5], dtype=torch.float32),
    )

    batch = collate_morpion_graph_supervised_samples((sample_a, sample_b))

    assert batch.get_input_layer().shape == (2, 4, MORPION_GRAPH_TOKEN_FEATURE_DIM)
    assert batch.get_target_value().shape == (2, 1)
    assert torch.all(batch.input_tensor[0, 2:, :] == 0.0)
    assert torch.all(batch.input_tensor[0, 2:, -1] == 0.0)
    torch.testing.assert_close(
        batch.target_tensor,
        torch.tensor([[0.25], [-0.5]], dtype=torch.float32),
    )


def test_graph_direction_names_match_feature_extractor_order() -> None:
    """Graph direction labels should match the raw Morpion direction indices."""
    assert MORPION_GRAPH_DIRECTIONS == (
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


def test_graph_regressor_args_reject_feature_dim_mismatch() -> None:
    """Graph models should match the v1 graph-token feature width exactly."""
    with pytest.raises(ValueError, match="graph_input_feature_dim"):
        MorpionRegressorArgs(
            model_kind=MORPION_GRAPH_MODEL_KIND,
            graph_input_feature_dim=MORPION_GRAPH_TOKEN_FEATURE_DIM + 1,
        )


def test_graph_regressor_args_reject_too_small_token_cap() -> None:
    """Graph models need room for VALUE and GLOBAL tokens."""
    with pytest.raises(ValueError, match="graph_max_tokens"):
        MorpionRegressorArgs(
            model_kind=MORPION_GRAPH_MODEL_KIND,
            graph_max_tokens=1,
        )


def test_graph_regressor_args_reject_incompatible_attention_width() -> None:
    """Transformer width should be divisible by the configured head count."""
    with pytest.raises(ValueError, match="divisible"):
        MorpionRegressorArgs(
            model_kind=MORPION_GRAPH_MODEL_KIND,
            graph_d_model=10,
            graph_n_head=4,
        )


def test_graph_model_builds_and_runs_forward() -> None:
    """A graph_transformer model should accept T x F and B x T x F tensors."""
    model = build_morpion_regressor(_small_graph_args())
    converter = MorpionGraphTokenConverter(max_tokens=128)
    input_tensor = converter.state_to_tensor(_make_standard_state())

    single_output = model(input_tensor)
    batch_output = model(input_tensor.unsqueeze(0))

    assert single_output.shape == (1,)
    assert batch_output.shape == (1, 1)
    assert torch.isfinite(single_output).all()
    assert torch.isfinite(batch_output).all()


def test_graph_bundle_roundtrip_runs_forward(tmp_path: Path) -> None:
    """Graph model bundles should persist graph args and load for inference."""
    args = _small_graph_args()
    model = build_morpion_regressor(args)
    bundle_dir = tmp_path / "graph_bundle"

    save_morpion_model_bundle(model, bundle_dir, model_args=args)
    loaded_model, loaded_args, manifest = load_morpion_model_bundle(bundle_dir)
    manifest_text = (bundle_dir / MORPION_MANIFEST_FILE_NAME).read_text(
        encoding="utf-8"
    )

    assert loaded_args.model_kind == MORPION_GRAPH_MODEL_KIND
    assert manifest.input_representation == MORPION_GRAPH_INPUT_REPRESENTATION
    assert manifest.input_dim == MORPION_GRAPH_TOKEN_FEATURE_DIM
    assert "graph_token_feature_names" in manifest_text

    input_tensor = MorpionGraphTokenConverter(max_tokens=128).state_to_tensor(
        _make_standard_state()
    )
    output = loaded_model(input_tensor)
    assert output.shape == (1,)
    assert torch.isfinite(output).all()


def test_graph_training_smoke_saves_bundle_and_metrics(tmp_path: Path) -> None:
    """The training helper should run the graph-token path end to end."""
    dataset_file = _build_rows_file(tmp_path)
    output_dir = tmp_path / "trained_graph_bundle"

    _model, metrics = train_morpion_regressor(
        MorpionTrainingArgs(
            dataset_file=dataset_file,
            output_dir=output_dir,
            batch_size=2,
            num_epochs=1,
            learning_rate=1e-3,
            shuffle=False,
            model_kind=MORPION_GRAPH_MODEL_KIND,
            graph_max_tokens=128,
            graph_d_model=16,
            graph_n_head=4,
            graph_n_layer=1,
            graph_dim_feedforward=32,
        )
    )

    assert output_dir.is_dir()
    assert math.isfinite(cast("float", metrics["final_loss"]))
    _loaded_model, loaded_args, manifest = load_morpion_model_bundle(output_dir)
    assert loaded_args.model_kind == MORPION_GRAPH_MODEL_KIND
    assert manifest.input_representation == MORPION_GRAPH_INPUT_REPRESENTATION


def test_graph_bundle_loads_as_anemone_evaluator(tmp_path: Path) -> None:
    """A graph bundle should load behind the Anemone evaluator protocol."""
    args = _small_graph_args()
    model = build_morpion_regressor(args)
    bundle_dir = tmp_path / "graph_bundle"
    save_morpion_model_bundle(model, bundle_dir, model_args=args)

    evaluator = load_morpion_evaluator_from_model_bundle(bundle_dir)
    value = evaluator.evaluate(_make_standard_state())

    assert value.certainty is Certainty.ESTIMATE
    assert math.isfinite(value.score)

"""Tests for the Morpion bootstrap toy-tree sanity laboratory."""
# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHIPIRON_PACKAGE_ROOT = _REPO_ROOT / "src" / "chipiron"
_ENVIRONMENTS_PACKAGE_ROOT = _CHIPIRON_PACKAGE_ROOT / "environments"
_MORPION_PACKAGE_ROOT = _ENVIRONMENTS_PACKAGE_ROOT / "morpion"
_BOOTSTRAP_PACKAGE_ROOT = _MORPION_PACKAGE_ROOT / "bootstrap"

if "chipiron" not in sys.modules:
    _chipiron_stub = ModuleType("chipiron")
    _chipiron_stub.__path__ = [str(_CHIPIRON_PACKAGE_ROOT)]
    sys.modules["chipiron"] = _chipiron_stub

if "chipiron.environments" not in sys.modules:
    _environments_stub = ModuleType("chipiron.environments")
    _environments_stub.__path__ = [str(_ENVIRONMENTS_PACKAGE_ROOT)]
    sys.modules["chipiron.environments"] = _environments_stub

if "chipiron.environments.morpion" not in sys.modules:
    _morpion_stub = ModuleType("chipiron.environments.morpion")
    _morpion_stub.__path__ = [str(_MORPION_PACKAGE_ROOT)]
    sys.modules["chipiron.environments.morpion"] = _morpion_stub

if "chipiron.environments.morpion.bootstrap" not in sys.modules:
    _bootstrap_stub = ModuleType("chipiron.environments.morpion.bootstrap")
    _bootstrap_stub.__path__ = [str(_BOOTSTRAP_PACKAGE_ROOT)]
    sys.modules["chipiron.environments.morpion.bootstrap"] = _bootstrap_stub

from chipiron.environments.morpion.bootstrap.evaluator_toy_tree_sanity import (
    ToyRunConfig,
    ToyTrainingRow,
    ToyValueNet,
    build_training_rows,
    built_in_toy_tree,
    compute_backed_up_values,
    make_toy_model,
    predict_all_nodes,
    run_toy_tree_sanity,
    train_weighted_regressor,
)


def test_case_a_max_backup_root_value_is_terminal_when_frontier_zero() -> None:
    """Case A should anchor the root to the exact terminal child."""
    tree = built_in_toy_tree("A_stable_terminal_wins")

    backed_up = compute_backed_up_values(tree, {2: 0.0}, "max")

    assert backed_up[0].value == 1.0
    assert backed_up[0].argmax_child_id == 1


def test_case_b_max_backup_root_value_is_optimistic_frontier() -> None:
    """Case B should expose hard-max poisoning from one optimistic frontier leaf."""
    tree = built_in_toy_tree("B_optimistic_frontier_poison")

    backed_up = compute_backed_up_values(tree, {2: 10.0}, "max")

    assert backed_up[0].value == 10.0
    assert backed_up[0].argmax_child_id == 2
    assert backed_up[0].support_frontier_count == 1


def test_exact_only_excludes_frontier_rows() -> None:
    """The exact-only dataset mode should keep only ground-truth anchors."""
    tree = built_in_toy_tree("B_optimistic_frontier_poison")
    predictions = {node_id: 0.0 for node_id in tree.nodes}
    backed_up = compute_backed_up_values(tree, {2: 10.0}, "max")

    rows = build_training_rows(
        tree=tree,
        backed_up_values=backed_up,
        train_targets="exact_only",
        prediction_before=predictions,
    )

    assert tuple(row.node_id for row in rows) == (1,)
    assert all(row.source == "ground_truth_exact_or_terminal" for row in rows)


def test_clipped_max_clips_pure_frontier_optimism() -> None:
    """The clipped max backup should cap pure frontier-supported branches."""
    tree = built_in_toy_tree("B_optimistic_frontier_poison")

    backed_up = compute_backed_up_values(
        tree,
        {2: 10.0},
        "clipped_max",
        frontier_clip=2.0,
    )

    assert backed_up[0].value == 2.0
    assert backed_up[0].argmax_child_id == 2


def test_weighted_loss_does_not_crash() -> None:
    """The local weighted trainer should accept simple rows and return a loss."""
    model = ToyValueNet(feature_dim=3, hidden_dim=4, num_layers=2, init_mode="zero")
    rows = (
        ToyTrainingRow(
            node_id=1,
            feature=(0.0, 1.0, 0.0),
            target=1.0,
            source="ground_truth_exact_or_terminal",
            weight=1.0,
        ),
        ToyTrainingRow(
            node_id=2,
            feature=(0.0, 0.0, 1.0),
            target=10.0,
            source="frontier_prediction",
            weight=0.1,
        ),
    )

    loss = train_weighted_regressor(
        model,
        rows,
        learning_rate=1e-3,
        train_epochs=2,
        batch_size=None,
    )

    assert loss >= 0.0


def test_output_dir_writes_plotting_artifacts(tmp_path: Path) -> None:
    """A run with output_dir should write config, metrics, and node history files."""
    run_toy_tree_sanity(
        ToyRunConfig(
            num_iterations=1,
            train_epochs=1,
            print_every=0,
            output_dir=tmp_path,
            save_csv=tmp_path / "metrics_copy.csv",
            save_json=tmp_path / "summary.json",
        )
    )

    assert (tmp_path / "toy_tree_config.json").is_file()
    assert (tmp_path / "iteration_metrics.csv").is_file()
    assert (tmp_path / "node_history.csv").is_file()
    assert (tmp_path / "metrics_copy.csv").is_file()
    assert (tmp_path / "summary.json").is_file()


def test_case_f_tree_features_and_topology() -> None:
    """Case F should expose the intended compositional linear features."""
    tree = built_in_toy_tree("F_linear_compositional_vicious_circle")

    assert tree.nodes[0].feature == (0.0, 1.0, 0.0)
    assert tree.nodes[0].child_ids == (1, 2)
    assert tree.nodes[1].feature == (1.0, 0.0, 0.0)
    assert tree.nodes[1].is_terminal
    assert tree.nodes[1].exact_value == 1.0
    assert tree.nodes[2].feature == (0.0, 0.0, 1.0)
    assert tree.nodes[2].child_ids == (3,)
    assert tree.nodes[3].feature == (0.0, 1.0, 1.0)


def test_case_f_initial_hard_max_with_a_override_is_ten() -> None:
    """The initial A override should back up through the path and root."""
    tree = built_in_toy_tree("F_linear_compositional_vicious_circle")

    backed_up = compute_backed_up_values(tree, {3: 10.0}, "max")

    assert backed_up[3].value == 10.0
    assert backed_up[2].value == 10.0
    assert backed_up[0].value == 10.0


def test_linear_no_bias_initial_predictions_are_zero() -> None:
    """The transparent linear model should start with zero predictions."""
    tree = built_in_toy_tree("F_linear_compositional_vicious_circle")
    model = make_toy_model(
        ToyRunConfig(model_kind="linear_no_bias"),
        tree.feature_dim,
    )

    predictions = predict_all_nodes(model, tree)

    assert all(prediction == 0.0 for prediction in predictions.values())


def test_linear_no_bias_run_writes_linear_weights(tmp_path: Path) -> None:
    """A short linear run should expose weights in metrics and CSV output."""
    result = run_toy_tree_sanity(
        ToyRunConfig(
            case="F_linear_compositional_vicious_circle",
            model_kind="linear_no_bias",
            num_iterations=1,
            train_epochs=1,
            print_every=0,
            output_dir=tmp_path,
        )
    )

    metric = result.metrics[0]
    assert metric.linear_w0 is not None
    assert metric.linear_w1 is not None
    assert metric.linear_w2 is not None
    assert metric.linear_a_value_from_weights is not None
    assert "linear_w0" in (tmp_path / "iteration_metrics.csv").read_text(
        encoding="utf-8"
    )

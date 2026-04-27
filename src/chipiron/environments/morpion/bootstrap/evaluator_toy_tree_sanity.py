"""Tiny controlled tree laboratory for bootstrap value-learning dynamics.

Linear vicious-circle example:

```
python -m chipiron.environments.morpion.bootstrap.evaluator_toy_tree_sanity \
  --case F_linear_compositional_vicious_circle \
  --model-kind linear_no_bias \
  --backup-operator max \
  --train-targets all \
  --frontier-weight 1.0 \
  --child-backup-weight 1.0 \
  --exact-weight 1.0 \
  --num-iterations 20 \
  --train-epochs 500 \
  --learning-rate 0.01 \
  --seed 0 \
  --output-dir /tmp/morpion_toy_F_linear_vicious
```
"""
# ruff: noqa: TRY003

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass, replace
from itertools import pairwise
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn

ToyCaseName = Literal[
    "A_stable_terminal_wins",
    "B_optimistic_frontier_poison",
    "C_many_frontier_outliers",
    "D_deep_terminal_vs_frontier",
    "E_two_level_mixed",
    "F_linear_compositional_vicious_circle",
]
BackupOperator = Literal["max", "softmax", "clipped_max"]
TrainTargets = Literal["exact_only", "exact_plus_child", "all"]
InitMode = Literal["zero", "random", "optimistic_feature"]
ModelKind = Literal["mlp", "linear_no_bias"]
FamilyTargetPolicy = Literal[
    "none",
    "pv_mean_prediction",
    "pv_min_prediction",
    "pv_quantile_prediction",
    "pv_blend_mean_prediction",
    "pv_blend_min_prediction",
    "pv_blend_quantile_prediction",
]
TargetSource = Literal[
    "ground_truth_exact_or_terminal",
    "child_backup",
    "frontier_prediction",
]


@dataclass(frozen=True, slots=True)
class ToyNode:
    """One node in a tiny synthetic value-backup tree."""

    node_id: int
    parent_id: int | None
    child_ids: tuple[int, ...]
    feature: tuple[float, ...]
    is_terminal: bool = False
    is_exact: bool = False
    exact_value: float | None = None


@dataclass(slots=True)
class ToyTree:
    """Small directed tree with parent/child invariants checked eagerly."""

    nodes: dict[int, ToyNode]
    root_id: int

    def __post_init__(self) -> None:
        """Validate topology and value anchors."""
        if self.root_id not in self.nodes:
            raise ValueError(f"Toy tree root {self.root_id} does not exist.")
        if len(self.nodes) >= 50:
            raise ValueError("Toy tree diagnostics are intended for fewer than 50 nodes.")

        root = self.nodes[self.root_id]
        if root.parent_id is not None:
            raise ValueError("Toy tree root must not have a parent.")

        feature_dims = {len(node.feature) for node in self.nodes.values()}
        if len(feature_dims) != 1:
            raise ValueError("All toy nodes must share the same feature dimension.")

        for node in self.nodes.values():
            if (node.is_terminal or node.is_exact) and node.exact_value is None:
                raise ValueError(
                    f"Exact or terminal toy node {node.node_id} needs exact_value."
                )
            for child_id in node.child_ids:
                if child_id not in self.nodes:
                    raise ValueError(
                        f"Toy node {node.node_id} references missing child {child_id}."
                    )
                child = self.nodes[child_id]
                if child.parent_id != node.node_id:
                    raise ValueError(
                        "Toy parent/child mismatch: "
                        f"{node.node_id} -> {child_id}, child parent={child.parent_id}."
                    )
            if node.parent_id is not None:
                parent = self.nodes.get(node.parent_id)
                if parent is None or node.node_id not in parent.child_ids:
                    raise ValueError(
                        f"Toy node {node.node_id} has inconsistent parent link."
                    )

        visited: set[int] = set()
        active: set[int] = set()

        def visit(node_id: int) -> None:
            if node_id in active:
                raise ValueError("Toy tree contains a cycle.")
            if node_id in visited:
                return
            active.add(node_id)
            for child_id in self.nodes[node_id].child_ids:
                visit(child_id)
            active.remove(node_id)
            visited.add(node_id)

        visit(self.root_id)
        if visited != set(self.nodes):
            missing = sorted(set(self.nodes) - visited)
            raise ValueError(f"Toy tree has unreachable nodes: {missing}.")

    @property
    def feature_dim(self) -> int:
        """Return the synthetic feature dimension."""
        return len(self.nodes[self.root_id].feature)

    def node_ids_by_depth(self, *, reverse: bool = False) -> tuple[int, ...]:
        """Return node IDs sorted by tree depth."""
        depths = self.depths()
        return tuple(sorted(self.nodes, key=lambda node_id: depths[node_id], reverse=reverse))

    def depths(self) -> dict[int, int]:
        """Return depth for every node."""
        depths = {self.root_id: 0}
        stack = [self.root_id]
        while stack:
            node_id = stack.pop()
            for child_id in self.nodes[node_id].child_ids:
                depths[child_id] = depths[node_id] + 1
                stack.append(child_id)
        return depths


@dataclass(frozen=True, slots=True)
class ToyBackedUpValue:
    """Backed-up value and provenance diagnostics for one toy node."""

    value: float
    source: TargetSource
    support_exact_count: int
    support_frontier_count: int
    distance_to_exact: int | None
    argmax_child_id: int | None = None
    argmax_child_source: str | None = None


@dataclass(frozen=True, slots=True)
class ToyTrainingRow:
    """One weighted supervised row generated from toy backed-up values."""

    node_id: int
    feature: tuple[float, ...]
    target: float
    source: TargetSource
    weight: float


@dataclass(frozen=True, slots=True)
class ToyIterationMetric:
    """Compact per-iteration diagnostic metrics."""

    iteration: int
    root_target: float
    root_prediction_before: float
    root_prediction_after: float
    max_frontier_prediction_before: float
    max_frontier_prediction_after: float
    mean_abs_prediction_drift_all: float
    max_abs_prediction_drift_all: float
    mean_abs_target_drift_all: float
    max_abs_target_drift_all: float
    argmax_child_id_at_root: int | None
    argmax_child_source_at_root: str | None
    num_rows: int
    num_exact_rows: int
    num_child_backup_rows: int
    num_frontier_rows: int
    weighted_train_loss_final: float
    unweighted_mae_all_rows: float
    mae_exact: float | None
    mae_child_backup: float | None
    mae_frontier: float | None
    max_abs_prediction: float
    diverged: bool
    linear_w0: float | None
    linear_w1: float | None
    linear_w2: float | None
    linear_a_value_from_weights: float | None


@dataclass(frozen=True, slots=True)
class ToyNodeHistoryRow:
    """Per-node row for later plotting and diagnosis."""

    iteration: int
    node_id: int
    prediction_before: float
    prediction_after: float
    target: float
    source: TargetSource
    weight: float
    support_exact_count: int
    support_frontier_count: int
    distance_to_exact: int | None
    argmax_child_id: int | None
    argmax_child_source: str | None


@dataclass(frozen=True, slots=True)
class ToyRunConfig:
    """Configuration for one toy bootstrap diagnostic run."""

    case: ToyCaseName = "B_optimistic_frontier_poison"
    hidden_dim: int = 16
    num_layers: int = 2
    learning_rate: float = 1e-3
    train_epochs: int = 200
    batch_size: int | None = None
    seed: int = 0
    init_mode: InitMode = "zero"
    model_kind: ModelKind = "mlp"
    frontier_initial_overrides: dict[int, float] | None = None
    backup_operator: BackupOperator = "max"
    backup_temperature: float = 1.0
    frontier_clip: float = 2.0
    family_target_policy: FamilyTargetPolicy = "none"
    family_quantile: float = 0.25
    family_prediction_blend: float = 0.5
    train_targets: TrainTargets = "all"
    frontier_weight: float = 0.1
    child_backup_weight: float = 0.5
    exact_weight: float = 1.0
    include_root: bool = True
    include_frontier_targets: bool = True
    use_direct_targets: bool = False
    num_iterations: int = 20
    eval_before_train: bool = True
    divergence_threshold: float = 100.0
    print_every: int = 1
    output_dir: Path | None = None
    save_csv: Path | None = None
    save_json: Path | None = None


@dataclass(frozen=True, slots=True)
class ToyRunResult:
    """Structured result from one toy bootstrap diagnostic run."""

    config: ToyRunConfig
    tree: ToyTree
    metrics: tuple[ToyIterationMetric, ...]
    node_history: tuple[ToyNodeHistoryRow, ...]
    summary: dict[str, object]


class ToyValueNet(nn.Module):
    """Small MLP regressor for synthetic toy-node features."""

    def __init__(
        self,
        *,
        feature_dim: int,
        hidden_dim: int = 16,
        num_layers: int = 2,
        init_mode: InitMode = "zero",
    ) -> None:
        """Initialize the toy network."""
        super().__init__()
        layers: list[nn.Module] = []
        if num_layers <= 1:
            layers.append(nn.Linear(feature_dim, 1))
        else:
            layers.append(nn.Linear(feature_dim, hidden_dim))
            layers.append(nn.Tanh())
            for _ in range(max(num_layers - 2, 0)):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.Tanh())
            layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)
        if init_mode == "zero":
            self._zero_final_layer()
        elif init_mode == "optimistic_feature":
            self._optimistic_feature_init(feature_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict scalar values for a batch of feature vectors."""
        return self.network(features).squeeze(-1)

    def _zero_final_layer(self) -> None:
        """Set the final linear layer to predict zero initially."""
        final = _final_linear_layer(self.network)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def _optimistic_feature_init(self, feature_dim: int) -> None:
        """Make high final feature values predict positively at initialization."""
        if len(self.network) == 1:
            final = _final_linear_layer(self.network)
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
            final.weight.data[0, feature_dim - 1] = 10.0
            return
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)
        first = next(module for module in self.network if isinstance(module, nn.Linear))
        first.weight.data.zero_()
        first.weight.data[0, feature_dim - 1] = 1.0
        final = _final_linear_layer(self.network)
        final.weight.data.zero_()
        final.bias.data.zero_()
        final.weight.data[0, 0] = 10.0


class ToyLinearNoBiasNet(nn.Module):
    """Transparent linear evaluator ``V(x) = w0*x0 + w1*x1 + w2*x2``."""

    def __init__(self, feature_dim: int) -> None:
        """Initialize a zero-prediction linear evaluator with no bias."""
        super().__init__()
        self.linear = nn.Linear(feature_dim, 1, bias=False)
        nn.init.zeros_(self.linear.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict scalar values for a batch of feature vectors."""
        return self.linear(features).squeeze(-1)


def make_toy_model(config: ToyRunConfig, feature_dim: int) -> nn.Module:
    """Build the configured toy evaluator."""
    if config.model_kind == "linear_no_bias":
        return ToyLinearNoBiasNet(feature_dim)
    return ToyValueNet(
        feature_dim=feature_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        init_mode=config.init_mode,
    )


def built_in_toy_tree(case: ToyCaseName) -> ToyTree:
    """Build one of the controlled toy tree cases."""
    if case == "A_stable_terminal_wins":
        return _make_tree(
            (
                ToyNode(0, None, (1, 2), (1.0, 0.0, 0.0)),
                ToyNode(1, 0, (), (0.0, 1.0, 0.0), is_terminal=True, exact_value=1.0),
                ToyNode(2, 0, (), (0.0, 0.0, 0.0)),
            )
        )
    if case == "B_optimistic_frontier_poison":
        return _make_tree(
            (
                ToyNode(0, None, (1, 2), (1.0, 0.0, 0.0)),
                ToyNode(1, 0, (), (0.0, 1.0, 0.0), is_terminal=True, exact_value=1.0),
                ToyNode(2, 0, (), (0.0, 0.0, 1.0)),
            )
        )
    if case == "C_many_frontier_outliers":
        frontier_nodes = tuple(
            ToyNode(node_id, 0, (), (0.0, float(node_id) / 20.0, 1.0))
            for node_id in range(2, 18)
        )
        return _make_tree(
            (
                ToyNode(0, None, tuple(range(1, 18)), (1.0, 0.0, 0.0)),
                ToyNode(1, 0, (), (0.0, 1.0, 0.0), is_terminal=True, exact_value=1.0),
                *frontier_nodes,
            )
        )
    if case == "D_deep_terminal_vs_frontier":
        return _make_tree(
            (
                ToyNode(0, None, (1, 4), (1.0, 0.0, 0.0)),
                ToyNode(1, 0, (2,), (0.0, 0.25, 0.0)),
                ToyNode(2, 1, (3,), (0.0, 0.5, 0.0)),
                ToyNode(3, 2, (), (0.0, 1.0, 0.0), is_terminal=True, exact_value=1.0),
                ToyNode(4, 0, (), (0.0, 0.1, 1.0)),
            )
        )
    if case == "E_two_level_mixed":
        return _make_tree(
            (
                ToyNode(0, None, (1, 2, 3), (1.0, 0.0, 0.0)),
                ToyNode(1, 0, (4, 5), (0.0, 0.2, 0.0)),
                ToyNode(2, 0, (6, 7), (0.0, 0.4, 0.0)),
                ToyNode(3, 0, (), (0.0, 0.1, 1.0)),
                ToyNode(4, 1, (), (0.0, 1.0, 0.0), is_terminal=True, exact_value=1.0),
                ToyNode(5, 1, (), (0.0, 0.0, 1.0)),
                ToyNode(6, 2, (), (0.0, 1.0, 0.0), is_terminal=True, exact_value=-1.0),
                ToyNode(7, 2, (), (0.0, 1.0, 0.0), is_terminal=True, exact_value=0.5),
            )
        )
    if case == "F_linear_compositional_vicious_circle":
        return _make_tree(
            (
                ToyNode(0, None, (1, 2), (0.0, 1.0, 0.0)),
                ToyNode(1, 0, (), (1.0, 0.0, 0.0), is_terminal=True, exact_value=1.0),
                ToyNode(2, 0, (3,), (0.0, 0.0, 1.0)),
                ToyNode(3, 2, (), (0.0, 1.0, 1.0)),
            )
        )
    raise ValueError(f"Unknown toy tree case: {case!r}.")


def compute_backed_up_values(
    tree: ToyTree,
    evaluator_predictions: dict[int, float],
    backup_operator: BackupOperator = "max",
    frontier_policy: str = "prediction",
    *,
    backup_temperature: float = 1.0,
    frontier_clip: float = 2.0,
) -> dict[int, ToyBackedUpValue]:
    """Evaluate frontier leaves and propagate values backward through the tree."""
    if frontier_policy != "prediction":
        raise ValueError(f"Unsupported toy frontier policy: {frontier_policy!r}.")
    if backup_operator == "softmax" and backup_temperature <= 0.0:
        raise ValueError("Softmax backup temperature must be positive.")

    backed_up: dict[int, ToyBackedUpValue] = {}
    for node_id in tree.node_ids_by_depth(reverse=True):
        node = tree.nodes[node_id]
        if node.is_terminal or node.is_exact:
            if node.exact_value is None:
                raise ValueError(f"Exact toy node {node_id} is missing exact_value.")
            backed_up[node_id] = ToyBackedUpValue(
                value=node.exact_value,
                source="ground_truth_exact_or_terminal",
                support_exact_count=1,
                support_frontier_count=0,
                distance_to_exact=0,
            )
            continue

        if not node.child_ids:
            if node_id not in evaluator_predictions:
                raise ValueError(f"Missing evaluator prediction for frontier node {node_id}.")
            backed_up[node_id] = ToyBackedUpValue(
                value=evaluator_predictions[node_id],
                source="frontier_prediction",
                support_exact_count=0,
                support_frontier_count=1,
                distance_to_exact=None,
            )
            continue

        children = tuple(backed_up[child_id] for child_id in node.child_ids)
        backup_values = tuple(
            _backup_child_value(
                child,
                backup_operator=backup_operator,
                frontier_clip=frontier_clip,
            )
            for child in children
        )
        argmax_offset = max(range(len(backup_values)), key=backup_values.__getitem__)
        argmax_child = children[argmax_offset]
        argmax_child_id = node.child_ids[argmax_offset]

        if backup_operator == "softmax":
            value = _temperature_logsumexp(backup_values, temperature=backup_temperature)
            support_exact_count = sum(child.support_exact_count for child in children)
            support_frontier_count = sum(child.support_frontier_count for child in children)
            exact_distances = [
                child.distance_to_exact + 1
                for child in children
                if child.distance_to_exact is not None
            ]
            distance_to_exact = min(exact_distances) if exact_distances else None
        else:
            value = backup_values[argmax_offset]
            support_exact_count = argmax_child.support_exact_count
            support_frontier_count = argmax_child.support_frontier_count
            distance_to_exact = (
                None
                if argmax_child.distance_to_exact is None
                else argmax_child.distance_to_exact + 1
            )

        backed_up[node_id] = ToyBackedUpValue(
            value=value,
            source="child_backup",
            support_exact_count=support_exact_count,
            support_frontier_count=support_frontier_count,
            distance_to_exact=distance_to_exact,
            argmax_child_id=argmax_child_id,
            argmax_child_source=argmax_child.source,
        )
    return backed_up


def build_training_rows(
    *,
    tree: ToyTree,
    backed_up_values: dict[int, ToyBackedUpValue],
    train_targets: TrainTargets,
    prediction_before: dict[int, float],
    frontier_weight: float = 0.1,
    child_backup_weight: float = 0.5,
    exact_weight: float = 1.0,
    include_root: bool = True,
    include_frontier_targets: bool = True,
    use_direct_targets: bool = False,
    effective_targets: dict[int, float] | None = None,
) -> tuple[ToyTrainingRow, ...]:
    """Construct weighted supervised rows from backed-up toy targets."""
    rows: list[ToyTrainingRow] = []
    for node_id in tree.node_ids_by_depth():
        if node_id == tree.root_id and not include_root:
            continue
        backed = backed_up_values[node_id]
        if train_targets == "exact_only" and backed.source != "ground_truth_exact_or_terminal":
            continue
        if train_targets == "exact_plus_child" and backed.source == "frontier_prediction":
            continue
        if (
            train_targets == "all"
            and backed.source == "frontier_prediction"
            and not include_frontier_targets
        ):
            continue
        if train_targets not in {"exact_only", "exact_plus_child", "all"}:
            raise ValueError(f"Unknown toy train target selection: {train_targets!r}.")

        target = backed.value if effective_targets is None else effective_targets[node_id]
        if use_direct_targets and backed.source != "ground_truth_exact_or_terminal":
            target = prediction_before[node_id]
        rows.append(
            ToyTrainingRow(
                node_id=node_id,
                feature=tree.nodes[node_id].feature,
                target=target,
                source=backed.source,
                weight=_source_weight(
                    backed.source,
                    exact_weight=exact_weight,
                    child_backup_weight=child_backup_weight,
                    frontier_weight=frontier_weight,
                ),
            )
        )
    return tuple(rows)


def principal_variation_families(
    tree: ToyTree,
    backed_up_values: dict[int, ToyBackedUpValue],
) -> dict[int, tuple[int, ...]]:
    """Group nodes by the leaf reached by repeatedly following argmax children."""
    families: dict[int, list[int]] = {}
    for node_id in tree.node_ids_by_depth():
        representative = _principal_variation_representative(
            node_id,
            backed_up_values,
        )
        families.setdefault(representative, []).append(node_id)
    return {
        representative: tuple(node_ids)
        for representative, node_ids in families.items()
    }


def family_adjusted_targets(
    *,
    tree: ToyTree,
    backed_up_values: dict[int, ToyBackedUpValue],
    prediction_before: dict[int, float],
    family_target_policy: FamilyTargetPolicy,
    family_quantile: float = 0.25,
    family_prediction_blend: float = 0.5,
) -> dict[int, float]:
    """Return effective supervised targets after optional PV-family smoothing."""
    if family_target_policy == "none":
        return {
            node_id: backed_up.value for node_id, backed_up in backed_up_values.items()
        }
    if not 0.0 <= family_quantile <= 1.0:
        raise ValueError("Family quantile must be between 0 and 1.")
    if not 0.0 <= family_prediction_blend <= 1.0:
        raise ValueError("Family prediction blend must be between 0 and 1.")

    targets: dict[int, float] = {}
    for family_node_ids in principal_variation_families(tree, backed_up_values).values():
        family_values = [
            _family_policy_value(
                node_id=node_id,
                tree=tree,
                backed_up_values=backed_up_values,
                prediction_before=prediction_before,
                family_target_policy=family_target_policy,
            )
            for node_id in family_node_ids
        ]
        family_target = _family_aggregate(
            family_values,
            family_target_policy=family_target_policy,
            family_quantile=family_quantile,
        )
        for node_id in family_node_ids:
            if family_target_policy.startswith("pv_blend_"):
                targets[node_id] = (
                    (1.0 - family_prediction_blend) * backed_up_values[node_id].value
                    + family_prediction_blend * family_target
                )
            else:
                targets[node_id] = family_target
    return targets


def train_weighted_regressor(
    model: nn.Module,
    rows: tuple[ToyTrainingRow, ...],
    *,
    learning_rate: float = 1e-3,
    train_epochs: int = 200,
    batch_size: int | None = None,
) -> float:
    """Train with normalized weighted MSE and return final epoch loss."""
    if not rows:
        return math.nan

    features = torch.tensor([row.feature for row in rows], dtype=torch.float32)
    targets = torch.tensor([row.target for row in rows], dtype=torch.float32)
    weights = torch.tensor([row.weight for row in rows], dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    batch_size = len(rows) if batch_size is None else max(1, min(batch_size, len(rows)))
    final_loss = math.nan

    for _ in range(train_epochs):
        permutation = torch.randperm(len(rows))
        epoch_loss_num = 0.0
        epoch_weight_sum = 0.0
        for start in range(0, len(rows), batch_size):
            indices = permutation[start : start + batch_size]
            prediction = model(features[indices])
            batch_targets = targets[indices]
            batch_weights = weights[indices]
            loss = torch.sum(batch_weights * (prediction - batch_targets).square())
            loss = loss / torch.clamp(torch.sum(batch_weights), min=1e-12)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss_num += float(
                torch.sum(
                    batch_weights.detach()
                    * (prediction.detach() - batch_targets).square(),
                )
            )
            epoch_weight_sum += float(torch.sum(batch_weights.detach()))
        final_loss = epoch_loss_num / max(epoch_weight_sum, 1e-12)
    return final_loss


def run_toy_tree_sanity(config: ToyRunConfig) -> ToyRunResult:
    """Run repeated fitted backup and evaluator updates on a toy tree."""
    set_deterministic_seeds(config.seed)
    tree = built_in_toy_tree(config.case)
    model = make_toy_model(config, tree.feature_dim)
    overrides = _effective_frontier_initial_overrides(config)
    previous_targets: dict[int, float] | None = None
    metrics: list[ToyIterationMetric] = []
    node_history: list[ToyNodeHistoryRow] = []

    if config.output_dir is not None:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        _write_tree_config(config.output_dir / "toy_tree_config.json", tree, config)

    if config.print_every > 0:
        _print_header()
    for iteration in range(config.num_iterations):
        prediction_before = predict_all_nodes(model, tree)
        backup_predictions = dict(prediction_before)
        if iteration == 0:
            backup_predictions.update(overrides)

        backed_up = compute_backed_up_values(
            tree,
            backup_predictions,
            config.backup_operator,
            backup_temperature=config.backup_temperature,
            frontier_clip=config.frontier_clip,
        )
        effective_targets = family_adjusted_targets(
            tree=tree,
            backed_up_values=backed_up,
            prediction_before=prediction_before,
            family_target_policy=config.family_target_policy,
            family_quantile=config.family_quantile,
            family_prediction_blend=config.family_prediction_blend,
        )
        rows = build_training_rows(
            tree=tree,
            backed_up_values=backed_up,
            train_targets=config.train_targets,
            prediction_before=prediction_before,
            frontier_weight=config.frontier_weight,
            child_backup_weight=config.child_backup_weight,
            exact_weight=config.exact_weight,
            include_root=config.include_root,
            include_frontier_targets=config.include_frontier_targets,
            use_direct_targets=config.use_direct_targets,
            effective_targets=effective_targets,
        )
        final_loss = train_weighted_regressor(
            model,
            rows,
            learning_rate=config.learning_rate,
            train_epochs=config.train_epochs,
            batch_size=config.batch_size,
        )
        prediction_after = predict_all_nodes(model, tree)
        weights_after = _linear_weights(model)
        metric = _iteration_metric(
            iteration=iteration,
            tree=tree,
            backed_up=backed_up,
            rows=rows,
            prediction_before=prediction_before,
            prediction_after=prediction_after,
            previous_targets=previous_targets,
            effective_targets=effective_targets,
            final_loss=final_loss,
            divergence_threshold=config.divergence_threshold,
            weights_after=weights_after,
        )
        metrics.append(metric)
        node_history.extend(
            _node_history_rows(
                iteration=iteration,
                tree=tree,
                backed_up=backed_up,
                rows=rows,
                prediction_before=prediction_before,
                prediction_after=prediction_after,
                effective_targets=effective_targets,
            )
        )
        if config.print_every > 0 and iteration % config.print_every == 0:
            _print_metric(metric)
        previous_targets = effective_targets

    result = ToyRunResult(
        config=config,
        tree=tree,
        metrics=tuple(metrics),
        node_history=tuple(node_history),
        summary=_summary(config=config, tree=tree, metrics=tuple(metrics)),
    )
    if config.print_every > 0:
        _print_summary(result.summary)
    if config.output_dir is not None:
        _write_csv(config.output_dir / "iteration_metrics.csv", result.metrics)
        _write_csv(config.output_dir / "node_history.csv", result.node_history)
    if config.save_csv is not None:
        _write_csv(config.save_csv, result.metrics)
    if config.save_json is not None:
        config.save_json.parent.mkdir(parents=True, exist_ok=True)
        config.save_json.write_text(
            json.dumps(result.summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return result


def set_deterministic_seeds(seed: int) -> None:
    """Seed Python, NumPy, and Torch for deterministic CPU diagnostics."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def predict_all_nodes(model: nn.Module, tree: ToyTree) -> dict[int, float]:
    """Run model predictions for every toy node."""
    model.eval()
    node_ids = tree.node_ids_by_depth()
    features = torch.tensor(
        [tree.nodes[node_id].feature for node_id in node_ids],
        dtype=torch.float32,
    )
    with torch.no_grad():
        raw = model(features).detach().cpu().tolist()
    return {node_id: float(value) for node_id, value in zip(node_ids, raw, strict=True)}


def compare_strategies(base_config: ToyRunConfig) -> tuple[dict[str, object], ...]:
    """Run a small fixed mitigation matrix and print a comparison table."""
    strategies = (
        (
            "max_all_fw1",
            replace(
                base_config,
                backup_operator="max",
                train_targets="all",
                frontier_weight=1.0,
                output_dir=None,
                save_csv=None,
                save_json=None,
                print_every=0,
            ),
        ),
        (
            "max_all_fw0.1",
            replace(
                base_config,
                backup_operator="max",
                train_targets="all",
                frontier_weight=0.1,
                output_dir=None,
                save_csv=None,
                save_json=None,
                print_every=0,
            ),
        ),
        (
            "max_exact_plus_child",
            replace(
                base_config,
                backup_operator="max",
                train_targets="exact_plus_child",
                output_dir=None,
                save_csv=None,
                save_json=None,
                print_every=0,
            ),
        ),
        (
            "clipped_max_all",
            replace(
                base_config,
                backup_operator="clipped_max",
                train_targets="all",
                output_dir=None,
                save_csv=None,
                save_json=None,
                print_every=0,
            ),
        ),
        (
            "softmax_all",
            replace(
                base_config,
                backup_operator="softmax",
                train_targets="all",
                output_dir=None,
                save_csv=None,
                save_json=None,
                print_every=0,
            ),
        ),
        (
            "max_all_pv_mean_pred",
            replace(
                base_config,
                backup_operator="max",
                train_targets="all",
                family_target_policy="pv_mean_prediction",
                output_dir=None,
                save_csv=None,
                save_json=None,
                print_every=0,
            ),
        ),
        (
            "max_all_pv_min_pred",
            replace(
                base_config,
                backup_operator="max",
                train_targets="all",
                family_target_policy="pv_min_prediction",
                output_dir=None,
                save_csv=None,
                save_json=None,
                print_every=0,
            ),
        ),
        (
            "max_all_pv_blend_mean",
            replace(
                base_config,
                backup_operator="max",
                train_targets="all",
                family_target_policy="pv_blend_mean_prediction",
                family_prediction_blend=0.5,
                output_dir=None,
                save_csv=None,
                save_json=None,
                print_every=0,
            ),
        ),
    )
    comparison: list[dict[str, object]] = []
    for strategy_name, config in strategies:
        result = run_toy_tree_sanity(config)
        final_metric = result.metrics[-1]
        comparison.append(
            {
                "strategy": strategy_name,
                "final_root_target": final_metric.root_target,
                "final_root_prediction": final_metric.root_prediction_after,
                "max_target_drift": max(
                    metric.max_abs_target_drift_all for metric in result.metrics
                ),
                "max_prediction_drift": max(
                    metric.max_abs_prediction_drift_all for metric in result.metrics
                ),
                "diverged": any(metric.diverged for metric in result.metrics),
                "root_argmax_history_short": _short_history(
                    metric.argmax_child_id_at_root for metric in result.metrics
                ),
            }
        )
    _print_comparison(comparison)
    return tuple(comparison)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for ``python -m ...evaluator_toy_tree_sanity``."""
    config, compare = _parse_args(argv)
    if compare:
        compare_strategies(config)
    else:
        run_toy_tree_sanity(config)
    return 0


def _parse_args(argv: list[str] | None) -> tuple[ToyRunConfig, bool]:
    parser = argparse.ArgumentParser(
        description="Run a tiny controlled tree bootstrap value-learning diagnostic."
    )
    parser.add_argument(
        "--case",
        choices=(
            "A_stable_terminal_wins",
            "B_optimistic_frontier_poison",
            "C_many_frontier_outliers",
            "D_deep_terminal_vs_frontier",
            "E_two_level_mixed",
            "F_linear_compositional_vicious_circle",
        ),
        default="B_optimistic_frontier_poison",
    )
    parser.add_argument(
        "--model-kind",
        choices=("mlp", "linear_no_bias"),
        default="mlp",
    )
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--init-mode",
        choices=("zero", "random", "optimistic_feature"),
        default="zero",
    )
    parser.add_argument(
        "--frontier-initial-override",
        action="append",
        default=(),
        metavar="NODE_ID=VALUE",
    )
    parser.add_argument(
        "--backup-operator",
        choices=("max", "softmax", "clipped_max"),
        default="max",
    )
    parser.add_argument("--backup-temperature", type=float, default=1.0)
    parser.add_argument("--frontier-clip", type=float, default=2.0)
    parser.add_argument(
        "--family-target-policy",
        choices=(
            "none",
            "pv_mean_prediction",
            "pv_min_prediction",
            "pv_quantile_prediction",
            "pv_blend_mean_prediction",
            "pv_blend_min_prediction",
            "pv_blend_quantile_prediction",
        ),
        default="none",
    )
    parser.add_argument("--family-quantile", type=float, default=0.25)
    parser.add_argument("--family-prediction-blend", type=float, default=0.5)
    parser.add_argument(
        "--train-targets",
        choices=("exact_only", "exact_plus_child", "all"),
        default="all",
    )
    parser.add_argument("--frontier-weight", type=float, default=0.1)
    parser.add_argument("--child-backup-weight", type=float, default=0.5)
    parser.add_argument("--exact-weight", type=float, default=1.0)
    parser.add_argument("--include-root", type=_parse_bool, default=True)
    parser.add_argument("--include-frontier-targets", type=_parse_bool, default=True)
    parser.add_argument("--use-direct-targets", action="store_true")
    parser.add_argument("--num-iterations", type=int, default=20)
    parser.add_argument("--eval-before-train", type=_parse_bool, default=True)
    parser.add_argument("--divergence-threshold", type=float, default=100.0)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--save-csv")
    parser.add_argument("--save-json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--compare-strategies", action="store_true")
    namespace = parser.parse_args(argv)

    config = ToyRunConfig(
        case=namespace.case,
        hidden_dim=namespace.hidden_dim,
        num_layers=namespace.num_layers,
        learning_rate=namespace.learning_rate,
        train_epochs=namespace.train_epochs,
        batch_size=namespace.batch_size,
        seed=namespace.seed,
        init_mode=namespace.init_mode,
        model_kind=namespace.model_kind,
        frontier_initial_overrides=_parse_overrides(
            namespace.frontier_initial_override
        ),
        backup_operator=namespace.backup_operator,
        backup_temperature=namespace.backup_temperature,
        frontier_clip=namespace.frontier_clip,
        family_target_policy=namespace.family_target_policy,
        family_quantile=namespace.family_quantile,
        family_prediction_blend=namespace.family_prediction_blend,
        train_targets=namespace.train_targets,
        frontier_weight=namespace.frontier_weight,
        child_backup_weight=namespace.child_backup_weight,
        exact_weight=namespace.exact_weight,
        include_root=namespace.include_root,
        include_frontier_targets=namespace.include_frontier_targets,
        use_direct_targets=namespace.use_direct_targets,
        num_iterations=namespace.num_iterations,
        eval_before_train=namespace.eval_before_train,
        divergence_threshold=namespace.divergence_threshold,
        print_every=namespace.print_every,
        output_dir=namespace.output_dir,
        save_csv=None if namespace.save_csv is None else Path(namespace.save_csv),
        save_json=None if namespace.save_json is None else Path(namespace.save_json),
    )
    return config, bool(namespace.compare_strategies)


def _make_tree(nodes: tuple[ToyNode, ...], root_id: int = 0) -> ToyTree:
    return ToyTree(nodes={node.node_id: node for node in nodes}, root_id=root_id)


def _final_linear_layer(network: nn.Sequential) -> nn.Linear:
    for module in reversed(network):
        if isinstance(module, nn.Linear):
            return module
    raise ValueError("ToyValueNet has no linear layer.")


def _backup_child_value(
    child: ToyBackedUpValue,
    *,
    backup_operator: BackupOperator,
    frontier_clip: float,
) -> float:
    if backup_operator != "clipped_max":
        return child.value
    pure_frontier = (
        child.source == "frontier_prediction"
        or (child.support_frontier_count > 0 and child.support_exact_count == 0)
    )
    return min(child.value, frontier_clip) if pure_frontier else child.value


def _principal_variation_representative(
    node_id: int,
    backed_up_values: dict[int, ToyBackedUpValue],
) -> int:
    seen: set[int] = set()
    current_id = node_id
    while True:
        if current_id in seen:
            raise ValueError("Principal-variation family traversal found a cycle.")
        seen.add(current_id)
        next_id = backed_up_values[current_id].argmax_child_id
        if next_id is None:
            return current_id
        current_id = next_id


def _family_policy_value(
    *,
    node_id: int,
    tree: ToyTree,
    backed_up_values: dict[int, ToyBackedUpValue],
    prediction_before: dict[int, float],
    family_target_policy: FamilyTargetPolicy,
) -> float:
    if family_target_policy.endswith("_backup"):
        return backed_up_values[node_id].value
    if family_target_policy.endswith("_prediction"):
        node = tree.nodes[node_id]
        if (node.is_terminal or node.is_exact) and node.exact_value is not None:
            return node.exact_value
        return prediction_before[node_id]
    raise ValueError(f"Unknown family target policy: {family_target_policy!r}.")


def _family_aggregate(
    values: list[float],
    *,
    family_target_policy: FamilyTargetPolicy,
    family_quantile: float,
) -> float:
    if not values:
        raise ValueError("Cannot aggregate an empty PV family.")
    if "_mean_" in family_target_policy:
        return _mean(values)
    if "_min_" in family_target_policy:
        return min(values)
    if "_quantile_" in family_target_policy:
        return float(np.quantile(np.asarray(values, dtype=np.float64), family_quantile))
    raise ValueError(f"Unknown family target policy: {family_target_policy!r}.")


def _linear_weights(model: nn.Module) -> tuple[float, ...] | None:
    if not isinstance(model, ToyLinearNoBiasNet):
        return None
    return tuple(
        float(value)
        for value in model.linear.weight.detach().cpu().flatten().tolist()
    )


def _linear_weight_at(weights: tuple[float, ...] | None, index: int) -> float | None:
    if weights is None or index >= len(weights):
        return None
    return weights[index]


def _linear_a_value_from_weights(weights: tuple[float, ...] | None) -> float | None:
    if weights is None or len(weights) < 3:
        return None
    return weights[1] + weights[2]


def _temperature_logsumexp(values: tuple[float, ...], *, temperature: float) -> float:
    scaled = tuple(value / temperature for value in values)
    maximum = max(scaled)
    return temperature * (
        maximum + math.log(sum(math.exp(value - maximum) for value in scaled))
    )


def _source_weight(
    source: TargetSource,
    *,
    exact_weight: float,
    child_backup_weight: float,
    frontier_weight: float,
) -> float:
    if source == "ground_truth_exact_or_terminal":
        return exact_weight
    if source == "child_backup":
        return child_backup_weight
    return frontier_weight


def _effective_frontier_initial_overrides(config: ToyRunConfig) -> dict[int, float]:
    if config.frontier_initial_overrides:
        return dict(config.frontier_initial_overrides)
    if config.case == "B_optimistic_frontier_poison":
        return {2: 10.0}
    if config.case == "D_deep_terminal_vs_frontier":
        return {4: 10.0}
    if config.case == "C_many_frontier_outliers":
        return {17: 10.0}
    if config.case == "E_two_level_mixed":
        return {3: 10.0, 5: 6.0}
    if config.case == "F_linear_compositional_vicious_circle":
        return {3: 10.0}
    return {}


def _iteration_metric(
    *,
    iteration: int,
    tree: ToyTree,
    backed_up: dict[int, ToyBackedUpValue],
    rows: tuple[ToyTrainingRow, ...],
    prediction_before: dict[int, float],
    prediction_after: dict[int, float],
    previous_targets: dict[int, float] | None,
    effective_targets: dict[int, float],
    final_loss: float,
    divergence_threshold: float,
    weights_after: tuple[float, ...] | None,
) -> ToyIterationMetric:
    root_value = backed_up[tree.root_id]
    prediction_drifts = [
        abs(prediction_after[node_id] - prediction_before[node_id])
        for node_id in tree.nodes
    ]
    target_drifts = (
        [0.0 for _ in tree.nodes]
        if previous_targets is None
        else [
            abs(effective_targets[node_id] - previous_targets[node_id])
            for node_id in tree.nodes
        ]
    )
    frontier_node_ids = [
        node_id for node_id, node in tree.nodes.items() if not node.child_ids and not (
            node.is_terminal or node.is_exact
        )
    ]
    row_counts = _row_source_counts(rows)
    max_abs_prediction = max(abs(value) for value in prediction_after.values())
    return ToyIterationMetric(
        iteration=iteration,
        root_target=effective_targets[tree.root_id],
        root_prediction_before=prediction_before[tree.root_id],
        root_prediction_after=prediction_after[tree.root_id],
        max_frontier_prediction_before=max(
            prediction_before[node_id] for node_id in frontier_node_ids
        )
        if frontier_node_ids
        else math.nan,
        max_frontier_prediction_after=max(
            prediction_after[node_id] for node_id in frontier_node_ids
        )
        if frontier_node_ids
        else math.nan,
        mean_abs_prediction_drift_all=_mean(prediction_drifts),
        max_abs_prediction_drift_all=max(prediction_drifts),
        mean_abs_target_drift_all=_mean(target_drifts),
        max_abs_target_drift_all=max(target_drifts),
        argmax_child_id_at_root=root_value.argmax_child_id,
        argmax_child_source_at_root=root_value.argmax_child_source,
        num_rows=len(rows),
        num_exact_rows=row_counts["ground_truth_exact_or_terminal"],
        num_child_backup_rows=row_counts["child_backup"],
        num_frontier_rows=row_counts["frontier_prediction"],
        weighted_train_loss_final=final_loss,
        unweighted_mae_all_rows=_row_mae(rows, prediction_after),
        mae_exact=_row_mae(rows, prediction_after, source="ground_truth_exact_or_terminal"),
        mae_child_backup=_row_mae(rows, prediction_after, source="child_backup"),
        mae_frontier=_row_mae(rows, prediction_after, source="frontier_prediction"),
        max_abs_prediction=max_abs_prediction,
        diverged=(
            abs(effective_targets[tree.root_id]) > divergence_threshold
            or max_abs_prediction > divergence_threshold
        ),
        linear_w0=_linear_weight_at(weights_after, 0),
        linear_w1=_linear_weight_at(weights_after, 1),
        linear_w2=_linear_weight_at(weights_after, 2),
        linear_a_value_from_weights=_linear_a_value_from_weights(weights_after),
    )


def _node_history_rows(
    *,
    iteration: int,
    tree: ToyTree,
    backed_up: dict[int, ToyBackedUpValue],
    rows: tuple[ToyTrainingRow, ...],
    prediction_before: dict[int, float],
    prediction_after: dict[int, float],
    effective_targets: dict[int, float],
) -> tuple[ToyNodeHistoryRow, ...]:
    row_by_node = {row.node_id: row for row in rows}
    history: list[ToyNodeHistoryRow] = []
    for node_id in tree.node_ids_by_depth():
        backed = backed_up[node_id]
        row = row_by_node.get(node_id)
        history.append(
            ToyNodeHistoryRow(
                iteration=iteration,
                node_id=node_id,
                prediction_before=prediction_before[node_id],
                prediction_after=prediction_after[node_id],
                target=effective_targets[node_id] if row is None else row.target,
                source=backed.source,
                weight=0.0 if row is None else row.weight,
                support_exact_count=backed.support_exact_count,
                support_frontier_count=backed.support_frontier_count,
                distance_to_exact=backed.distance_to_exact,
                argmax_child_id=backed.argmax_child_id,
                argmax_child_source=backed.argmax_child_source,
            )
        )
    return tuple(history)


def _row_source_counts(rows: tuple[ToyTrainingRow, ...]) -> dict[TargetSource, int]:
    return {
        "ground_truth_exact_or_terminal": sum(
            row.source == "ground_truth_exact_or_terminal" for row in rows
        ),
        "child_backup": sum(row.source == "child_backup" for row in rows),
        "frontier_prediction": sum(row.source == "frontier_prediction" for row in rows),
    }


def _row_mae(
    rows: tuple[ToyTrainingRow, ...],
    predictions: dict[int, float],
    *,
    source: TargetSource | None = None,
) -> float | None:
    selected = [row for row in rows if source is None or row.source == source]
    if not selected:
        return None
    return _mean([abs(predictions[row.node_id] - row.target) for row in selected])


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def _summary(
    *,
    config: ToyRunConfig,
    tree: ToyTree,
    metrics: tuple[ToyIterationMetric, ...],
) -> dict[str, object]:
    if not metrics:
        return {"status": "empty"}
    final = metrics[-1]
    drift_window = tuple(metric.max_abs_target_drift_all for metric in metrics[-5:])
    increasing_drift = len(drift_window) >= 3 and all(
        later > earlier for earlier, later in pairwise(drift_window)
    )
    status = "diverged" if any(metric.diverged for metric in metrics) else "stable"
    if status == "stable" and increasing_drift:
        status = "drifting"
    elif status == "stable" and final.max_abs_target_drift_all < 1e-3:
        status = "converged"
    frontier_node_ids = [
        node_id for node_id, node in tree.nodes.items() if not node.child_ids and not (
            node.is_terminal or node.is_exact
        )
    ]
    return {
        "status": status,
        "case": config.case,
        "model_kind": config.model_kind,
        "backup_operator": config.backup_operator,
        "family_target_policy": config.family_target_policy,
        "family_prediction_blend": config.family_prediction_blend,
        "train_targets": config.train_targets,
        "final_root_target": final.root_target,
        "final_root_prediction": final.root_prediction_after,
        "final_source_counts": {
            "exact": final.num_exact_rows,
            "child_backup": final.num_child_backup_rows,
            "frontier": final.num_frontier_rows,
        },
        "root_argmax_child_history": [
            metric.argmax_child_id_at_root for metric in metrics
        ],
        "root_target_history": [metric.root_target for metric in metrics],
        "frontier_node_ids": frontier_node_ids,
        "frontier_max_prediction_after_history": [
            metric.max_frontier_prediction_after for metric in metrics
        ],
        "linear_weight_history": [
            (metric.linear_w0, metric.linear_w1, metric.linear_w2)
            for metric in metrics
            if metric.linear_w0 is not None
        ],
        "linear_a_value_from_weights_history": [
            metric.linear_a_value_from_weights
            for metric in metrics
            if metric.linear_a_value_from_weights is not None
        ],
    }


def _print_header() -> None:
    print(
        "iter root_t root_pred_b root_pred_a max_front_b max_front_a "
        "pred_drift target_drift root_argmax rows loss mae linear"
    )


def _print_metric(metric: ToyIterationMetric) -> None:
    print(
        f"{metric.iteration:>4} "
        f"{metric.root_target:>7.3f} "
        f"{metric.root_prediction_before:>11.3f} "
        f"{metric.root_prediction_after:>11.3f} "
        f"{metric.max_frontier_prediction_before:>11.3f} "
        f"{metric.max_frontier_prediction_after:>11.3f} "
        f"{metric.max_abs_prediction_drift_all:>10.3f} "
        f"{metric.max_abs_target_drift_all:>12.3f} "
        f"{metric.argmax_child_id_at_root!s:>11} "
        f"{metric.num_rows:>4} "
        f"{metric.weighted_train_loss_final:>8.4f} "
        f"{metric.unweighted_mae_all_rows:>8.4f} "
        f"{_linear_weight_text(metric)}"
    )


def _print_summary(summary: dict[str, object]) -> None:
    print("\nSummary")
    print(f"status: {summary['status']}")
    print(f"final_root_target: {summary.get('final_root_target')}")
    print(f"final_root_prediction: {summary.get('final_root_prediction')}")
    print(f"root_argmax_child_history: {summary.get('root_argmax_child_history')}")
    print(f"root_target_history: {summary.get('root_target_history')}")
    print(
        "frontier_max_prediction_after_history: "
        f"{summary.get('frontier_max_prediction_after_history')}"
    )
    if summary.get("linear_weight_history"):
        print(f"linear_weight_history: {summary.get('linear_weight_history')}")
        print(
            "linear_a_value_from_weights_history: "
            f"{summary.get('linear_a_value_from_weights_history')}"
        )


def _linear_weight_text(metric: ToyIterationMetric) -> str:
    if metric.linear_w0 is None:
        return "w=-"
    return (
        f"w=[{metric.linear_w0:.3f},{metric.linear_w1:.3f},{metric.linear_w2:.3f}] "
        f"A={metric.linear_a_value_from_weights:.3f}"
    )


def _print_comparison(comparison: list[dict[str, object]]) -> None:
    print("\nStrategy comparison")
    print(
        "strategy final_root_target final_root_prediction max_target_drift "
        "max_prediction_drift diverged root_argmax_history_short"
    )
    for row in comparison:
        print(
            f"{row['strategy']} "
            f"{float(row['final_root_target']):.3f} "
            f"{float(row['final_root_prediction']):.3f} "
            f"{float(row['max_target_drift']):.3f} "
            f"{float(row['max_prediction_drift']):.3f} "
            f"{row['diverged']} "
            f"{row['root_argmax_history_short']}"
        )


def _short_history(values: object) -> str:
    sequence = list(values)
    if len(sequence) <= 8:
        return ",".join(str(value) for value in sequence)
    head = ",".join(str(value) for value in sequence[:4])
    tail = ",".join(str(value) for value in sequence[-3:])
    return f"{head},...,{tail}"


def _write_tree_config(path: Path, tree: ToyTree, config: ToyRunConfig) -> None:
    payload = {
        "config": _jsonable_config(config),
        "root_id": tree.root_id,
        "nodes": [asdict(tree.nodes[node_id]) for node_id in tree.node_ids_by_depth()],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: tuple[object, ...]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _jsonable_config(config: ToyRunConfig) -> dict[str, object]:
    payload = asdict(config)
    for key, value in tuple(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def _parse_overrides(raw_overrides: tuple[str, ...] | list[str]) -> dict[int, float]:
    overrides: dict[int, float] = {}
    for raw in raw_overrides:
        node_id_text, separator, value_text = raw.partition("=")
        if separator != "=":
            raise argparse.ArgumentTypeError(
                f"Expected NODE_ID=VALUE override, got {raw!r}."
            )
        overrides[int(node_id_text)] = float(value_text)
    return overrides


def _parse_bool(raw: str | bool) -> bool:
    if isinstance(raw, bool):
        return raw
    lowered = raw.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {raw!r}.")


if __name__ == "__main__":
    raise SystemExit(main())

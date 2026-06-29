"""Observability metadata helpers for Morpion artifact-pipeline stages."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from chipiron.environments.morpion.bootstrap.pipeline_memory import current_rss_mb

if TYPE_CHECKING:
    from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
        MorpionBootstrapPaths,
    )


def _optional_runner_mapping(
    runner: object,
    method_name: str,
) -> dict[str, object] | None:
    """Return one optional mapping exposed by the runner."""
    method = getattr(runner, method_name, None)
    if not callable(method):
        return None
    value = method()
    if not isinstance(value, Mapping):
        return None
    return dict(value)


def _optional_ratio(numerator: object, denominator: object) -> float | None:
    """Return one safe ratio for numeric dashboard metrics."""
    if not isinstance(numerator, int | float) or not isinstance(
        denominator, int | float
    ):
        return None
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _observability_metadata_for_dashboard(
    *,
    runner: object,
    generation: int,
    node_count: int,
    branch_count: int | None,
    branch_count_before_growth: int | None = None,
    growth_budget_metadata: Mapping[str, object] | None = None,
    nodes_added: int,
    growth_duration_s: float,
    cycle_duration_s: float,
    current_rss_provider: object = current_rss_mb,
) -> dict[str, object]:
    """Build compact status metadata for memory/checkpoint/export observability."""
    rss_mb = current_rss_provider() if callable(current_rss_provider) else None
    state_eviction = _optional_runner_mapping(runner, "profile_state_eviction_runtime")
    checkpoint = _optional_runner_mapping(runner, "latest_checkpoint_metrics")
    training_export = _optional_runner_mapping(runner, "latest_training_export_stats")
    training_export_profile = _optional_runner_mapping(
        runner,
        "latest_training_export_profile",
    )
    memory: dict[str, object] = {
        "event": "done",
        "generation": generation,
        "rss_mb": rss_mb,
        "rss_mb_per_100k_nodes": None
        if rss_mb is None or node_count <= 0
        else rss_mb * 100_000.0 / node_count,
    }
    if checkpoint is not None:
        memory["rss_before_checkpoint_save_mb"] = checkpoint.get("rss_before_mb")
        memory["rss_after_checkpoint_save_mb"] = checkpoint.get("rss_after_mb")
    if training_export is not None:
        memory["rss_before_training_export_mb"] = training_export.get("rss_before_mb")
        memory["rss_after_training_export_mb"] = training_export.get("rss_after_mb")
    tree: dict[str, object] = {
        "generation": generation,
        "node_count": node_count,
        "branch_count": branch_count,
        "branch_count_before_growth": branch_count_before_growth,
        "branch_count_after_growth": branch_count,
        "nodes_added": nodes_added,
        "cycle_elapsed_s": cycle_duration_s,
        "growth_elapsed_s": growth_duration_s,
        "branch_count_per_node": _optional_ratio(branch_count, node_count),
    }
    if growth_budget_metadata is not None:
        tree.update(dict(growth_budget_metadata))
    return {
        "tree": tree,
        "memory": memory,
        "state_eviction": {} if state_eviction is None else state_eviction,
        "checkpoint": {} if checkpoint is None else checkpoint,
        "training_export": {} if training_export is None else training_export,
        "training_export_profile": {}
        if training_export_profile is None
        else training_export_profile,
    }


def _configure_linoo_selection_artifact_for_growth(
    *,
    runner: object,
    paths: MorpionBootstrapPaths,
    cycle_index: int,
    generation: int,
) -> None:
    """Configure latest Linoo table persistence when the runner supports it."""
    configure = getattr(runner, "configure_linoo_selection_table_artifact", None)
    if not callable(configure):
        return
    configure(
        path=paths.latest_linoo_selection_table_path,
        cycle_index=cycle_index,
        generation=generation,
    )

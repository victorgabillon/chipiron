"""Rollout-related runtime logging helpers for Morpion bootstrap."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from anemone.tree_manager import (
    OpeningExpansionConfig,
    OpeningExpansionKind,
    RolloutActionSelectorKind,
    RolloutExpansionConfig,
)

if TYPE_CHECKING:
    from anemone.factory import SearchArgs

    from chipiron.environments.morpion.bootstrap.config import (
        MorpionBootstrapRolloutConfig,
    )

LOGGER = logging.getLogger(__name__)


def _metric_value(value: object) -> str:
    """Render one metric field as a stable log token."""
    if value is None:
        return "none"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _opening_expansion_config_from_rollout(
    rollout: MorpionBootstrapRolloutConfig | None,
) -> OpeningExpansionConfig:
    """Build Anemone opening-expansion config from persisted Morpion rollout."""
    if rollout is None or not rollout.enabled:
        return OpeningExpansionConfig()

    return OpeningExpansionConfig(
        kind=OpeningExpansionKind.ROLLOUT,
        rollout=RolloutExpansionConfig(
            max_extra_steps=rollout.max_extra_steps,
            action_selector_kind=RolloutActionSelectorKind(
                rollout.action_selector_kind
            ),
            random_seed=rollout.random_seed,
            stop_on_existing_node=rollout.stop_on_existing_node,
        ),
    )


def _opening_type_name(search_args: SearchArgs) -> str:
    """Return a stable operator-facing opening-type label for logs."""
    opening_type = search_args.opening_type
    value = getattr(opening_type, "value", None)
    return value if isinstance(value, str) else str(opening_type)


def _opening_expansion_kind_name(search_args: SearchArgs) -> str:
    """Return a stable operator-facing opening-expansion kind label."""
    opening_expansion = search_args.opening_expansion
    kind = getattr(opening_expansion, "kind", None)
    value = getattr(kind, "value", None)
    return value if isinstance(value, str) else str(kind)


def _log_search_rollout_config(search_args: SearchArgs) -> None:
    """Emit the effective rollout expansion config for operator logs."""
    opening_expansion = search_args.opening_expansion
    rollout = getattr(opening_expansion, "rollout", None)
    enabled = getattr(opening_expansion, "kind", None) == OpeningExpansionKind.ROLLOUT
    LOGGER.info(
        "[search] rollout enabled=%s max_extra_steps=%s action_selector=%s random_seed=%s stop_on_existing_node=%s",
        enabled,
        _metric_value(getattr(rollout, "max_extra_steps", None)),
        _metric_value(getattr(rollout, "action_selector_kind", None)),
        _metric_value(getattr(rollout, "random_seed", None)),
        _metric_value(getattr(rollout, "stop_on_existing_node", None)),
    )


def _log_latest_rollout_report(runtime: object) -> None:
    """Emit the latest Anemone rollout report when the runtime exposes one."""
    tree_manager = getattr(runtime, "tree_manager", None)
    report = getattr(tree_manager, "latest_rollout_report", None)
    if report is None:
        return
    LOGGER.info(
        "[rollout] total_edges=%s initial_edges=%s extra_edges=%s traversals=%s stops=%s",
        _metric_value(getattr(report, "total_edge_count", None)),
        _metric_value(getattr(report, "initial_edge_count", None)),
        _metric_value(getattr(report, "extra_edge_count", None)),
        _metric_value(getattr(report, "traversal_count", None)),
        _metric_value(getattr(report, "stop_reason_counts", None)),
    )
    path_reports = _rollout_path_reports(report)
    if not path_reports:
        return
    LOGGER.info(
        "[rollout-lengths] count=%s total_lengths=%s extra_lengths=%s stops=%s",
        len(path_reports),
        [
            getattr(path_report, "total_edge_count", None)
            for path_report in path_reports
        ],
        [
            getattr(path_report, "extra_edge_count", None)
            for path_report in path_reports
        ],
        _rollout_path_stop_counts(path_reports),
    )
    for rollout_index, path_report in enumerate(path_reports):
        LOGGER.info(
            "[rollout-detail] rollout_index=%s start_node_id=%s start_depth=%s "
            "end_node_id=%s end_depth=%s total_edges=%s initial_edges=%s "
            "extra_edges=%s traversals=%s stop_reason=%s end_terminal=%s "
            "end_exact=%s end_created_node=%s end_existing_node=%s "
            "end_legal_actions=%s end_openable_actions=%s end_opened_actions=%s "
            "end_non_opened_branches=%s no_legal_but_not_terminal=%s",
            rollout_index,
            _metric_value(getattr(path_report, "start_node_id", None)),
            _metric_value(getattr(path_report, "start_depth", None)),
            _metric_value(getattr(path_report, "end_node_id", None)),
            _metric_value(getattr(path_report, "end_depth", None)),
            _metric_value(getattr(path_report, "total_edge_count", None)),
            _metric_value(getattr(path_report, "initial_edge_count", None)),
            _metric_value(getattr(path_report, "extra_edge_count", None)),
            _metric_value(getattr(path_report, "traversal_count", None)),
            _metric_value(_rollout_path_stop_reason(path_report)),
            _metric_value(getattr(path_report, "end_is_terminal", None)),
            _metric_value(getattr(path_report, "end_is_exact", None)),
            _metric_value(getattr(path_report, "end_was_created_node", None)),
            _metric_value(getattr(path_report, "end_was_existing_node", None)),
            _metric_value(getattr(path_report, "end_legal_action_count", None)),
            _metric_value(getattr(path_report, "end_openable_action_count", None)),
            _metric_value(getattr(path_report, "end_opened_action_count", None)),
            _metric_value(getattr(path_report, "end_non_opened_branch_count", None)),
            _metric_value(_rollout_no_legal_but_not_terminal(path_report)),
        )
        if _rollout_no_legal_but_not_terminal(path_report):
            LOGGER.warning(
                "[rollout-warning] no_legal_actions_but_not_terminal "
                "rollout_index=%s end_node_id=%s end_depth=%s "
                "end_legal_actions=%s end_non_opened_branches=%s",
                rollout_index,
                _metric_value(getattr(path_report, "end_node_id", None)),
                _metric_value(getattr(path_report, "end_depth", None)),
                _metric_value(getattr(path_report, "end_legal_action_count", None)),
                _metric_value(
                    getattr(path_report, "end_non_opened_branch_count", None)
                ),
            )


def _rollout_path_reports(report: object) -> tuple[object, ...]:
    """Return path reports from an Anemone rollout report when available."""
    path_reports = getattr(report, "path_reports", ())
    if path_reports is None:
        return ()
    try:
        return tuple(path_reports)
    except TypeError:
        return ()


def _rollout_path_stop_counts(path_reports: tuple[object, ...]) -> dict[str, int]:
    """Aggregate stop reasons from rollout path reports."""
    stop_counts: dict[str, int] = {}
    for path_report in path_reports:
        stop_reason = _rollout_path_stop_reason(path_report)
        stop_counts[stop_reason] = stop_counts.get(stop_reason, 0) + 1
    return stop_counts


def _rollout_path_stop_reason(path_report: object) -> str:
    """Return one stable rollout path stop-reason token."""
    stop_reason = getattr(path_report, "stop_reason", None)
    value = getattr(stop_reason, "value", None)
    if isinstance(value, str):
        return value
    if stop_reason is None:
        return "none"
    return str(stop_reason)


def _rollout_no_legal_but_not_terminal(path_report: object) -> bool:
    """Return whether a path stopped with no legal actions but is not terminal."""
    reported_flag = getattr(path_report, "no_legal_actions_but_not_terminal", None)
    if isinstance(reported_flag, bool):
        return reported_flag
    return (
        _rollout_path_stop_reason(path_report) == "no_legal_actions"
        and getattr(path_report, "end_is_terminal", None) is False
    )

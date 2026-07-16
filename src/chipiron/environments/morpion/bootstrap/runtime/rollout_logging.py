"""Rollout-related runtime logging helpers for Morpion bootstrap."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
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
_VERBOSE_ROLLOUT_DETAILS_ENV = "MORPION_VERBOSE_ROLLOUT_DETAILS"


@dataclass(frozen=True, slots=True)
class RolloutLogSummary:
    """Compact rollout metrics emitted for one growth step."""

    paths: int
    total_edges: object
    initial_edges: object
    extra_edges: object
    traversals: object
    stops: object
    total_len: str
    extra_len: str
    start_depth: str
    end_depth: str
    depth_delta: str

    @property
    def enabled(self) -> bool:
        """Return whether a rollout report was present for this step."""
        return True


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


def _log_latest_rollout_report(
    runtime: object,
    *,
    step: int | None = None,
    verbose_details: bool | None = None,
) -> RolloutLogSummary | None:
    """Emit the latest Anemone rollout report when the runtime exposes one."""
    tree_manager = getattr(runtime, "tree_manager", None)
    report = getattr(tree_manager, "latest_rollout_report", None)
    if report is None:
        return None
    path_reports = _rollout_path_reports(report)
    stop_counts = (
        _rollout_path_stop_counts(path_reports)
        if path_reports
        else getattr(report, "stop_reason_counts", None)
    )
    summary = RolloutLogSummary(
        paths=len(path_reports),
        total_edges=getattr(report, "total_edge_count", None),
        initial_edges=getattr(report, "initial_edge_count", None),
        extra_edges=getattr(report, "extra_edge_count", None),
        traversals=getattr(report, "traversal_count", None),
        stops=stop_counts,
        total_len=_rollout_path_length_summary(path_reports, "total_edge_count"),
        extra_len=_rollout_path_length_summary(path_reports, "extra_edge_count"),
        start_depth=_rollout_path_value_summary(path_reports, "start_depth"),
        end_depth=_rollout_path_value_summary(path_reports, "end_depth"),
        depth_delta=_rollout_depth_delta_summary(path_reports),
    )
    LOGGER.debug(
        "[rollout-summary] step=%s paths=%s edges=%s initial=%s extra=%s "
        "traversals=%s stops=%s total_len=%s extra_len=%s "
        "start_depth=%s end_depth=%s depth_delta=%s",
        _metric_value(step),
        summary.paths,
        _metric_value(summary.total_edges),
        _metric_value(summary.initial_edges),
        _metric_value(summary.extra_edges),
        _metric_value(summary.traversals),
        _format_stop_counts(summary.stops),
        summary.total_len,
        summary.extra_len,
        summary.start_depth,
        summary.end_depth,
        summary.depth_delta,
    )
    LOGGER.info(
        "[rollout-execution] step=%s paths=%s %s",
        _metric_value(step),
        summary.paths,
        _format_rollout_execution(path_reports, stop_counts),
    )
    if not path_reports:
        return summary
    if verbose_details is None:
        verbose_details = _env_flag_enabled(_VERBOSE_ROLLOUT_DETAILS_ENV)
    detail_log = LOGGER.info if verbose_details else LOGGER.debug
    for rollout_index, path_report in enumerate(path_reports):
        detail_log(
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
                "no_legal_but_not_terminal=True rollout_index=%s "
                "end_node_id=%s end_depth=%s "
                "end_legal_actions=%s end_non_opened_branches=%s",
                rollout_index,
                _metric_value(getattr(path_report, "end_node_id", None)),
                _metric_value(getattr(path_report, "end_depth", None)),
                _metric_value(getattr(path_report, "end_legal_action_count", None)),
                _metric_value(
                    getattr(path_report, "end_non_opened_branch_count", None)
                ),
            )
    return summary


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


def _rollout_path_length_summary(
    path_reports: tuple[object, ...],
    attr_name: str,
) -> str:
    """Return min/mean/max path-length stats as one compact log token."""
    values = [
        value
        for path_report in path_reports
        if isinstance((value := getattr(path_report, attr_name, None)), int | float)
    ]
    if not values:
        return "none"
    return (
        f"min:{min(values):g} mean:{sum(values) / len(values):.1f} max:{max(values):g}"
    )


def _rollout_path_value_summary(
    path_reports: tuple[object, ...],
    attr_name: str,
) -> str:
    """Return a single value or compact stats for one rollout path attribute."""
    values = _numeric_path_values(path_reports, attr_name)
    if not values:
        return "none"
    if len(values) == 1:
        return _metric_value(values[0])
    return _format_numeric_summary(values)


def _rollout_depth_delta_summary(path_reports: tuple[object, ...]) -> str:
    """Return a single value or compact stats for rollout depth deltas."""
    deltas = _rollout_depth_deltas(path_reports)
    if not deltas:
        return "none"
    if len(deltas) == 1:
        return _metric_value(deltas[0])
    return _format_numeric_summary(deltas)


def _format_rollout_execution(
    path_reports: tuple[object, ...],
    stop_counts: object,
) -> str:
    """Format the INFO-level rollout execution aggregate."""
    if not path_reports:
        return (
            "start_depth=none end_depth=none depth_delta=none "
            f"stops={_format_stop_counts(stop_counts)} end_terminal=0/0 "
            "end_legal_actions=none"
        )
    path_count = len(path_reports)
    start_depths = _numeric_path_values(path_reports, "start_depth")
    end_depths = _numeric_path_values(path_reports, "end_depth")
    depth_delta_values = _rollout_depth_delta_values(path_reports)
    depth_deltas = _numeric_values(depth_delta_values)
    legal_actions = _numeric_path_values(path_reports, "end_legal_action_count")
    openable_actions = _numeric_path_values(path_reports, "end_openable_action_count")
    fragments: list[str] = []
    if path_count <= 5:
        fragments.extend((
            f"start_depths={_format_list_token(_path_values(path_reports, 'start_depth'))}",
            f"end_depths={_format_list_token(_path_values(path_reports, 'end_depth'))}",
            f"depth_deltas={_format_list_token(depth_delta_values)}",
            f"end_node_ids={_format_list_token(_path_values(path_reports, 'end_node_id'))}",
        ))
    else:
        fragments.extend((
            f"start_depth={_format_numeric_summary_or_none(start_depths)}",
            f"end_depth={_format_numeric_summary_or_none(end_depths)}",
            f"depth_delta={_format_numeric_summary_or_none(depth_deltas)}",
        ))
    fragments.extend((
        f"stops={_format_stop_counts(stop_counts)}",
        f"end_terminal={_rollout_terminal_count(path_reports)}/{path_count}",
        f"end_legal_actions={_format_numeric_summary_or_none(legal_actions)}",
    ))
    if openable_actions:
        fragments.append(
            f"end_openable_actions={_format_numeric_summary(openable_actions)}"
        )
    fragments.extend(_rollout_created_existing_fragments(path_reports))
    return " ".join(fragments)


def _path_values(
    path_reports: tuple[object, ...],
    attr_name: str,
) -> tuple[object, ...]:
    """Return raw path-report values for one attribute."""
    return tuple(getattr(path_report, attr_name, None) for path_report in path_reports)


def _numeric_path_values(
    path_reports: tuple[object, ...],
    attr_name: str,
) -> tuple[int | float, ...]:
    """Return numeric path-report values for one attribute."""
    return tuple(
        value
        for path_report in path_reports
        if isinstance((value := getattr(path_report, attr_name, None)), int | float)
    )


def _rollout_depth_deltas(path_reports: tuple[object, ...]) -> tuple[int | float, ...]:
    """Return rollout depth deltas for paths with numeric start and end depths."""
    return _numeric_values(_rollout_depth_delta_values(path_reports))


def _rollout_depth_delta_values(path_reports: tuple[object, ...]) -> tuple[object, ...]:
    """Return rollout depth deltas, preserving one value per path."""
    deltas: list[object] = []
    for path_report in path_reports:
        start_depth = getattr(path_report, "start_depth", None)
        end_depth = getattr(path_report, "end_depth", None)
        if isinstance(start_depth, int | float) and isinstance(end_depth, int | float):
            deltas.append(end_depth - start_depth)
        else:
            deltas.append(None)
    return tuple(deltas)


def _numeric_values(values: tuple[object, ...]) -> tuple[int | float, ...]:
    """Return numeric values from a raw value tuple."""
    return tuple(value for value in values if isinstance(value, int | float))


def _format_list_token(values: tuple[object, ...]) -> str:
    """Format a short value list without Python quotes."""
    return "[" + ",".join(_metric_value(value) for value in values) + "]"


def _format_numeric_summary_or_none(values: tuple[int | float, ...]) -> str:
    """Format numeric stats, preserving an explicit none token for empty data."""
    return _format_numeric_summary(values) if values else "none"


def _format_numeric_summary(values: tuple[int | float, ...]) -> str:
    """Return min/mean/max stats as one compact log token."""
    return (
        f"min:{min(values):g} mean:{sum(values) / len(values):.1f} max:{max(values):g}"
    )


def _rollout_terminal_count(path_reports: tuple[object, ...]) -> int:
    """Count rollout paths whose final node is reported terminal."""
    return sum(
        1
        for path_report in path_reports
        if getattr(path_report, "end_is_terminal", None) is True
    )


def _rollout_created_existing_fragments(
    path_reports: tuple[object, ...],
) -> tuple[str, ...]:
    """Return created/existing-node fragments when the path reports expose them."""
    fragments: list[str] = []
    created_values = _bool_path_values(path_reports, "end_was_created_node")
    if created_values:
        fragments.append(f"created_node={sum(created_values)}/{len(path_reports)}")
    existing_values = _bool_path_values(path_reports, "end_was_existing_node")
    if existing_values:
        fragments.append(f"existing_node={sum(existing_values)}/{len(path_reports)}")
    return tuple(fragments)


def _bool_path_values(
    path_reports: tuple[object, ...],
    attr_name: str,
) -> tuple[bool, ...]:
    """Return boolean path-report values for one attribute."""
    return tuple(
        value
        for path_report in path_reports
        if isinstance((value := getattr(path_report, attr_name, None)), bool)
    )


def _format_stop_counts(value: object) -> str:
    """Format stop counts without Python dict quotes when possible."""
    if not isinstance(value, Mapping):
        return _metric_value(value)
    return "{" + ",".join(f"{key}:{count}" for key, count in value.items()) + "}"


def _env_flag_enabled(name: str) -> bool:
    """Return whether an operator-facing boolean env flag is enabled."""
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes", "on"}


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

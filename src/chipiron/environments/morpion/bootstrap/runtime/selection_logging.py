"""Formatting helpers for Morpion growth selection logs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .checkpoint_io import _metric_value

__all__ = [
    "checkpoint_selector_state_fields",
    "format_linoo_selection_depth_table",
    "format_optional_int_log",
    "format_selected_metric",
    "linoo_depth_active_column",
    "resolve_selected_int",
    "selector_growth_diagnostic_fields",
    "selector_heap_detail_fields",
    "selector_report_row_count",
]


def format_optional_int_log(value: object) -> str:
    """Format one optional integer for stable structured logs."""
    return str(value) if isinstance(value, int) else "unknown"


def format_selected_metric(value: object) -> str:
    """Format selected node/depth fields so missing values are visually obvious."""
    return str(value) if isinstance(value, int) else "unknown"


def _format_optional_table_int(value: object) -> str:
    """Format an optional integer for human-facing aligned tables."""
    return str(value) if isinstance(value, int) else "-"


def _format_optional_table_float(value: object) -> str:
    """Format an optional float for human-facing aligned tables."""
    return f"{float(value):.3f}" if isinstance(value, int | float) else "-"


def _format_text_table(
    headers: tuple[str, ...],
    rows: Sequence[Sequence[str]],
) -> str:
    """Format string rows as a simple aligned whitespace table."""
    widths = tuple(
        max([len(headers[column]), *(len(row[column]) for row in rows)])
        for column in range(len(headers))
    )
    formatted_rows = tuple(
        " ".join(value.rjust(widths[column]) for column, value in enumerate(row))
        for row in rows
    )
    return "\n".join((
        " ".join(header.rjust(widths[column]) for column, header in enumerate(headers)),
        *formatted_rows,
    ))


def resolve_selected_int(
    *,
    explicit_value: object,
    report: object,
    report_attribute: str,
) -> int | None:
    """Return an explicit selected metric, falling back to the selector report."""
    if isinstance(explicit_value, int):
        return explicit_value
    report_value = getattr(report, report_attribute, None)
    return report_value if isinstance(report_value, int) else None


def format_linoo_selection_depth_table(
    *,
    selector_report: object,
    selected_depth: object,
) -> str | None:
    """Return the step-scoped Linoo depth table body with a marked selected row."""
    depth_rows = getattr(selector_report, "depth_rows", None)
    if depth_rows is None:
        return None
    resolved_selected_depth = resolve_selected_int(
        explicit_value=selected_depth,
        report=selector_report,
        report_attribute="selected_depth",
    )
    headers = (
        "mark",
        "depth",
        "total",
        "opened",
        "frontier",
        "terminal",
        "exact",
        "uncached_terminal",
        "non_openable",
        "deterministic_index",
        "weight",
        "probability",
    )
    rows: tuple[tuple[str, ...], ...] = tuple(
        (
            "*" if row_depth == resolved_selected_depth else "",
            _format_optional_table_int(row_depth),
            _format_optional_table_int(getattr(row, "total_nodes", None)),
            _format_optional_table_int(getattr(row, "opened_count", None)),
            _format_optional_table_int(getattr(row, "frontier_count", None)),
            _format_optional_table_int(getattr(row, "terminal_count", None)),
            _format_optional_table_int(getattr(row, "exact_count", None)),
            _format_optional_table_int(
                getattr(row, "uncached_terminal_candidates", None)
            ),
            _format_optional_table_int(getattr(row, "non_openable_count", None)),
            _format_optional_table_int(getattr(row, "selection_index", None)),
            _format_optional_table_float(getattr(row, "selection_weight", None)),
            _format_optional_table_float(getattr(row, "selection_probability", None)),
        )
        for row in depth_rows
        for row_depth in (getattr(row, "depth", None),)
    )
    return _format_text_table(headers, rows)


def linoo_depth_active_column(selector_report: object) -> str:
    """Return the report-table column used by the active depth subpolicy."""
    if getattr(selector_report, "depth_selection_subpolicy", None) == "inverse_depth":
        return "probability"
    return "deterministic_index"


def format_mapping_metric(value: object) -> str:
    """Format a compact mapping as a stable log token."""
    if not isinstance(value, Mapping):
        return _metric_value(value)
    return "{" + ",".join(f"{key}:{count}" for key, count in value.items()) + "}"


def selector_report_row_count(selector_report: object | None) -> int | None:
    """Return the selector report row count when exposed by the report."""
    if selector_report is None:
        return None
    depth_row_count = getattr(selector_report, "depth_row_count", None)
    if isinstance(depth_row_count, int):
        return depth_row_count
    depth_rows = getattr(selector_report, "depth_rows", None)
    if depth_rows is None:
        return None
    try:
        row_count = len(depth_rows)
    except TypeError:
        return None
    return row_count


def selector_growth_diagnostic_fields(
    selector_report: object | None,
) -> dict[str, object]:
    """Return stable optional selector diagnostics for growth-step logs."""
    return {
        "selector_state_rebuilt": getattr(selector_report, "state_rebuilt", None),
        "selector_nodes_incrementally_updated": getattr(
            selector_report,
            "nodes_incrementally_updated",
            None,
        ),
        "selector_total_nodes_scanned": getattr(
            selector_report,
            "total_nodes_scanned",
            None,
        ),
        "selector_frontier_nodes_scanned": getattr(
            selector_report,
            "frontier_nodes_scanned",
            None,
        ),
    }


def selector_heap_detail_fields(selector_report: object | None) -> dict[str, object]:
    """Return optional Linoo heap detail counters for growth-step logs."""
    return {
        "candidate_count": getattr(
            selector_report, "heap_update_candidate_count", None
        ),
        "push_count": getattr(selector_report, "heap_update_push_count", None),
        "pop_count": getattr(selector_report, "heap_update_pop_count", None),
        "stale_skip_count": getattr(
            selector_report, "heap_update_stale_skip_count", None
        ),
        "signature_check_count": getattr(
            selector_report, "heap_update_signature_check_count", None
        ),
        "signature_recompute_count": getattr(
            selector_report, "heap_update_signature_recompute_count", None
        ),
        "version_mismatch_count": getattr(
            selector_report, "heap_update_version_mismatch_count", None
        ),
        "total_heap_entries": getattr(
            selector_report, "heap_update_total_heap_entries", None
        ),
        "max_heap_size": getattr(selector_report, "heap_update_max_heap_size", None),
        "depth_count": getattr(selector_report, "heap_update_depth_count", None),
        "frontier_node_count_seen": getattr(
            selector_report, "heap_update_frontier_node_count_seen", None
        ),
    }


def checkpoint_selector_state_fields(
    payload: object,
    *,
    prefix: str,
) -> dict[str, object]:
    """Return stable selector-state presence fields for checkpoint logs."""
    selector_state = getattr(payload, "selector_state", None)
    selector_state_type = getattr(selector_state, "type", None)
    selector_state_version = getattr(selector_state, "version", None)
    return {
        f"{prefix}_selector_state_present": selector_state is not None,
        f"{prefix}_selector_state_type": selector_state_type,
        f"{prefix}_selector_state_version": selector_state_version,
    }

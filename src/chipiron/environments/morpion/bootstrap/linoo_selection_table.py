"""Latest Linoo selector table artifact helpers for Morpion bootstrap."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class LinooSelectionTableRow:
    """One depth row in the persisted latest Linoo selection table."""

    depth: int
    opened: int
    frontier: int
    index: int
    best_node: int | None
    best_value: float | None
    selected: bool


@dataclass(frozen=True, slots=True)
class LinooSelectionTable:
    """Small latest-state artifact for the Linoo selector."""

    updated_at_utc: str
    cycle_index: int | None
    generation: int | None
    step: int
    selected_depth: int | None
    selected_node_id: int | None
    rows: tuple[LinooSelectionTableRow, ...]


def latest_linoo_selection_table_path(work_dir: str | Path) -> Path:
    """Return the stable latest Linoo selection table artifact path."""
    return Path(work_dir).resolve() / "pipeline" / "latest_linoo_selection_table.json"


def linoo_selection_table_from_report(
    *,
    report: object,
    updated_at_utc: str,
    cycle_index: int | None,
    generation: int | None,
    step: int,
    selected_depth: int | None = None,
    selected_node_id: int | None = None,
) -> LinooSelectionTable | None:
    """Build a persisted table from an Anemone LinooSelectionReport-like object."""
    depth_rows = getattr(report, "depth_rows", None)
    if depth_rows is None:
        return None

    report_selected_depth = getattr(report, "selected_depth", None)
    report_selected_node_id = getattr(report, "selected_node_id", None)
    resolved_selected_depth = (
        selected_depth if isinstance(selected_depth, int) else report_selected_depth
    )
    resolved_selected_node_id = (
        selected_node_id
        if isinstance(selected_node_id, int)
        else report_selected_node_id
    )
    if not isinstance(resolved_selected_depth, int):
        resolved_selected_depth = None
    if not isinstance(resolved_selected_node_id, int):
        resolved_selected_node_id = None

    rows: list[LinooSelectionTableRow] = []
    for row in depth_rows:
        depth = getattr(row, "depth", None)
        opened_count = getattr(row, "opened_count", None)
        frontier_count = getattr(row, "frontier_count", None)
        if not (
            isinstance(depth, int)
            and isinstance(opened_count, int)
            and isinstance(frontier_count, int)
        ):
            return None
        best_node_id = getattr(row, "best_node_id", None)
        best_direct_value = getattr(row, "best_direct_value", None)
        rows.append(
            LinooSelectionTableRow(
                depth=depth,
                opened=opened_count,
                frontier=frontier_count,
                index=opened_count * (depth + 1),
                best_node=best_node_id if isinstance(best_node_id, int) else None,
                best_value=(
                    float(best_direct_value)
                    if isinstance(best_direct_value, int | float)
                    else None
                ),
                selected=depth == resolved_selected_depth,
            )
        )

    rows.sort(key=lambda row: (row.index, row.depth))
    return LinooSelectionTable(
        updated_at_utc=updated_at_utc,
        cycle_index=cycle_index,
        generation=generation,
        step=step,
        selected_depth=resolved_selected_depth,
        selected_node_id=resolved_selected_node_id,
        rows=tuple(rows),
    )


def linoo_selection_table_to_dict(
    table: LinooSelectionTable,
) -> dict[str, object]:
    """Serialize one latest Linoo table to the stable JSON schema."""
    return {
        "updated_at_utc": table.updated_at_utc,
        "cycle_index": table.cycle_index,
        "generation": table.generation,
        "step": table.step,
        "selected_depth": table.selected_depth,
        "selected_node_id": table.selected_node_id,
        "rows": [
            {
                "depth": row.depth,
                "opened": row.opened,
                "frontier": row.frontier,
                "index": row.index,
                "best_node": row.best_node,
                "best_value": row.best_value,
                "selected": row.selected,
            }
            for row in table.rows
        ],
    }


def save_linoo_selection_table(table: LinooSelectionTable, path: str | Path) -> None:
    """Atomically persist the latest Linoo selection table."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(linoo_selection_table_to_dict(table), handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(output_path)


def load_linoo_selection_table(path: str | Path) -> LinooSelectionTable | None:
    """Load the latest Linoo table, returning None when absent or malformed."""
    input_path = Path(path)
    if not input_path.exists():
        return None
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return _linoo_selection_table_from_payload(payload)


def _linoo_selection_table_from_payload(
    payload: dict[Any, Any],
) -> LinooSelectionTable | None:
    updated_at_utc = payload.get("updated_at_utc")
    step = payload.get("step")
    rows_payload = payload.get("rows")
    if not isinstance(updated_at_utc, str) or not isinstance(step, int):
        return None
    if not isinstance(rows_payload, list):
        return None
    rows: list[LinooSelectionTableRow] = []
    for row_payload in rows_payload:
        if not isinstance(row_payload, dict):
            return None
        row = _linoo_selection_table_row_from_payload(row_payload)
        if row is None:
            return None
        rows.append(row)
    return LinooSelectionTable(
        updated_at_utc=updated_at_utc,
        cycle_index=_optional_int(payload.get("cycle_index")),
        generation=_optional_int(payload.get("generation")),
        step=step,
        selected_depth=_optional_int(payload.get("selected_depth")),
        selected_node_id=_optional_int(payload.get("selected_node_id")),
        rows=tuple(rows),
    )


def _linoo_selection_table_row_from_payload(
    payload: dict[Any, Any],
) -> LinooSelectionTableRow | None:
    depth = payload.get("depth")
    opened = payload.get("opened")
    frontier = payload.get("frontier")
    index = payload.get("index")
    selected = payload.get("selected")
    if not (
        isinstance(depth, int)
        and isinstance(opened, int)
        and isinstance(frontier, int)
        and isinstance(index, int)
        and isinstance(selected, bool)
    ):
        return None
    return LinooSelectionTableRow(
        depth=depth,
        opened=opened,
        frontier=frontier,
        index=index,
        best_node=_optional_int(payload.get("best_node")),
        best_value=_optional_float(payload.get("best_value")),
        selected=selected,
    )


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _optional_float(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) else None


__all__ = [
    "LinooSelectionTable",
    "LinooSelectionTableRow",
    "latest_linoo_selection_table_path",
    "linoo_selection_table_from_report",
    "linoo_selection_table_to_dict",
    "load_linoo_selection_table",
    "save_linoo_selection_table",
]

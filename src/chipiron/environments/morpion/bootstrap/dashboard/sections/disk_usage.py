"""Dashboard disk-usage section rendering."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from chipiron.environments.morpion.bootstrap.dashboard.formatting import (
    format_disk_usage_pct as _format_disk_usage_pct,
)
from chipiron.environments.morpion.bootstrap.dashboard.history_view import (
    format_num_bytes,
)

if TYPE_CHECKING:
    from chipiron.environments.morpion.bootstrap.dashboard.history_view import (
        DiskUsageSummary,
    )

__all__ = ["render_disk_usage_section"]


def render_disk_usage_section(
    *,
    st: Any,
    summary: DiskUsageSummary,
) -> None:
    """Render operator-facing run and device disk usage information."""
    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Run Dir Size",
        format_num_bytes(summary.run_dir_num_bytes),
        delta=_format_disk_usage_pct(summary.run_dir_pct_of_device_total),
    )
    metric_columns[1].metric(
        "Device Free Space",
        format_num_bytes(summary.device_free_num_bytes),
    )
    metric_columns[2].metric(
        "Device Used Space",
        format_num_bytes(summary.device_used_num_bytes),
    )
    metric_columns[3].metric(
        "Device Total Space",
        format_num_bytes(summary.device_total_num_bytes),
    )
    breakdown_rows = [
        {"artifact_group": row.label, "size": format_num_bytes(row.num_bytes)}
        for row in summary.breakdown_rows
    ]
    st.dataframe(breakdown_rows, width="stretch", hide_index=True)
    st.caption(
        "Run breakdown is sorted largest-first. Retention keeps only the latest "
        "checkpoint and tree export by default."
    )

"""Dashboard evaluator training diagnostics rendering."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from chipiron.environments.morpion.bootstrap.dashboard.formatting import (
    format_value as _format_value,
)
from chipiron.environments.morpion.bootstrap.evaluator_diagnostics import (
    MorpionEvaluatorTrainingDiagnostics,
    load_latest_evaluator_training_diagnostics,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from chipiron.environments.morpion.bootstrap.evaluator_diagnostics import (
        MorpionEvaluatorDiagnosticExample,
    )

__all__ = [
    "diagnostic_examples_rows",
    "has_known_optional_series_values",
    "load_latest_evaluator_training_diagnostics_for_dashboard",
    "render_evaluator_training_diagnostics_section",
]


def render_evaluator_training_diagnostics_section(
    *,
    st: Any,
    work_dir: Path,
) -> None:
    """Render the latest persisted evaluator diagnostics for one work directory."""
    diagnostics_by_evaluator = load_latest_evaluator_training_diagnostics_for_dashboard(
        work_dir
    )
    if not diagnostics_by_evaluator:
        st.caption("No evaluator diagnostics have been saved yet.")
        return

    evaluator_names = tuple(sorted(diagnostics_by_evaluator))
    selected_evaluator_name = st.selectbox(
        "Diagnostics evaluator",
        options=evaluator_names,
        key="evaluator_training_diagnostics_name",
    )
    diagnostics = diagnostics_by_evaluator[selected_evaluator_name]
    summary_columns = st.columns(5)
    summary_columns[0].metric("Generation", diagnostics.generation)
    summary_columns[1].metric("Dataset Size", diagnostics.dataset_size)
    summary_columns[2].metric("MAE Before", _format_value(diagnostics.mae_before))
    summary_columns[3].metric("MAE After", _format_value(diagnostics.mae_after))
    summary_columns[4].metric(
        "Max Error After",
        _format_value(diagnostics.max_abs_error_after),
    )
    st.caption(
        f"Created at {diagnostics.created_at} UTC. "
        "Representative rows are deterministic scale windows; worst rows are sorted by post-training absolute error."
    )
    st.markdown("Representative Examples")
    st.dataframe(
        diagnostic_examples_rows(diagnostics.representative_examples),
        width="stretch",
        hide_index=True,
    )
    st.markdown("Worst Error Examples")
    worst_rows = diagnostic_examples_rows(diagnostics.worst_examples)
    if worst_rows:
        st.dataframe(worst_rows, width="stretch", hide_index=True)
    else:
        st.caption(
            "No post-training predictions were available for worst-error ranking."
        )


def load_latest_evaluator_training_diagnostics_for_dashboard(
    work_dir: str | Path,
) -> dict[str, MorpionEvaluatorTrainingDiagnostics]:
    """Load latest evaluator diagnostics, tolerating absent artifacts."""
    return load_latest_evaluator_training_diagnostics(work_dir)


def diagnostic_examples_rows(
    examples: Sequence[MorpionEvaluatorDiagnosticExample],
) -> list[dict[str, object | None]]:
    """Return dashboard-friendly diagnostic example rows."""
    return [
        {
            "row_index": example.row_index,
            "node_id": example.node_id,
            "state_tag": example.state_tag,
            "depth": example.depth,
            "target_value": example.target_value,
            "prediction_before": example.prediction_before,
            "prediction_after": example.prediction_after,
            "abs_error_before": example.abs_error_before,
            "abs_error_after": example.abs_error_after,
        }
        for example in examples
    ]


def has_known_optional_series_values(series: tuple[Any, ...]) -> bool:
    """Return whether one optional-value time series contains any known value."""
    return any(getattr(point, "value", None) is not None for point in series)

"""The first screen: truthful state, scale, and three useful trends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from chipiron.environments.morpion.bootstrap.dashboard.theme import (
    PALETTE,
    metric_row,
    page_header,
    worker_strip,
)
from chipiron.environments.morpion.bootstrap.dashboard.view_model import mapping, number

if TYPE_CHECKING:
    from chipiron.environments.morpion.bootstrap.dashboard.view_model import (
        BootstrapDashboardSnapshot,
    )


def trend(
    st: Any, *, title: str, rows: list[dict[str, Any]], color: str, key: str
) -> None:
    """Render bounded interactive time data without changing reusable plot APIs."""
    import plotly.graph_objects as go

    with st.container(border=True):
        st.markdown(f"**{title}**")
        if not rows:
            st.caption("Waiting for the first saved observation.")
            return
        figure = go.Figure()
        groups = sorted({str(row.get("series", title)) for row in rows})
        for index, group in enumerate(groups):
            points = [row for row in rows if str(row.get("series", title)) == group]
            figure.add_trace(
                go.Scatter(
                    x=[p["time"] for p in points],
                    y=[p["value"] for p in points],
                    name=group,
                    mode="lines+markers",
                    line={
                        "color": color if index == 0 else PALETTE["blue"],
                        "width": 2,
                    },
                    marker={"size": 4},
                    hovertemplate="%{x}<br>%{y:,.4g}<extra>%{fullData.name}</extra>",
                )
            )
        figure.update_layout(
            height=210,
            margin={"l": 8, "r": 8, "t": 12, "b": 8},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "Arial, sans-serif", "size": 11, "color": "#697873"},
            showlegend=len(groups) > 1,
            legend={"orientation": "h", "y": 1.25},
            xaxis={"showgrid": False, "title": None, "nticks": 3, "tickangle": 0},
            yaxis={"gridcolor": "#edf0eb", "zeroline": False, "title": None},
        )
        st.plotly_chart(
            figure, width="stretch", config={"displayModeBar": False}, key=key
        )


def history_rows(
    snapshot: BootstrapDashboardSnapshot, section: str, field: str
) -> list[dict[str, Any]]:
    """Extract only present numeric observations; never interpolate missing values."""
    if section == "record" and snapshot.dataset_history:
        rows = []
        best = None
        for artifact in snapshot.dataset_history:
            record = mapping(artifact.get("record_status"))
            value = number(record.get(field))
            certified = (
                record.get("current_best_is_exact") is True
                or record.get("current_best_is_terminal") is True
            )
            if value is not None and certified:
                best = value if best is None else max(best, value)
                rows.append({"time": artifact.get("updated_at_utc"), "value": best})
        return rows
    return [
        {"time": row.get("timestamp_utc"), "value": value}
        for row in snapshot.history
        if (value := number(mapping(row.get(section)).get(field))) is not None
    ]


def loss_rows(
    snapshot: BootstrapDashboardSnapshot, evaluator: str | None = None
) -> list[dict[str, Any]]:
    """Prefer completed pipeline training metrics, with generation-time provenance."""
    source = [
        (row.get("updated_at_utc"), mapping(row.get("evaluator_results")))
        for row in snapshot.training_history
    ]
    if not source:
        source = [
            (row.get("timestamp_utc"), mapping(row.get("evaluators")))
            for row in snapshot.history
        ]
    rows = []
    for time, metrics in source:
        for name, result in metrics.items():
            if evaluator is not None and name != evaluator:
                continue
            for field, label in [
                ("train_loss", "Train"),
                ("validation_loss", "Validation"),
            ]:
                value = number(mapping(result).get(field))
                if value is not None:
                    rows.append({
                        "time": time,
                        "value": value,
                        "series": label if evaluator else name + " · " + label,
                    })
    return rows


def render(st: Any, snapshot: BootstrapDashboardSnapshot) -> None:
    """Render the lightweight default page without checkpoint or weight loading."""
    page_header(
        st, snapshot, "Bootstrap", "A clear view of search, records and learning."
    )
    metric_row(
        st,
        [
            ("Generation", snapshot.generation),
            ("Tree nodes", snapshot.tree_nodes),
            ("Active evaluator", snapshot.evaluator_name),
            ("Dataset rows", snapshot.dataset_rows),
            ("Certified · points", snapshot.record.get("current_best_total_points")),
            ("Frontier · points", snapshot.frontier.get("current_best_total_points")),
        ],
    )
    st.caption(snapshot.status_detail)
    worker_strip(st, snapshot)
    left, middle, right = st.columns(3)
    with left:
        trend(
            st,
            title="Certified record progression",
            rows=history_rows(snapshot, "record", "current_best_total_points"),
            color=PALETTE["amber"],
            key="overview-record",
        )
    with middle:
        trend(
            st,
            title="Tree growth",
            rows=history_rows(snapshot, "tree", "num_nodes"),
            color=PALETTE["green"],
            key="overview-tree",
        )
    losses = loss_rows(
        snapshot, str(snapshot.evaluator_name) if snapshot.evaluator_name else None
    )
    with right:
        trend(
            st,
            title="Evaluator learning · saved loss",
            rows=losses,
            color=PALETTE["green"],
            key="overview-loss",
        )
    st.caption(
        "Certified results and frontier estimates are distinct. Saved training loss does not establish gameplay strength."
    )

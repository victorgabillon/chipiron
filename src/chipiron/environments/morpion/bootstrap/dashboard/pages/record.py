"""Certified state presentation, kept explicitly separate from frontier estimates."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from chipiron.environments.morpion.bootstrap.dashboard.theme import (
    PALETTE,
    metric_row,
    page_header,
)
from chipiron.environments.morpion.bootstrap.dashboard.view_model import display

from .overview import history_rows, trend

if TYPE_CHECKING:
    from chipiron.environments.morpion.bootstrap.dashboard.view_model import (
        BootstrapDashboardSnapshot,
    )


def render(st: Any, snapshot: BootstrapDashboardSnapshot) -> None:
    """Load the certified board only on its own page, retaining all source details."""
    from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
        MorpionBootstrapPaths,
    )
    from chipiron.environments.morpion.bootstrap.dashboard.data_cache import (
        cached_build_current_certified_record_board_view,
        cached_certified_record_board_freshness_tokens,
    )

    page_header(
        st,
        snapshot,
        "Certified record",
        "An exact or terminal result, with its persisted provenance.",
    )
    paths = MorpionBootstrapPaths.from_work_dir(snapshot.work_dir)
    board = None
    try:
        with st.spinner("Reading the certified state…"):
            board = cached_build_current_certified_record_board_view(
                str(snapshot.work_dir),
                cached_certified_record_board_freshness_tokens(paths),
            )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        st.warning(f"Certified state temporarily unavailable: {exc}")
    left, right = st.columns((2, 1))
    with left, st.container(border=True):
        if board is None:
            st.markdown("**No certified state yet**")
            st.caption(
                "The board appears once an exact or terminal result is persisted. Frontier estimates remain separate."
            )
        else:
            # Render the SVG as a native image rather than embedding XML in HTML.
            st.image(board.board_svg, width="stretch")
    with right:
        record = snapshot.record
        st.metric(
            "Certified · total points",
            display(
                board.total_points if board else record.get("current_best_total_points")
            ),
        )
        st.metric(
            "Moves since start",
            display(
                board.moves_since_start
                if board
                else record.get("current_best_moves_since_start")
            ),
        )
        st.write(
            "Exact:",
            display(board.is_exact if board else record.get("current_best_is_exact")),
        )
        st.write(
            "Terminal:",
            display(
                board.is_terminal if board else record.get("current_best_is_terminal")
            ),
        )
        st.caption(
            "Source: "
            + display(board.source if board else record.get("current_best_source"))
        )
    metric_row(
        st,
        [
            ("Certified · points", snapshot.record.get("current_best_total_points")),
            (
                "Frontier estimate · points",
                snapshot.frontier.get("current_best_total_points"),
            ),
        ],
    )
    trend(
        st,
        title="Certified record history",
        rows=history_rows(snapshot, "record", "current_best_total_points"),
        color=PALETTE["amber"],
        key="record-history",
    )
    with st.expander("Advanced record details"):
        st.write("Certified record metadata")
        st.json(snapshot.record)
        st.write("Frontier metadata (not a certified record)")
        st.json(snapshot.frontier)
        if board is not None and board.board_text:
            st.code(board.board_text)
        st.caption(
            "A complete move path is shown only if present in the stored source metadata."
        )

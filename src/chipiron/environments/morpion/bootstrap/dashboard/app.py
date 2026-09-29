"""Compose a lazy five-page Streamlit operator interface."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .navigation import PAGES, REFRESH_OPTIONS, refresh_seconds, selected_page
from .theme import apply_theme
from .view_model import load_dashboard_snapshot

if TYPE_CHECKING:
    from pathlib import Path


class MissingStreamlitDashboardDependencyError(RuntimeError):
    """Explain the opt-in dashboard extra when it is unavailable."""

    def __init__(self) -> None:
        """Give the canonical install command."""
        super().__init__(
            "Dashboard support is not installed. Install: pip install 'chipiron[dashboard]'"
        )


def _get_streamlit() -> Any:
    """Keep the dashboard extra out of core imports."""
    try:
        return import_module("streamlit")
    except ModuleNotFoundError as exc:
        raise MissingStreamlitDashboardDependencyError from exc


def run_dashboard_app(work_dir: Path) -> None:
    """Route only the active page; auto-refresh never evaluates hidden pages."""
    st = _get_streamlit()
    st.set_page_config(
        page_title="Morpion · Bootstrap",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_theme(st)
    with st.sidebar:
        st.markdown("### Morpion")
        st.caption("BOOTSTRAP OPERATOR")
        state_key = "bootstrap_operator_page"
        st.session_state[state_key] = selected_page(st.session_state.get(state_key))
        page = st.radio(
            "Workspace views", PAGES, key=state_key, label_visibility="collapsed"
        )
        st.divider()
        st.caption("WORKSPACE")
        st.write(work_dir.name)
        with st.expander("Full path"):
            st.code(str(work_dir))
        if st.button("Refresh", width="stretch"):
            st.rerun()
        choice = st.selectbox(
            "Auto refresh",
            REFRESH_OPTIONS,
            index=2 if page == "Overview" else 0,
            key=f"operator_refresh_{page}",
        )
        if page == "Tree":
            st.caption(
                "Manual by default. Refresh reads a newer checkpoint; navigation reuses the current cache."
            )
        st.divider()
        st.caption(
            "Observations come from saved artifacts. Unknown is shown when evidence is unavailable."
        )
    renderer = import_module(f"{__package__}.pages.{page.lower()}").render

    def render_page() -> None:
        snapshot = load_dashboard_snapshot(work_dir)
        renderer(st, snapshot)

    st.fragment(render_page, run_every=refresh_seconds(page, choice))()


__all__ = ["run_dashboard_app"]

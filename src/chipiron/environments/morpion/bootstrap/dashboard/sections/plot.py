"""Dashboard plot rendering section helpers."""

from __future__ import annotations

from typing import Any

from matplotlib import pyplot as plt

__all__ = ["render_plot"]


def render_plot(st: Any, build_plot: Any) -> None:
    """Render one existing matplotlib plot helper into Streamlit."""
    build_plot()
    figure = plt.gcf()
    st.pyplot(figure, clear_figure=True)
    plt.close(figure)

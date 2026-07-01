"""Stable Morpion bootstrap dashboard APIs."""

from __future__ import annotations

from typing import Any

__all__ = ["run_dashboard_app"]


def __getattr__(name: str) -> Any:
    """Load the Streamlit app entry point only when callers request it."""
    if name == "run_dashboard_app":
        from .app import run_dashboard_app

        globals()[name] = run_dashboard_app
        return run_dashboard_app
    raise AttributeError(name)

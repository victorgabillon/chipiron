"""Import-time bootstrap for Qt runtime environment configuration."""

from __future__ import annotations

from .qt_runtime_env import configure_qt_runtime_env_for_gui

configure_qt_runtime_env_for_gui()

QT_RUNTIME_BOOTSTRAPPED: bool = True

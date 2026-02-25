"""Runtime environment helpers for Qt-based GUI processes."""

from __future__ import annotations

import os
import platform


def _is_truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def configure_qt_runtime_env_for_gui() -> None:
    """Configure environment variables to reduce Qt AT-SPI log spam on Linux.

    By default, this disables the AT-SPI bridge for Chipiron GUI processes.
    Users can opt out by setting CHIPIRON_QT_DISABLE_ATSPI=0.
    """
    if platform.system() != "Linux":
        return

    if not _is_truthy(os.environ.get("CHIPIRON_QT_DISABLE_ATSPI", "1")):
        return

    # Disable Qt accessibility integration itself.
    os.environ["QT_ACCESSIBILITY"] = "0"
    os.environ["QT_LINUX_ACCESSIBILITY_ALWAYS_ON"] = "0"

    # Disable the Linux desktop AT-SPI bridge.
    os.environ["NO_AT_BRIDGE"] = "1"

    rules_to_add = [
        "qt.accessibility.atspi=false",
        "qt.accessibility.atspi.*=false",
    ]
    existing = os.environ.get("QT_LOGGING_RULES")
    if not existing:
        os.environ["QT_LOGGING_RULES"] = ";".join(rules_to_add)
        return

    parts = [part.strip() for part in existing.split(";") if part.strip()]
    for rule in rules_to_add:
        if rule not in parts:
            parts.append(rule)
    if parts:
        os.environ["QT_LOGGING_RULES"] = ";".join(parts)

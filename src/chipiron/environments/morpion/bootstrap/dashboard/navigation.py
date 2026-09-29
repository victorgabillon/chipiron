"""Stable page identities and conservative refresh defaults."""

PAGES = ("Overview", "Record", "Tree", "Evaluator", "Operations")
REFRESH_OPTIONS = ("Off", "5s", "15s", "30s")


def selected_page(value: object) -> str:
    """Normalize stale session navigation to the lightweight default view."""
    return value if isinstance(value, str) and value in PAGES else "Overview"


def refresh_seconds(page: str, choice: str | None = None) -> int | None:
    """Refresh overview every 15s; keep expensive pages manual unless requested."""
    resolved = choice or ("15s" if page == "Overview" else "Off")
    return int(resolved[:-1]) if resolved in REFRESH_OPTIONS[1:] else None

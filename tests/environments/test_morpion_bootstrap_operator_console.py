"""Tests for Morpion operator terminal rendering helpers."""

from __future__ import annotations

import builtins
import re
from typing import TYPE_CHECKING

from chipiron.environments.morpion.bootstrap import operator_console

if TYPE_CHECKING:
    import pytest

_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def test_operator_console_falls_back_when_rich_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Operator output should stay usable when Rich is unavailable."""
    real_import = builtins.__import__

    def _blocked_import(
        name: str,
        globals_: object | None = None,
        locals_: object | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name.startswith("rich"):
            raise ImportError("blocked")
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    rendered = operator_console.render_operator_panel(
        "Growth",
        ["ACTIVE EVALUATOR: mlp_41"],
        force_rich=True,
    )

    assert not operator_console.rich_available()
    assert "[Growth]" in rendered
    assert "ACTIVE EVALUATOR: mlp_41" in rendered
    assert _ANSI_RE.search(rendered) is None


def test_operator_console_plain_fallback_has_no_ansi() -> None:
    """Plain fallback output must be safe for logs and dumb terminals."""
    rendered = operator_console.render_operator_panel(
        "Status",
        ["generation=38"],
        force_plain=True,
    )

    assert rendered == "[Status]\n  generation=38"
    assert _ANSI_RE.search(rendered) is None


def test_operator_console_rich_rendering_includes_title_and_text() -> None:
    """Rich rendering should still expose readable title/text in captured output."""
    rendered = operator_console.render_operator_panel(
        "Morpion Training Recap",
        ["selected mlp_41"],
        force_rich=True,
    )

    assert "Morpion Training Recap" in rendered
    assert "selected mlp_41" in rendered

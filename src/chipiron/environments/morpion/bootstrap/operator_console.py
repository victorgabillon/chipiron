"""Operator-facing terminal rendering helpers for Morpion bootstrap workers."""

from __future__ import annotations

import argparse
import os
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence


def rich_available() -> bool:
    """Return whether Rich can be imported in the current environment."""
    return _rich_modules() is not None


def use_rich_output(
    *,
    force_plain: bool = False,
    force_rich: bool = False,
    stream: object | None = None,
    env: dict[str, str] | None = None,
) -> bool:
    """Return whether operator output should use Rich rendering."""
    if force_plain:
        return False
    if not rich_available():
        return False

    resolved_env = os.environ if env is None else env
    if "NO_COLOR" in resolved_env:
        return False
    if resolved_env.get("TERM") == "dumb":
        return False

    mode = resolved_env.get("MORPION_OPERATOR_RICH", "auto").strip().lower()
    if force_rich:
        mode = "always"
    if mode == "never":
        return False
    if mode == "always":
        return True

    output_stream = sys.stdout if stream is None else stream
    isatty = getattr(output_stream, "isatty", None)
    return bool(callable(isatty) and isatty())


def render_operator_panel(
    title: str,
    lines: Sequence[str],
    *,
    style: str = "cyan",
    force_plain: bool = False,
    force_rich: bool = False,
) -> str:
    """Render a compact operator panel as text."""
    if not use_rich_output(force_plain=force_plain, force_rich=force_rich):
        return _render_plain_panel(title, lines)

    modules = _rich_modules()
    if modules is None:
        return _render_plain_panel(title, lines)
    console_cls, panel_cls, text_cls = modules
    console = console_cls(record=True, force_terminal=True, color_system=None)
    console.print(
        panel_cls(
            text_cls("\n".join(lines)),
            title=title,
            border_style=style,
        )
    )
    return str(console.export_text(styles=False)).rstrip()


def print_operator_panel(
    title: str,
    lines: Sequence[str],
    *,
    style: str = "cyan",
    force_plain: bool = False,
    force_rich: bool = False,
) -> None:
    """Print a compact operator panel to stdout."""
    if not use_rich_output(force_plain=force_plain, force_rich=force_rich):
        print(_render_plain_panel(title, lines))
        return

    modules = _rich_modules()
    if modules is None:
        print(_render_plain_panel(title, lines))
        return
    console_cls, panel_cls, text_cls = modules
    console = console_cls(force_terminal=True)
    console.print(
        panel_cls(
            text_cls("\n".join(lines)),
            title=title,
            border_style=style,
        )
    )


def _render_plain_panel(title: str, lines: Sequence[str]) -> str:
    """Render a dependency-free plain text panel."""
    body = "\n".join(f"  {line}" for line in lines)
    return f"[{title}]\n{body}" if body else f"[{title}]"


def _rich_modules() -> tuple[type[Any], type[Any], type[Any]] | None:
    """Return Rich classes when importable."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text
    except ImportError:
        return None
    return Console, Panel, Text


def main(argv: list[str] | None = None) -> int:
    """Run a tiny diagnostic CLI for operator console rendering."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("title", nargs="?", default="Morpion Operator Console")
    parser.add_argument("lines", nargs="*")
    parser.add_argument("--rich", action="store_true", dest="force_rich")
    parser.add_argument("--no-rich", action="store_true", dest="force_plain")
    args = parser.parse_args(argv)

    lines = args.lines or [f"rich_available={rich_available()}"]
    print_operator_panel(
        args.title,
        lines,
        force_plain=args.force_plain,
        force_rich=args.force_rich,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

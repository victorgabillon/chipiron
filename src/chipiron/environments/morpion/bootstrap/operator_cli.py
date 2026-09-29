"""Small installed operator interface; never synthesizes a scientific experiment."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import sys
import tempfile
from pathlib import Path

from platformdirs import user_config_path


class BootstrapOperatorError(ValueError):
    """An actionable workspace or operator configuration error."""


def current_run_config_path() -> Path:
    """Locate user preferences independently of the experiment artifacts."""
    return user_config_path("chipiron") / "bootstrap_operator.json"


def looks_like_workspace(path: Path) -> bool:
    """Recognize explicit bootstrap metadata without scanning parent directories."""
    return path.is_dir() and (path / "bootstrap_config.json").is_file()


def resolve_work_dir(explicit: str | Path | None = None) -> Path:
    """Resolve explicit, environment, saved, then clearly identified cwd workspace."""
    selected = explicit or os.environ.get("MORPION_WORK_DIR")
    config_path = current_run_config_path()
    if selected is None and config_path.exists():
        message = "Current-run config is invalid. Select a run with chipiron-bootstrap use PATH."
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise BootstrapOperatorError(message) from exc
        if (
            not isinstance(config, dict)
            or config.get("version") != 1
            or not isinstance(config.get("work_dir"), str)
            or not Path(config["work_dir"]).is_absolute()
        ):
            raise BootstrapOperatorError(message)
        selected = config["work_dir"]
    if selected is None and looks_like_workspace(Path.cwd()):
        selected = Path.cwd()
    if selected is None:
        message = "Choose a run first: chipiron-bootstrap use /path/to/run"
        raise BootstrapOperatorError(message)
    path = Path(selected).expanduser().resolve()
    if not looks_like_workspace(path):
        message = (
            f"Not a bootstrap workspace: {path} (bootstrap_config.json is missing)."
        )
        raise BootstrapOperatorError(message)
    return path


def save_current_run(path: Path) -> Path:
    """Atomically save a versioned absolute selection outside the run directory."""
    resolved = resolve_work_dir(path)
    destination = current_run_config_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump({"version": 1, "work_dir": str(resolved)}, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return resolved


def dashboard_command(
    work_dir: Path, *, port: int = 8501, no_browser: bool = False
) -> tuple[str, ...]:
    """Launch the installed entry script with this interpreter on loopback only."""
    return (
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(Path(__file__).with_name("dashboard_streamlit_entry.py")),
        "--server.address=127.0.0.1",
        f"--server.port={port}",
        f"--server.headless={'true' if no_browser else 'false'}",
        "--browser.gatherUsageStats=false",
        "--theme.base=light",
        "--theme.primaryColor=#245b52",
        "--theme.backgroundColor=#f7f8f5",
        "--theme.secondaryBackgroundColor=#eef1ed",
        "--theme.textColor=#263631",
        "--",
        "--work-dir",
        str(work_dir),
    )


def prepared_launcher_command(work_dir: Path) -> tuple[str, ...]:
    """Select only the workspace-owned canonical launcher; fail closed otherwise."""
    launcher = work_dir / "run_bootstrap.sh"
    if not launcher.is_file():
        message = (
            "No prepared run_bootstrap.sh exists in this workspace. "
            "Prepare a workspace or use the advanced low-level launcher."
        )
        raise BootstrapOperatorError(message)
    return ("bash", str(launcher))


def build_parser() -> argparse.ArgumentParser:
    """Expose only the six common operator actions."""
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--work-dir", type=Path, default=argparse.SUPPRESS)
    parser = argparse.ArgumentParser(
        prog="chipiron-bootstrap",
        description="Open your current Morpion dashboard (the default), inspect or launch a prepared run.",
        parents=[shared],
    )
    parser.set_defaults(work_dir=None)
    commands = parser.add_subparsers(dest="command")
    dashboard = commands.add_parser(
        "dashboard", parents=[shared], help="Open dashboard"
    )
    dashboard.add_argument("--port", type=int, default=8501)
    dashboard.add_argument("--no-browser", action="store_true")
    commands.add_parser(
        "status", parents=[shared], help="Print a quick read-only summary"
    )
    commands.add_parser("current", parents=[shared], help="Print the selected run path")
    use = commands.add_parser("use", help="Select a run once")
    use.add_argument("path", type=Path)
    run = commands.add_parser(
        "run", parents=[shared], help="Launch the prepared workspace script"
    )
    run.add_argument(
        "--dry-run", action="store_true", help="Print command without launching"
    )
    run.add_argument("--yes", action="store_true", help="Confirm a deliberate launch")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run an operator action; exec preserves foreground exit and signal semantics."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "use":
            print(save_current_run(args.path))
            return 0
        work_dir = resolve_work_dir(args.work_dir)
        if args.command == "current":
            print(work_dir)
            return 0
        if args.command == "status":
            from .dashboard.view_model import load_dashboard_snapshot, render_status

            print(render_status(load_dashboard_snapshot(work_dir)))
            return 0
        if args.command == "run":
            return _run_prepared(work_dir, dry_run=args.dry_run, confirmed=args.yes)
        if importlib.util.find_spec("streamlit") is None:
            print(
                "Dashboard support is not installed.\n\nInstall:\n    pip install 'chipiron[dashboard]'",
                file=sys.stderr,
            )
            return 2
        port = getattr(args, "port", 8501)
        if not 1 <= port <= 65535:
            parser.error("--port must be between 1 and 65535")
        command = dashboard_command(
            work_dir, port=port, no_browser=getattr(args, "no_browser", False)
        )
        print(f"Dashboard: http://127.0.0.1:{port}\nRun: {work_dir.name}", flush=True)
        return os.execv(sys.executable, command)
    except (BootstrapOperatorError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


def _run_prepared(work_dir: Path, *, dry_run: bool, confirmed: bool) -> int:
    """Preview and confirm the existing launch protocol, without reimplementing it."""
    from .dashboard.view_model import load_dashboard_snapshot

    command = prepared_launcher_command(work_dir)
    snapshot = load_dashboard_snapshot(work_dir)
    print(f"Run: {work_dir.name}\nLauncher: {command[1]}")
    print(
        "Scientific configuration: "
        + (
            "from prepared workspace manifest"
            if snapshot.manifest
            else "manifest unavailable; consult workspace launcher"
        )
    )
    print(f"Active runtime maximum: {snapshot.active_runtime_label}")
    print(f"Command: {shlex.join(command)}", flush=True)
    if dry_run:
        print("Dry run: nothing launched.")
        return 0
    if not confirmed:
        if not sys.stdin.isatty():
            print(
                "Confirmation required. Use --yes for a deliberate non-interactive launch.",
                file=sys.stderr,
            )
            return 2
        try:
            confirmed = (
                input("Launch this prepared experiment? [y/N] ").strip().lower() == "y"
            )
        except (EOFError, KeyboardInterrupt):
            confirmed = False
    if not confirmed:
        print("Launch cancelled.")
        return 0
    os.chdir(work_dir)
    return os.execvp(command[0], command)


if __name__ == "__main__":
    raise SystemExit(main())

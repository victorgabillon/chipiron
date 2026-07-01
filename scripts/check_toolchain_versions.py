"""Check that quality-tool versions have one source of truth."""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_EXACT_PINS = {
    "lint": ("ruff", "pylint"),
    "typecheck": ("mypy", "pyright"),
    "dev": ("tox", "black", "build", "pre-commit"),
}
PINNED_DEP_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9_.-]+)(?:\[[^]]+\])?==(?P<version>[^;\s]+)"
)


def normalized_name(name: str) -> str:
    """Normalize a Python distribution name for comparisons."""
    return name.replace("_", "-").lower()


def read_text(path: str) -> str:
    """Read a repository file as UTF-8 text."""
    return (ROOT / path).read_text(encoding="utf-8")


def pinned_versions(dependencies: list[str]) -> dict[str, str]:
    """Return exact package pins from a dependency list."""
    versions: dict[str, str] = {}
    for dependency in dependencies:
        match = PINNED_DEP_RE.match(dependency)
        if match is None:
            continue
        versions[normalized_name(match.group("name"))] = match.group("version")
    return versions


def project_optional_deps(
    pyproject: dict[str, Any], errors: list[str]
) -> dict[str, Any]:
    """Return pyproject optional dependencies, or record a structural error."""
    project = pyproject.get("project", {})
    if not isinstance(project, dict):
        errors.append("pyproject.toml is missing [project].")
        return {}

    optional_deps = project.get("optional-dependencies", {})
    if not isinstance(optional_deps, dict):
        errors.append("pyproject.toml is missing [project.optional-dependencies].")
        return {}

    return optional_deps


def tool_pins(pyproject: dict[str, Any], errors: list[str]) -> dict[str, str]:
    """Collect required tool pins from pyproject.toml."""
    optional_deps = project_optional_deps(pyproject, errors)
    pins: dict[str, str] = {}

    for extra_name, package_names in REQUIRED_EXACT_PINS.items():
        raw_deps = optional_deps.get(extra_name, [])
        if not isinstance(raw_deps, list) or not all(
            isinstance(item, str) for item in raw_deps
        ):
            errors.append(f"Extra `{extra_name}` must be a list of strings.")
            continue

        versions = pinned_versions(raw_deps)
        for package_name in package_names:
            version = versions.get(package_name)
            if version is None:
                errors.append(
                    f"`{package_name}` in extra `{extra_name}` must be pinned "
                    "with `==`."
                )
                continue
            pins[package_name] = version

    return pins


def tox_config(pyproject: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    """Return tox configuration from pyproject.toml."""
    tool = pyproject.get("tool", {})
    tox = tool.get("tox", {}) if isinstance(tool, dict) else {}
    if not isinstance(tox, dict):
        errors.append("pyproject.toml is missing [tool.tox].")
        return {}
    return tox


def check_ruff_config(pyproject: dict[str, Any], errors: list[str]) -> None:
    """Check that Ruff does not auto-fix through default configuration."""
    tool = pyproject.get("tool", {})
    ruff_config = tool.get("ruff", {}) if isinstance(tool, dict) else {}
    if not isinstance(ruff_config, dict):
        errors.append("pyproject.toml is missing [tool.ruff].")
        return
    if ruff_config.get("fix") is not False:
        errors.append("Set [tool.ruff].fix = false; use explicit --fix locally.")


def check_tox_config(
    pyproject: dict[str, Any], pins: dict[str, str], errors: list[str]
) -> None:
    """Check tox uses the pyproject tool pins."""
    tox = tox_config(pyproject, errors)
    if not tox:
        return

    tox_version = pins.get("tox")
    if tox.get("min_version") != tox_version:
        errors.append(
            f"[tool.tox].min_version must match tox=={tox_version} from dev extra."
        )

    expected_envs = ["tooling", "py313", "lint", "typecheck"]
    if tox.get("env_list") != expected_envs:
        errors.append(f"[tool.tox].env_list must be {expected_envs!r}.")

    tox_env = tox.get("env", {})
    if not isinstance(tox_env, dict):
        errors.append("pyproject.toml is missing [tool.tox.env].")
        return

    py313 = tox_env.get("py313", {})
    py313_deps = py313.get("deps", []) if isinstance(py313, dict) else []
    build_pin = f"build=={pins.get('build')}"
    if build_pin not in py313_deps:
        errors.append(f"[tool.tox.env.py313].deps must include {build_pin!r}.")

    tooling = tox_env.get("tooling", {})
    if not isinstance(tooling, dict) or tooling.get("package") != "skip":
        errors.append('[tool.tox.env.tooling] must use package = "skip".')
    tooling_commands = tooling.get("commands", []) if isinstance(tooling, dict) else []
    if tooling_commands != [["python", "scripts/check_toolchain_versions.py"]]:
        errors.append("[tool.tox.env.tooling] must run the drift guard only.")

    lint = tox_env.get("lint", {})
    lint_commands = lint.get("commands", []) if isinstance(lint, dict) else []
    required_lint_commands = [
        ["python", "-m", "ruff", "format", "--check", "src/chipiron"],
        ["python", "-m", "ruff", "check", "src/chipiron"],
        ["python", "-m", "pylint", "src/chipiron"],
    ]
    errors.extend(
        f"[tool.tox.env.lint] is missing command {command!r}."
        for command in required_lint_commands
        if command not in lint_commands
    )

    typecheck = tox_env.get("typecheck", {})
    typecheck_commands = (
        typecheck.get("commands", []) if isinstance(typecheck, dict) else []
    )
    errors.extend(
        f"[tool.tox.env.typecheck] must run `python -m {command_name}`."
        for command_name in ("mypy", "pyright")
        if not any(
            command[:3] == ["python", "-m", command_name]
            for command in typecheck_commands
        )
    )


def check_pre_commit(pins: dict[str, str], errors: list[str]) -> None:
    """Check pre-commit uses the pinned local Ruff entry points."""
    config = read_text(".pre-commit-config.yaml")
    pre_commit_version = pins.get("pre-commit")
    if f'minimum_pre_commit_version: "{pre_commit_version}"' not in config:
        errors.append(
            ".pre-commit-config.yaml must declare "
            f'minimum_pre_commit_version: "{pre_commit_version}".'
        )
    if "ruff-pre-commit" in config:
        errors.append("pre-commit must not pin a separate ruff-pre-commit revision.")
    if "repo: local" not in config:
        errors.append("pre-commit hooks must be local and use the project toolchain.")
    if "python scripts/check_toolchain_versions.py" not in config:
        errors.append(
            "pre-commit must run `python scripts/check_toolchain_versions.py`."
        )
    if "python -m ruff check --no-fix" not in config:
        errors.append("pre-commit must run `python -m ruff check --no-fix`.")
    if "python -m ruff format --check" not in config:
        errors.append("pre-commit must run `python -m ruff format --check`.")
    if re.search(r"(?<!no-)--fix\b", config):
        errors.append("pre-commit must not auto-fix by default.")


def check_workflows(pins: dict[str, str], errors: list[str]) -> None:
    """Check GitHub workflows install the pyproject tool pins."""
    for workflow_path in (".github/workflows/ci.yaml", ".github/workflows/release.yml"):
        workflow = read_text(workflow_path)
        for package_name in ("tox", "build"):
            package_pin = f"{package_name}=={pins.get(package_name)}"
            if package_pin not in workflow:
                errors.append(f"{workflow_path} must install {package_pin}.")


def check_dependabot(pins: dict[str, str], errors: list[str]) -> None:
    """Check Dependabot is configured to update the pinned quality tools."""
    dependabot = read_text(".github/dependabot.yml")
    if "quality-toolchain:" not in dependabot:
        errors.append("Dependabot must group pinned quality-toolchain updates.")
    errors.extend(
        f"Dependabot quality-toolchain group must include {package_name!r}."
        for package_name in sorted(pins)
        if f'- "{package_name}"' not in dependabot
    )


def main() -> int:
    """Run the toolchain drift checks."""
    pyproject = tomllib.loads(read_text("pyproject.toml"))
    errors: list[str] = []
    pins = tool_pins(pyproject, errors)
    check_ruff_config(pyproject, errors)
    check_tox_config(pyproject, pins, errors)
    check_pre_commit(pins, errors)
    check_workflows(pins, errors)
    check_dependabot(pins, errors)

    if errors:
        print("Toolchain version drift detected:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Toolchain versions are coherent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Report inherited analyzer debt and fail on new diagnostics or baseline growth."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import tomllib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

TOOLS = ("mypy", "pyright", "pylint")
FIELDS = ("tool", "file", "rule", "scope", "source", "message")


class RatchetError(RuntimeError):
    """The analysis or baseline cannot be trusted."""


def normalized_message(message: str, root: Path) -> str:
    """Remove checkout prefixes and Pylint duplicate-code line ranges."""
    message = message.replace(str(root), "<checkout>")
    message = re.sub(r":\[\d+:\d+\]", ":[lines]", message)
    return " ".join(message.split())


def location(root: Path, filename: str, line: int) -> tuple[str, str, str]:
    """Use lexical scope and source text so unrelated inserted lines do not add debt."""
    path = Path(filename)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    relative = path.relative_to(root).as_posix()
    source = path.read_text(encoding="utf-8")
    scopes = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.lineno <= line <= (node.end_lineno or node.lineno)
    ]
    scopes.sort(key=lambda node: node.lineno)
    scope = ".".join(node.name for node in scopes) or "<module>"
    lines = source.splitlines()
    text = " ".join(lines[line - 1].split()) if 0 < line <= len(lines) else ""
    return relative, scope, text


def parse_diagnostics(tool: str, output: str, root: Path) -> list[dict[str, Any]]:
    """Normalize machine-readable reports while preserving duplicate multiplicities."""
    if tool == "mypy":
        records = []
        for line in output.splitlines():
            if not line.strip():
                continue
            if not line.lstrip().startswith("{") and re.fullmatch(
                r".+: note: .+", line
            ):
                print(f"mypy configuration note: {line}")
                continue
            records.append(json.loads(line))
    elif tool == "pyright":
        records = json.loads(output)["generalDiagnostics"]
    else:
        records = json.loads(output)
    result: list[dict[str, Any]] = []
    for record in records:
        severity = record.get("severity", record.get("type", "error"))
        if severity in {"note", "information"}:
            continue
        if tool == "pyright":
            filename = record["file"]
            line = record["range"]["start"]["line"] + 1
            rule = record.get("rule", severity)
        else:
            filename = record["file"] if tool == "mypy" else record["path"]
            line = record["line"]
            rule = record.get("code") if tool == "mypy" else record["message-id"]
        path, scope, source = location(root, filename, line)
        result.append({
            "tool": tool,
            "file": path,
            "rule": rule or severity,
            "scope": scope,
            "source": source,
            "message": normalized_message(record["message"], root),
            "line": line,
        })
    return result


def identities(records: list[dict[str, Any]]) -> Counter[str]:
    """Count stable identities; duplicate occurrences must not disappear in a set."""
    result: Counter[str] = Counter()
    for record in records:
        identity = {field: record[field] for field in FIELDS}
        if record["tool"] == "pylint" and record["rule"] == "R0801":
            # Pylint attaches project-wide similarities to the last visited module.
            # The message already identifies the actual files and duplicated text.
            identity.update(file="<project>", scope="<project>", source="")
        result[json.dumps(identity, sort_keys=True)] += 1
    return result


def compare(
    baseline: list[dict[str, Any]], current: list[dict[str, Any]]
) -> tuple[Counter[str], Counter[str]]:
    """Return newly introduced and removed diagnostic occurrences."""
    old, new = identities(baseline), identities(current)
    return new - old, old - new


def read_manifest(text: str) -> dict[str, Any]:
    """Reject malformed baselines instead of treating them as an empty report."""
    manifest = json.loads(text)
    if manifest.get("schema") != 1 or not isinstance(manifest.get("diagnostics"), list):
        message = "Invalid static-analysis baseline schema."
        raise RatchetError(message)
    identities(manifest["diagnostics"])
    return manifest


def git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Inspect baseline ancestry without modifying Git state."""
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def check_baseline_growth(root: Path, path: Path, manifest: dict[str, Any]) -> None:
    """Enforce the first committed ceiling and the PR base's possibly smaller debt."""
    relative = path.relative_to(root).as_posix()
    history = git(root, "log", "--diff-filter=A", "--format=%H", "--", relative)
    revisions = history.stdout.splitlines()
    if not revisions:
        if os.environ.get("CI"):
            message = (
                "Baseline history is missing; CI requires checkout fetch-depth: 0."
            )
            raise RatchetError(message)
        print("INITIAL BASELINE: must be committed and reviewed before CI/publication.")
        return
    references = [revisions[-1]]
    reference = os.environ.get("STATIC_ANALYSIS_BASE_REF")
    if reference and set(reference) != {"0"}:
        references.append(reference)
    for revision in references:
        previous = git(root, "show", f"{revision}:{relative}")
        if previous.returncode:
            # The first introduction has no baseline in its PR base.
            exists = git(root, "cat-file", "-e", f"{revision}^{{commit}}")
            if exists.returncode:
                message = f"Cannot inspect baseline reference {revision}."
                raise RatchetError(message)
            print(
                f"Baseline absent at {revision}; first committed ceiling still applies."
            )
            continue
        prior = read_manifest(previous.stdout)
        added, _removed = compare(prior["diagnostics"], manifest["diagnostics"])
        if added:
            message = f"Baseline growth is forbidden: {sum(added.values())} added occurrences versus {revision}."
            raise RatchetError(message)


def analyze(
    root: Path, source: str, output: Path, *, mypy_strict: bool, workers: int = 3
) -> list[dict[str, Any]]:
    """Run all strict analyzers; tool crashes and invalid reports fail closed."""
    commands = {
        "mypy": [
            *(["--strict"] if mypy_strict else []),
            "--output",
            "json",
            "--no-error-summary",
            source,
        ],
        "pyright": ["--pythonpath", sys.executable, "--outputjson", source],
        "pylint": ["--output-format=json", "--score=n", source],
    }
    output.mkdir(parents=True, exist_ok=True)
    diagnostics: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = {
            tool: executor.submit(
                subprocess.run,
                [sys.executable, "-m", tool, *args],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
                timeout=600,
            )
            for tool, args in commands.items()
        }
        for tool in commands:
            result = results[tool].result()
            (output / f"{tool}.json").write_text(result.stdout, encoding="utf-8")
            (output / f"{tool}.stderr.txt").write_text(result.stderr, encoding="utf-8")
            accepted = {0, 1} if tool != "pylint" else set(range(32))
            if result.returncode not in accepted:
                message = (
                    f"{tool} failed with exit {result.returncode}: {result.stderr}"
                )
                raise RatchetError(message)
            records = parse_diagnostics(tool, result.stdout, root)
            if result.returncode and not records:
                message = f"{tool} failed without a usable diagnostic report."
                raise RatchetError(message)
            diagnostics.extend(records)
            for row in records:
                print(
                    f"{tool}: {row['file']}:{row['line']}: {row['rule']}: {row['message']}"
                )
    return sorted(diagnostics, key=lambda row: tuple(str(row[key]) for key in FIELDS))


def run(args: argparse.Namespace) -> int:
    """Compare visible current debt with an explicitly reviewed immutable ceiling."""
    root = Path(__file__).resolve().parents[1]
    baseline = (root / args.baseline).resolve()
    versions = {tool: importlib.metadata.version(tool) for tool in TOOLS}
    configuration = tomllib.loads((root / "pyproject.toml").read_text())["tool"]
    policy = {
        "mypy_strict": args.mypy_strict,
        "configuration": {tool: configuration[tool] for tool in TOOLS},
    }
    policy_hash = hashlib.sha256(
        json.dumps(policy, sort_keys=True).encode()
    ).hexdigest()
    current = analyze(
        root,
        args.source,
        args.output.resolve(),
        mypy_strict=args.mypy_strict,
        workers=args.workers,
    )
    if args.record_baseline:
        if os.environ.get("CI") or baseline.exists():
            message = "Baseline recording is local-only and never overwrites an existing baseline."
            raise RatchetError(message)
        manifest = {
            "schema": 1,
            "source": args.source,
            "source_commit": git(root, "rev-parse", "HEAD").stdout.strip(),
            "tools": versions,
            "policy_sha256": policy_hash,
            "diagnostics": current,
        }
        baseline.parent.mkdir(parents=True, exist_ok=True)
        baseline.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Recorded {len(current)} diagnostics for explicit review: {baseline}")
        return 0
    manifest = read_manifest(baseline.read_text(encoding="utf-8"))
    if (
        manifest["source"] != args.source
        or manifest["tools"] != versions
        or manifest["policy_sha256"] != policy_hash
    ):
        message = "Analyzer source/tool versions/configuration differ from the reviewed baseline."
        raise RatchetError(message)
    check_baseline_growth(root, baseline, manifest)
    added, removed = compare(manifest["diagnostics"], current)
    for tool in TOOLS:
        old_rows = [row for row in manifest["diagnostics"] if row["tool"] == tool]
        new_rows = [row for row in current if row["tool"] == tool]
        new_items, removed_items = compare(old_rows, new_rows)
        print(
            f"{tool}: baseline={len(old_rows)} current={len(new_rows)} new={sum(new_items.values())} removed={sum(removed_items.values())}"
        )
    report = {
        "baseline": len(manifest["diagnostics"]),
        "current": len(current),
        "new": sum(added.values()),
        "removed": sum(removed.values()),
    }
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    if added:
        for identity, count in added.items():
            print(f"NEW ({count}): {identity}", file=sys.stderr)
    if removed:
        print(
            "Debt decreased: shrink the reviewed baseline in a follow-up; CI never updates it."
        )
    return int(bool(added))


def main() -> int:
    """Expose explicit local baseline recording and read-only CI comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="src/chipiron")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("docs/maintenance/static_analysis_baseline/diagnostics.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record-baseline", action="store_true")
    parser.add_argument("--mypy-strict", action="store_true")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    try:
        return run(args)
    except (
        RatchetError,
        ValueError,
        KeyError,
        OSError,
        subprocess.TimeoutExpired,
    ) as error:
        print(f"Static-analysis ratchet failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

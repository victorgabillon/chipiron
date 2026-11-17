"""Verify raw shard coverage against authenticated, complete installed sources."""

import json
import tempfile
from pathlib import Path
from typing import Any

from coverage import Coverage, CoverageData


def require(condition: bool, message: str) -> None:
    """Refuse coverage that cannot be reconstructed from its raw evidence."""
    if not condition:
        raise ValueError(message)


def report(coverage: Coverage, files: set[str]) -> dict[str, Any]:
    """Analyze explicit sources, including unimported namespace-package files."""
    with tempfile.TemporaryDirectory(prefix="chipiron-coverage-") as temporary:
        output = Path(temporary) / "coverage.json"
        coverage.json_report(
            morfs=sorted(files), outfile=str(output), ignore_errors=False
        )
        return dict(json.loads(output.read_text()))


def read_coverage(data: Path, receipt: Path) -> tuple[Coverage, dict[str, Any]]:
    """Reconstruct every reported line, arc, exclusion and total from SQLite/source."""
    require(data.is_file(), "Missing raw coverage")
    coverage = Coverage(data_file=str(data))
    coverage.load()
    raw = coverage.get_data()
    require(raw.has_arcs(), "Lost raw branch coverage")
    files = raw.measured_files()
    require(bool(files), "Empty raw coverage")
    require(not any(raw.file_tracer(name) for name in files), "Unexpected file tracer")
    actual = json.loads(receipt.read_text())
    expected = report(coverage, files)
    require(
        {k: v for k, v in actual["meta"].items() if k != "timestamp"}
        == {k: v for k, v in expected["meta"].items() if k != "timestamp"},
        "Coverage metadata changed",
    )
    require(
        actual["files"] == expected["files"], "Coverage report differs from raw/source"
    )
    require(
        actual["totals"] == expected["totals"], "Coverage totals differ from raw/source"
    )
    return coverage, actual


def verify_union(
    parts: list[Path], combined: Path, receipt: Path, installed: Path, checkout: Path
) -> dict[str, Any]:
    """Add only proven unexecuted sources; retain exact line/branch/subprocess union.

    The caller authenticates all artifact bytes and verifies installed/checkout
    membership against the candidate before entering here. Distinct checkout
    paths remain distinct coverage entries, as in the original unsharded report.
    """
    sources = {str(p) for p in installed.rglob("*.py")}
    require(
        all(Path(p).resolve() == Path(p) for p in sources),
        "Installed source path override",
    )
    require(bool(sources), "Missing installed coverage sources")
    raw_parts: list[CoverageData] = []
    reports = []
    allowed = set(sources)
    # Validate names before asking coverage to open any source named in an artifact.
    for folder in parts:
        data = CoverageData(basename=str(folder / ".coverage"))
        require((folder / ".coverage").is_file(), "Missing raw coverage")
        data.read()
        for name in data.measured_files():
            path = Path(name)
            require(
                path.is_absolute() and str(path.resolve()) == name,
                "Coverage path override",
            )
            if name not in sources:
                require(path.is_relative_to(checkout), "Foreign coverage source")
                original = installed / path.relative_to(checkout)
                require(str(original) in sources, "Unexpected checkout coverage source")
                require(
                    path.read_bytes() == original.read_bytes(),
                    "Checkout coverage bytes changed",
                )
                allowed.add(name)
        coverage, fragment = read_coverage(
            folder / ".coverage", folder / "full-coverage.json"
        )
        raw_parts.append(coverage.get_data())
        reports.append(fragment)
    raw_union = set().union(*(raw.measured_files() for raw in raw_parts))
    # coverage combine is still performed by the existing workflow. Check its
    # exact raw union as well as its JSON; equal report percentages are insufficient.
    raw_final = CoverageData(basename=str(combined))
    require(combined.is_file(), "Missing combined raw coverage")
    raw_final.read()
    require(
        raw_final.measured_files() == raw_union, "Combined coverage membership changed"
    )
    for name in raw_union:
        require(
            set(raw_final.arcs(name) or [])
            == set().union(*(set(raw.arcs(name) or []) for raw in raw_parts)),
            "Incomplete raw arc union",
        )
    coverage, _ = read_coverage(combined, receipt)
    final = report(coverage, allowed)
    empty = Coverage(data_file=None)
    empty.get_data().add_arcs({name: [] for name in allowed})
    universe = report(empty, allowed)
    for name, result in final["files"].items():
        fragments = [part["files"][name] for part in reports if name in part["files"]]
        # An omitted fragment was proved absent from its raw DB by read_coverage,
        # so it contributes no execution. Its source universe comes from the
        # installed candidate, never from another shard's coverage claims.
        for kind in ("lines", "branches"):

            def values(file: dict[str, Any], key: str) -> set[Any]:
                return {tuple(x) if isinstance(x, list) else x for x in file[key]}

            executed = set().union(*(values(f, f"executed_{kind}") for f in fragments))
            expected = values(universe["files"][name], f"missing_{kind}")
            excluded = set(result["excluded_lines"]) if kind == "lines" else set()
            require(
                all(
                    (values(f, f"executed_{kind}") | values(f, f"missing_{kind}"))
                    - excluded
                    == expected
                    for f in [result, *fragments]
                ),
                "Coverage statement/branch universe changed",
            )
            require(
                values(result, f"executed_{kind}") == executed,
                "Incomplete coverage union",
            )
        require(
            all(
                f["excluded_lines"] == universe["files"][name]["excluded_lines"]
                for f in [result, *fragments]
            ),
            "Coverage exclusions changed",
        )
    return final

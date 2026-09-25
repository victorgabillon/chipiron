"""Run the unchanged pytest selection and retain timing and identity evidence."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from _pytest.reports import TestReport


class TestMetrics:
    """Observe pytest hooks without altering collection, fixtures or outcomes."""

    def __init__(self) -> None:
        """Create a separate recorder for this invocation."""
        self.selected: list[str] = []
        self.deselected: list[str] = []
        self.reports: list[dict[str, object]] = []

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        """Record the final selection after marker filtering."""
        self.selected = [item.nodeid for item in session.items]

    def pytest_deselected(self, items: list[pytest.Item]) -> None:
        """Retain excluded identities so disappearance remains visible."""
        self.deselected.extend(item.nodeid for item in items)

    def pytest_runtest_logreport(self, report: TestReport) -> None:
        """Record every phase, including failures and skips."""
        self.reports.append({
            "nodeid": report.nodeid,
            "phase": report.when,
            "duration": report.duration,
            "outcome": report.outcome,
        })


def main() -> int:
    """Pass through pytest's arguments and exit code, adding only measurements."""
    recorder = TestMetrics()
    started = time.monotonic()
    code = pytest.main(sys.argv[1:], plugins=[recorder])
    output = Path(".ci-reports")
    output.mkdir(exist_ok=True)
    (output / "tests.json").write_text(
        json.dumps(
            {
                "selected_ids": recorder.selected,
                "deselected_ids": recorder.deselected,
                "reports": recorder.reports,
                "elapsed_seconds": time.monotonic() - started,
                "exit_code": int(code),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())

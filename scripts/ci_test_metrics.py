"""Run pytest and retain actual serial or distributed test identity evidence."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from _pytest.reports import TestReport


class TestMetrics:
    """Observe collection and test reports without changing selection or outcomes."""

    def __init__(self) -> None:
        """Create an invocation-local recorder."""
        self.selected: list[str] = []
        self.deselected: list[str] = []
        self.reports: list[dict[str, object]] = []
        self.worker_collections: dict[str, dict[str, list[str]]] = {}

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        """Retain the final selection after marker filtering."""
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

    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        """Return actual worker collections to the controlling pytest process."""
        if hasattr(session.config, "workeroutput"):
            session.config.workeroutput["ci_collection"] = {
                "selected": self.selected,
                "deselected": self.deselected,
            }

    @pytest.hookimpl(optionalhook=True)
    def pytest_testnodedown(self, node: Any, error: object) -> None:
        """Require identical full collections from every successful worker."""
        if error is not None:
            return  # xdist already fails the invocation for the crashed worker.
        collection = node.workeroutput["ci_collection"]
        for other in self.worker_collections.values():
            if collection != other:
                message = "Workers reported different selected or deselected tests."
                raise RuntimeError(message)
        self.worker_collections[node.gateway.id] = collection
        self.selected = collection["selected"]
        self.deselected = collection["deselected"]


RECORDER: TestMetrics | None = None


def pytest_configure(config: pytest.Config) -> None:
    """Register the same observer in the controller and every xdist worker."""
    global RECORDER
    RECORDER = TestMetrics()
    config.pluginmanager.register(RECORDER, "ci-test-metrics-recorder")


def main() -> int:
    """Pass through pytest's arguments and exit code, adding only measurements."""
    started = time.monotonic()
    # Loading by name makes pytest pass this plugin to xdist's subprocesses too.
    code = pytest.main(["-p", "ci_test_metrics", *sys.argv[1:]])
    recorder = sys.modules["ci_test_metrics"].RECORDER
    assert recorder is not None
    output = Path(".ci-reports")
    output.mkdir(exist_ok=True)
    (output / "tests.json").write_text(
        json.dumps(
            {
                "selected_ids": recorder.selected,
                "deselected_ids": recorder.deselected,
                "reports": recorder.reports,
                "worker_collections": recorder.worker_collections,
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

"""Run pytest and retain actual serial or distributed test identity evidence."""

import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

    from _pytest.reports import TestReport


SHARD_COUNT = 4
PARTITION = "sha256-file-first8-big-mod-v1"


def shard_for(nodeid: str, count: int) -> int:
    """Keep a whole file on one runner, independently of collection order."""
    if count < 1:
        message = "Shard count must be positive"
        raise ValueError(message)
    path = nodeid.split("::", 1)[0]
    return int.from_bytes(hashlib.sha256(path.encode()).digest()[:8], "big") % count


class TestMetrics:
    """Observe real test phases and optionally partition the normal selection by file."""

    def __init__(self) -> None:
        """Create an invocation-local recorder."""
        self.selected: list[str] = []
        self.deselected: list[str] = []
        self.reports: list[dict[str, object]] = []
        self.worker_collections: dict[str, dict[str, Any]] = {}
        self.collect_only = False
        self.shard: dict[str, Any] | None = None

    def execution_errors(self) -> list[str]:
        """Fail closed on missing or duplicate executions, including setup skips."""
        errors = []
        expected = Counter(self.selected)
        if any(count != 1 for count in expected.values()):
            errors.append("Duplicate selected test identities")
        for phase in ("setup", "call", "teardown"):
            required = (
                Counter(
                    str(report["nodeid"])
                    for report in self.reports
                    if report["phase"] == "setup" and report["outcome"] == "passed"
                )
                if phase == "call"
                else expected
            )
            observed = Counter(
                str(report["nodeid"])
                for report in self.reports
                if report["phase"] == phase
            )
            if required != observed:
                errors.append(
                    f"{phase}: missing={required - observed}; extra={observed - required}"
                )
        return errors

    @pytest.hookimpl(wrapper=True, tryfirst=True)
    def pytest_collection_modifyitems(
        self, config: pytest.Config, items: list[pytest.Item]
    ) -> Generator[None]:
        """Partition only after the unchanged normal collection/filtering hooks."""
        yield
        index = config.getoption("ci_shard_index")
        count = config.getoption("ci_shard_count")
        if index is None:
            return
        assert count is not None  # Both options were validated by pytest_configure.
        self.shard = {
            "index": index,
            "count": count,
            "algorithm": PARTITION,
            "full_selected_ids": [item.nodeid for item in items],
            "full_deselected_ids": self.deselected.copy(),
        }
        excluded = [item for item in items if shard_for(item.nodeid, count) != index]
        items[:] = [item for item in items if shard_for(item.nodeid, count) == index]
        config.hook.pytest_deselected(items=excluded)

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
                "shard": self.shard,
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
        self.shard = collection["shard"]


RECORDER: TestMetrics | None = None
STARTED = time.monotonic()


def pytest_addoption(parser: pytest.Parser) -> None:
    """Enable explicit sharding; the default invocation remains unpartitioned."""
    parser.addoption("--ci-shard-index", type=int, default=None)
    parser.addoption("--ci-shard-count", type=int, default=None)


def pytest_configure(config: pytest.Config) -> None:
    """Register the same observer in the controller and every xdist worker."""
    global RECORDER
    index = config.getoption("ci_shard_index")
    count = config.getoption("ci_shard_count")
    if (index is None) != (count is None) or (
        index is not None and (count is None or count < 1 or not 0 <= index < count)
    ):
        message = "Specify both shard options with 0 <= index < count"
        raise pytest.UsageError(message)
    RECORDER = TestMetrics()
    RECORDER.collect_only = config.option.collectonly
    config.pluginmanager.register(RECORDER, "ci-test-metrics-recorder")


def main() -> int:
    """Pass through pytest's arguments and exit code, adding only measurements."""
    started = time.monotonic()
    # Loading by name makes pytest pass this plugin to xdist's subprocesses too.
    code = pytest.main(["-p", "ci_test_metrics", *sys.argv[1:]])
    recorder = sys.modules["ci_test_metrics"].RECORDER
    assert recorder is not None
    return write_report(recorder, code, started, Path(".ci-reports/tests.json"))


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session) -> None:
    """Optionally observe the unchanged python -m pytest integration command."""
    output = os.environ.get("CI_TEST_METRICS_OUTPUT")
    if output and not hasattr(session.config, "workeroutput"):
        assert RECORDER is not None
        session.exitstatus = write_report(
            RECORDER, session.exitstatus, STARTED, Path(output)
        )


def write_report(recorder: TestMetrics, code: int, started: float, output: Path) -> int:
    """Retain selections and phases, rejecting incomplete successful executions."""
    execution_errors = [] if recorder.collect_only else recorder.execution_errors()
    if execution_errors and code == pytest.ExitCode.OK:
        code = pytest.ExitCode.TESTS_FAILED
    for error in execution_errors:
        print(f"CI execution identity failure: {error}", file=sys.stderr)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "selected_ids": recorder.selected,
                "deselected_ids": recorder.deselected,
                "reports": recorder.reports,
                "worker_collections": recorder.worker_collections,
                "shard": recorder.shard,
                "collect_only": recorder.collect_only,
                "execution_errors": execution_errors,
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

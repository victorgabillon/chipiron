"""Pytest-local environment setup for PR1 characterization tests."""

import os
import sys
import tempfile
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TESTS_ROOT.parent
TEST_OUTPUT_DIR = Path(os.environ.get("CHIPIRON_OUTPUT_DIR", tempfile.gettempdir()))

if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

# Test this checkout; sibling development must be explicitly installed or put
# on PYTHONPATH. Silently preferring adjacent dirty repositories hides release
# incompatibilities and makes the same command mean different things in CI.
source_path = REPO_ROOT / "src"
if (
    os.environ.get("CHIPIRON_TEST_INSTALLED") != "1"
    and str(source_path) not in sys.path
):
    sys.path.insert(0, str(source_path))


# xdist workers must never share SQLite tracking stores or application outputs.
# Serial invocations also own a fresh root. Inherited tracking URIs are never
# test destinations; child processes inherit the same private scope.
def _guard_test_output() -> None:
    """Check the configured base before creating any test directories."""
    from chipiron.utils.path_runtime import assert_test_output_path

    assert_test_output_path(TEST_OUTPUT_DIR)


os.environ["CHIPIRON_TEST_OUTPUT_ROOT"] = str(TEST_OUTPUT_DIR)
_guard_test_output()
worker_id = os.environ.get("PYTEST_XDIST_WORKER")
if worker_id:
    run_id = os.environ["PYTEST_XDIST_TESTRUNUID"]
    TEST_OUTPUT_DIR = TEST_OUTPUT_DIR / run_id / worker_id
else:
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEST_OUTPUT_DIR = Path(
        tempfile.mkdtemp(prefix="chipiron-tests-", dir=TEST_OUTPUT_DIR)
    )

TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ["CHIPIRON_TEST_OUTPUT_ROOT"] = str(TEST_OUTPUT_DIR)
os.environ["CHIPIRON_OUTPUT_DIR"] = str(TEST_OUTPUT_DIR)
os.environ["MORPION_WORK_DIR"] = str(TEST_OUTPUT_DIR / "bootstrap")
os.environ["XDG_CACHE_HOME"] = str(TEST_OUTPUT_DIR / "cache")
os.environ["ML_FLOW_DB_PATH"] = str(TEST_OUTPUT_DIR / "mlruns.db")
os.environ["ML_FLOW_URI_PATH"] = f"sqlite:///{TEST_OUTPUT_DIR / 'mlruns.db'}"
os.environ["ML_FLOW_URI_PATH_TEST"] = f"sqlite:///{TEST_OUTPUT_DIR / 'mlruns_test.db'}"

# Default record persistence must never rewrite the operator's home leaderboard.
# Reuse the existing isolated application-output root; tests can override explicitly.
os.environ["MORPION_LEADERBOARD_PATH"] = str(TEST_OUTPUT_DIR / "leaderboard.jsonl")

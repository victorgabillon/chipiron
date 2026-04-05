"""Pytest-local environment setup for PR1 characterization tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TESTS_ROOT.parent
TEST_OUTPUT_DIR = Path("/tmp/chipiron-test-output")

if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

for path in (
    REPO_ROOT / "src",
    REPO_ROOT.parent / "atomheart" / "src",
    REPO_ROOT.parent / "anemone" / "src",
):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("CHIPIRON_OUTPUT_DIR", str(TEST_OUTPUT_DIR))
os.environ.setdefault(
    "ML_FLOW_URI_PATH",
    f"sqlite:///{(TEST_OUTPUT_DIR / 'mlruns.db').as_posix()}",
)
os.environ.setdefault(
    "ML_FLOW_URI_PATH_TEST",
    f"sqlite:///{(TEST_OUTPUT_DIR / 'mlruns_test.db').as_posix()}",
)

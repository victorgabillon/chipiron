"""Reject tracked research outputs and runtime caches without banning fixtures."""

import subprocess
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
# Hand-authored historical source snapshots and a scientific interpretation note.
RETAINED_EVIDENCE = {
    "docs/research/activation_completion/evidence/superseded/README.txt",
    "docs/research/atomic_model_activation/evidence/probe-measured.py.txt",
    "docs/research/bounded_anemone_integration/evidence/restore_control.py.txt",
    "docs/research/bounded_anemone_integration/evidence/run_pair.py.txt",
}
CACHE_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    ".tox",
    ".venv",
    "tensor_cache",
}
RUNTIME_ROOTS = {
    ".ci-reports",
    "morpion_runs",
    "comparisons",
    "action_supervision",
    "mlruns",
}
SCRIPT_OUTPUTS = {
    "outputs",
    "test_outputs",
    "tests_outputs",
    "default_output_folder",
    "mlflow_data",
    "learn_nn_supervised_outputs",
    "learn_from_scratch_outputs",
}


def generated(path: str) -> bool:
    """Classify by reserved locations; JSON/YAML extensions are unrestricted."""
    parts = PurePosixPath(path).parts
    if parts[-1] == ".gitkeep" or path in RETAINED_EVIDENCE:
        return False
    return bool(
        (path.startswith("docs/research/") and "evidence" in parts[2:-1])
        or CACHE_DIRS.intersection(parts[:-1])
        or parts[0] in RUNTIME_ROOTS
        or path.endswith((".pyc", ".pyo"))
        or path.startswith(("chipiron/runs/", "src/chipiron/runs/"))
        or (
            path.startswith(("chipiron/scripts/", "src/chipiron/scripts/"))
            and SCRIPT_OUTPUTS.intersection(parts[:-1])
        )
    )


def main() -> int:
    """Inspect the index, including force-added files and staged deletions."""
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    rejected = sorted(
        path for path in paths.decode().split("\0") if path and generated(path)
    )
    if rejected:
        print("Generated artifacts must stay outside the tracked tree:")
        print("\n".join(rejected))
        return 1
    print("Tracked generated-artifact guard passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

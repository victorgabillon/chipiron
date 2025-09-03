import os
import subprocess
import sys

SCRIPT_PATH = os.path.join(
    os.path.dirname(__file__), "../../chipiron/scripts/main_chipiron.py"
)


def test_main_chipiron_one_match_executes():
    """Test that main_chipiron.py runs with --script_name one_match without error."""
    result = subprocess.run(
        [sys.executable, SCRIPT_PATH, "--script_name", "one_match"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Process failed: {result.stderr}"
    assert "error" not in result.stderr.lower(), f"Error in output: {result.stderr}"
    # Optionally check for expected output in result.stdout


if __name__ == "__main__":
    test_main_chipiron_one_match_executes()

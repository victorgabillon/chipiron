"""Module for test main chipiron execution."""
import os
import subprocess
import sys

SCRIPT_PATH = os.path.join(
    os.path.dirname(__file__), "../../chipiron/scripts/main_chipiron.py"
)


def run_with_live_output(cmd, env):
    # Merge stderr into stdout so ordering is preserved
    """Run with live output."""
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # line-buffered
        universal_newlines=True,
        env=env,
    )

    lines = []
    assert p.stdout is not None
    for line in p.stdout:
        print(line, end="")  # live to terminal
        lines.append(line)  # capture

    returncode = p.wait()
    output = "".join(lines)
    return returncode, output


def test_main_chipiron_one_match_executes():
    """Test main chipiron one match executes."""
    cmd = [
        sys.executable,
        SCRIPT_PATH,
        "--script_name",
        "one_match",
        "--match_args.player_one_overwrite.main_move_selector.anemone_args.stopping_criterion.tree_branch_limit",
        "100",
    ]

    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["MPLBACKEND"] = "Agg"

    returncode, output = run_with_live_output(cmd, env)

    assert returncode == 0, f"Process failed:\n{output}"
    assert "error" not in output.lower(), f"Error in output:\n{output}"


if __name__ == "__main__":
    test_main_chipiron_one_match_executes()
    print("Test passed.")

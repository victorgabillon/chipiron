"""Module for test main chipiron execution."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

TESTS_ROOT = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(TESTS_ROOT, "../.."))
ANEMONE_SRC = os.path.abspath(os.path.join(REPO_ROOT, "../anemone/src"))

SCRIPT_PATH = os.path.join(
    os.path.dirname(__file__), "../../src/chipiron/scripts/main_chipiron.py"
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


def test_main_chipiron_one_match_executes(tmp_path: Path) -> None:
    """Test main chipiron one match executes."""
    # NOTE:
    # This test runs the full application via subprocess.
    # Coverage is collected through `.coveragerc` subprocess support.
    config_path = tmp_path / "fast_one_match.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            gui: false
            match_args:
              player_one: Random
              player_two: Random
              match_setting:
                schedule:
                  type: two_role_match_schedule
                  number_of_games_player_one_on_first_role: 1
                  number_of_games_player_one_on_second_role: 0
                game_args:
                  game_kind: chess
                  each_player_has_its_own_thread: false
                  max_half_moves: 2
                  starting_position:
                    type: from_file
                    file_name: Board1.text
            base_script_args:
              seed: 11
              profiling: false
              testing: true
              relative_script_instance_experiment_output_folder: test_main_chipiron_execution
            """
        ).strip(),
        encoding="utf-8",
    )

    cmd = [
        sys.executable,
        SCRIPT_PATH,
        "--script_name",
        "one_match",
        "--config_file_name",
        str(config_path),
    ]

    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["MPLBACKEND"] = "Agg"
    env["PYTHONPATH"] = os.pathsep.join(
        path
        for path in (
            ANEMONE_SRC,
            env.get("PYTHONPATH"),
        )
        if path
    )

    returncode, output = run_with_live_output(cmd, env)

    assert returncode == 0, f"Process failed:\n{output}"
    assert "error" not in output.lower(), f"Error in output:\n{output}"


if __name__ == "__main__":
    test_main_chipiron_one_match_executes()
    print("Test passed.")

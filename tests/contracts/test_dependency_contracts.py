"""Targeted cross-repo compatibility checks for the PR1 safety net."""

from __future__ import annotations

import json
import subprocess
import sys

from test_support.import_compat import bootstrap_test_imports

bootstrap_test_imports()

from valanga import Color

from atomheart.games.checkers.state import CheckersState
from atomheart.games.integer_reduction.state import IntegerReductionState


def test_installed_valanga_drift_is_visible_in_a_clean_subprocess() -> None:
    """Expose the currently installed Valanga surface that chipiron still sees."""
    script = """
import json
import valanga
import valanga.evaluations as evaluations

print(json.dumps({
    "Role": hasattr(valanga, "Role"),
    "Outcome": hasattr(valanga, "Outcome"),
    "SoloRole": hasattr(valanga, "SoloRole"),
    "Value": hasattr(evaluations, "Value"),
    "Certainty": hasattr(evaluations, "Certainty"),
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    surface = json.loads(completed.stdout)

    assert surface == {
        "Role": False,
        "Outcome": False,
        "SoloRole": False,
        "Value": False,
        "Certainty": False,
    }


def test_local_atomheart_contains_both_turnless_and_turn_based_states() -> None:
    """Make the current single-player versus two-player state mismatch explicit."""
    solo_state = IntegerReductionState(3)
    alternating_state = CheckersState.standard(turn=Color.BLACK)

    assert hasattr(solo_state, "turn") is False
    assert hasattr(alternating_state, "turn") is True
    assert alternating_state.turn is Color.BLACK

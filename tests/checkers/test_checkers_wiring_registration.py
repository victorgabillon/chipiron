"""Regression tests for checkers wiring registration."""

from chipiron.environments.player_wiring_registry import get_observer_wiring
from chipiron.environments.types import GameKind


def test_checkers_wiring_is_registered() -> None:
    """Verify checkers wiring is registered."""
    wiring = get_observer_wiring(GameKind.CHECKERS)
    assert wiring is not None


def test_morpion_wiring_is_registered() -> None:
    """Verify morpion wiring is registered."""
    wiring = get_observer_wiring(GameKind.MORPION)
    assert wiring is not None

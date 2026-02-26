from chipiron.environments.player_wiring_registry import get_observer_wiring
from chipiron.environments.types import GameKind


def test_checkers_wiring_is_registered() -> None:
    wiring = get_observer_wiring(GameKind.CHECKERS)
    assert wiring is not None

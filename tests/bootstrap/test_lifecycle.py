"""Pure lifecycle routing never creates a second state authority."""

import pytest

from chipiron.bootstrap.lifecycle import select_action


@pytest.mark.fast
@pytest.mark.parametrize("mode", ["auto", "resume", "fresh"])
def test_existing_routing(mode: str) -> None:
    """Fresh refuses populated destinations; only valid committed state resumes."""
    if mode == "fresh":
        with pytest.raises(ValueError, match="Fresh requires"):
            select_action("fresh", empty=False, committed=True)
    else:
        assert select_action(mode, empty=False, committed=True) == "resume"
        assert (
            select_action(mode, empty=False, committed=True, terminal=True) == "no-work"
        )
        with pytest.raises(ValueError, match="Partial"):
            select_action(mode, empty=False, committed=False)


@pytest.mark.fast
def test_empty_routing() -> None:
    """An explicit resume never initializes."""
    assert select_action("auto", empty=True, committed=False) == "create"
    assert select_action("fresh", empty=True, committed=False) == "create"
    with pytest.raises(ValueError, match="existing committed"):
        select_action("resume", empty=True, committed=False)

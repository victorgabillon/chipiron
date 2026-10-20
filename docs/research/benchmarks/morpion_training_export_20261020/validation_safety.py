"""Keep optional pipeline-test leaderboard writes inside temporary fixtures."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def temporary_leaderboard(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Redirect only default leaderboard destinations; retain all record logic."""
    from chipiron.environments.morpion.bootstrap import bootstrap_loop, record_status
    from chipiron.environments.morpion.bootstrap.pipeline import stages

    original = record_status.persist_certified_leaderboard_candidates

    def isolated(**kwargs: Any) -> Any:
        if kwargs.get("leaderboard_path") is None:
            kwargs["leaderboard_path"] = tmp_path / "isolated-leaderboard.jsonl"
        return original(**kwargs)

    for module in [bootstrap_loop, record_status, stages]:
        monkeypatch.setattr(
            module, "persist_certified_leaderboard_candidates", isolated
        )

"""Independent scientific oracles through explicit atomic-protocol genesis."""

from functools import partial
from pathlib import Path
from typing import Any, get_args

import pytest

from chipiron.environments.morpion.bootstrap.pv_family_targets import (
    PvFamilyTargetPolicy,
)
from tests.environments import test_morpion_durable_scientific_v2 as scientific
from tests.environments.test_morpion_durable_closed_science import closed_adapter


def atomic_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Change only genesis protocol selection; retain every scientific assertion."""
    from chipiron.environments.morpion.bootstrap import (
        scientific_publication_initialization as initialization,
    )

    closed_adapter(monkeypatch)
    monkeypatch.setattr(
        initialization,
        "initialize_scientific_v2_publication",
        partial(
            initialization.initialize_scientific_v2_publication,
            atomic_reevaluation=True,
        ),
    )


@pytest.mark.parametrize("suffix", ["json.zst", "sharded"])
def test_atomic_checkpoint_selector_rng(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str
) -> None:
    """Preserve actual checkpoint values and selector/RNG continuation in both codecs."""
    atomic_adapter(monkeypatch)
    scientific.test_real_runner_checkpoint_selector_and_scientific_parity(
        tmp_path, suffix
    )


@pytest.fixture(scope="module")
def atomic_endgame(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Generate the actual FORCED35/direct ESTIMATE42 regression under new genesis."""
    with pytest.MonkeyPatch.context() as patch:
        atomic_adapter(patch)
        return scientific.committed_endgame.__wrapped__(tmp_path_factory)


@pytest.mark.parametrize("policy", get_args(PvFamilyTargetPolicy))
@pytest.mark.parametrize("backup", [False, True])
@pytest.mark.parametrize("exact_only", [False, True])
def test_atomic_endgame_confidence(
    tmp_path: Path,
    atomic_endgame: Any,
    monkeypatch: pytest.MonkeyPatch,
    policy: str,
    backup: bool,
    exact_only: bool,
) -> None:
    """Compare complete reference/indexed rows; five enabled policies, four gated."""
    atomic_adapter(monkeypatch)
    scientific.test_managed_real_endgame_confidence_all_policies(
        tmp_path, atomic_endgame, policy, backup, exact_only
    )

"""Run established scientific assertions through the new explicit storage adapter.

Only fixture source selection changes; all prior scientific assertions are reused.
The worker routing adapter here is test-only: production defaults stay disabled.
"""

from functools import partial
from typing import TYPE_CHECKING, Any, get_args

import pytest

from chipiron.environments.morpion.bootstrap.closed_scientific_snapshot import (
    load_closed_scientific_snapshot,
)
from chipiron.environments.morpion.bootstrap.pv_family_targets import (
    PvFamilyTargetPolicy,
)
from chipiron.environments.morpion.bootstrap.sharded_training_reader import (
    ShardedTrainingGenerationView,
)
from tests.environments import test_morpion_durable_scientific_v2 as scientific
from tests.environments.morpion_durable_closed_helpers import setup_closed_baseline

if TYPE_CHECKING:
    from pathlib import Path


def closed_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Select closed storage for the existing scientific assertions."""
    from chipiron.environments.morpion.bootstrap import (
        scientific_publication_initialization as initialization,
    )
    from chipiron.environments.morpion.bootstrap.runtime.runner import (
        AnemoneMorpionSearchRunner,
    )

    monkeypatch.setattr(scientific, "setup_scientific_baseline", setup_closed_baseline)
    monkeypatch.setattr(
        scientific,
        "load_scientific_training_generation",
        load_closed_scientific_snapshot,
    )
    monkeypatch.setattr(
        scientific,
        "ShardedTrainingGenerationView",
        partial(ShardedTrainingGenerationView, allow_closed_snapshot=True),
    )
    monkeypatch.setattr(
        initialization,
        "initialize_scientific_v2_publication",
        partial(
            initialization.initialize_scientific_v2_publication,
            closed_scientific_snapshot=True,
        ),
    )
    monkeypatch.setattr(
        AnemoneMorpionSearchRunner,
        "export_scientific_training_snapshot",
        AnemoneMorpionSearchRunner.export_closed_scientific_snapshot,
    )


@pytest.mark.parametrize("suffix", ["json.zst", "sharded"])
def test_actual_checkpoint_selector_rng(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str
) -> None:
    """Verify actual checkpoint selector rng."""
    closed_adapter(monkeypatch)
    scientific.test_real_runner_checkpoint_selector_and_scientific_parity(
        tmp_path, suffix
    )


@pytest.fixture(scope="module")
def closed_endgame(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Publish the actual FORCED35 and ESTIMATE42 regression fixture."""
    with pytest.MonkeyPatch.context() as patch:
        closed_adapter(patch)
        return scientific.committed_endgame.__wrapped__(tmp_path_factory)


@pytest.mark.parametrize("policy", get_args(PvFamilyTargetPolicy))
@pytest.mark.parametrize("backup", [False, True])
@pytest.mark.parametrize("exact_only", [False, True])
def test_real_endgame_all_confidence_policies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    closed_endgame: Any,
    policy: str,
    backup: bool,
    exact_only: bool,
) -> None:
    """Verify real endgame all confidence policies."""
    closed_adapter(monkeypatch)
    scientific.test_managed_real_endgame_confidence_all_policies(
        tmp_path, closed_endgame, policy, backup, exact_only
    )


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("cursor", [None, "root", "missing"])
def test_exact_reevaluation_patch_and_cursor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fallback: bool, cursor: Any
) -> None:
    """Verify exact reevaluation patch and cursor."""
    from chipiron.environments.morpion.bootstrap import reevaluation_worker as worker

    closed_adapter(monkeypatch)
    monkeypatch.setattr(worker, "is_sharded_generation_manifest", lambda _: True)
    monkeypatch.setattr(
        worker,
        "ShardedTrainingGenerationView",
        partial(ShardedTrainingGenerationView, allow_closed_snapshot=True),
    )
    scientific.test_managed_v2_reevaluation_complete_patch_and_cursor_parity(
        tmp_path, monkeypatch, fallback, cursor
    )

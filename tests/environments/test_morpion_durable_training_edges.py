"""Independent producer and serializer checks for accepted Training outputs."""

from typing import TYPE_CHECKING, Any

import pytest

from chipiron.environments.morpion.bootstrap.pipeline.stages import (
    run_pipeline_training_stage,
)
from tests.environments.morpion_durable_training_helpers import setup_training
from tests.environments.morpion_fenced_dataset_helpers import local_leaderboard

if TYPE_CHECKING:
    from pathlib import Path


def test_training_keeps_exact_reevaluation_proposal(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Training acceptance must not retag an existing Reevaluation producer or model."""
    from tests.environments.morpion_atomic_reevaluation_helpers import atomic_worker

    local_leaderboard(monkeypatch, tmp_path)
    args, paths, _ = setup_training(tmp_path / "workspace")
    with pytest.MonkeyPatch.context() as adapter:
        atomic_worker(paths.work_dir, adapter, identifier="unchanged-patch")
    before = paths.pipeline_reevaluation_patch_path.read_bytes()
    paths.pipeline_reevaluation_patch_path.unlink()  # Only this fixture's unaccepted proposal.
    run_pipeline_training_stage(args, generation=1)
    with pytest.MonkeyPatch.context() as adapter:
        atomic_worker(paths.work_dir, adapter, identifier="unchanged-patch")
    assert paths.pipeline_reevaluation_patch_path.read_bytes() == before


@pytest.mark.parametrize("writer", ["bundle", "metadata"])
def test_accepted_model_cannot_be_rewritten(
    tmp_path: Path, monkeypatch: Any, writer: str
) -> None:
    """Both normal model serializers respect the accepted immutable output boundary."""
    from chipiron.environments.morpion.players.evaluators.neural_networks.bundle import (
        load_morpion_model_bundle,
        save_morpion_model_bundle,
    )
    from chipiron.environments.morpion.players.evaluators.neural_networks.training.metadata import (
        update_saved_manifest_metadata,
    )

    local_leaderboard(monkeypatch, tmp_path)
    args, paths, _ = setup_training(tmp_path / "workspace")
    result = run_pipeline_training_stage(args, generation=1)
    bundle = paths.work_dir / result.model_bundle_paths["linear"]
    before = {p: p.read_bytes() for p in bundle.iterdir()}
    model, model_args, manifest = load_morpion_model_bundle(bundle)
    with pytest.raises(ValueError, match="immutable"):
        if writer == "bundle":
            save_morpion_model_bundle(
                model, bundle, model_args=model_args, metadata=manifest.metadata
            )
        else:
            update_saved_manifest_metadata(bundle, {})
    assert {p: p.read_bytes() for p in bundle.iterdir()} == before

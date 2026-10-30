"""Combined I3/G4 preserve their separate lineage and restoration contracts."""

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
import torch

from chipiron.environments.morpion.bootstrap import candidate_resources
from chipiron.environments.morpion.bootstrap.candidate_job import (
    CandidateJob,
    run_candidate,
)
from chipiron.environments.morpion.bootstrap.cycle_training import (
    morpion_training_args_from_evaluator_spec,
)
from chipiron.environments.morpion.bootstrap.evaluator_family import (
    transformer_v1_morpion_evaluator_spec,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    artifact_digest,
    file_digest,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.bundle import (
    load_morpion_model_bundle,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training import (
    MorpionStreamingTrainingArgs,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training.continuation import (
    MorpionTrainingContinuation,
)
from chipiron.learning.torch_runtime import (
    TORCH_EXECUTION_PROFILE_KEY,
    TorchExecutionProfileError,
    torch_execution_profile,
)
from tests.environments.test_morpion_continual_training import rows_file

if TYPE_CHECKING:
    from pathlib import Path

    from chipiron.learning.torch_runtime import TorchExecutionProfile


def _job(
    rows: Path, output: Path, threads: int | None, parent: Path | None = None
) -> CandidateJob:
    continuation = MorpionTrainingContinuation(
        "transformer_v1",
        171,
        parent_bundle=None if parent is None else str(parent),
        parent_bundle_sha256=None if parent is None else artifact_digest(parent),
        parent_weights_sha256=None
        if parent is None
        else file_digest(parent / "param.pt"),
    )
    args = morpion_training_args_from_evaluator_spec(
        spec=replace(
            transformer_v1_morpion_evaluator_spec(), num_epochs=1, batch_size=4
        ),
        dataset_file=rows,
        output_dir=output,
        shuffle=True,
        validation_fraction=0.2,
        validation_seed=0,
        device="cpu",
        continuation=continuation,
    )
    return CandidateJob(
        "transformer_v1", MorpionStreamingTrainingArgs(args, row_chunk_size=5), threads
    )


@pytest.mark.fast
@pytest.mark.parametrize(
    "profile,threads", [("existing", 2), ("strict_math", None), ("strict_math", 1)]
)
def test_combined_candidate_preserves_separate_metadata_and_continuation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile: TorchExecutionProfile,
    threads: int | None,
) -> None:
    """Two-thread existing and one-thread strict each retain their own lineage."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setenv("CHIPIRON_CANDIDATE_WORKERS", "1")
    monkeypatch.setenv("CHIPIRON_CANDIDATE_CPU_BUDGET", "4")
    monkeypatch.setenv("CHIPIRON_CANDIDATE_RESERVED_CPUS", "2")
    monkeypatch.setattr(candidate_resources, "cpu_capacity", lambda: 8.0)
    try:
        rows = rows_file(tmp_path)
        with torch_execution_profile(profile):
            first = tmp_path / "first"
            run_candidate(_job(rows, first, threads))
            parent_digest = artifact_digest(first)
            second = tmp_path / "second"
            run_candidate(_job(rows, second, threads, first))
            _, _, manifest = load_morpion_model_bundle(second)
            metadata = manifest.metadata
            assert metadata["continuation"]["optimizer_state_loaded"]
            assert metadata["continuation"]["parent_bundle_sha256"] == parent_digest
            assert artifact_digest(first) == parent_digest
            assert torch.get_num_threads() == 1
            if profile == "existing":
                assert TORCH_EXECUTION_PROFILE_KEY not in metadata
            else:
                assert metadata[TORCH_EXECUTION_PROFILE_KEY]["intra_op_threads"] == 1
                assert (
                    metadata[TORCH_EXECUTION_PROFILE_KEY]["effective"] == "strict_math"
                )
            if threads is None:
                assert "cpu_threading" not in metadata
            else:
                assert metadata["cpu_threading"]["actual_intra_op_threads"] == threads
                assert metadata["cpu_threading"]["restored_intra_op_threads"] == 1
    finally:
        torch.set_num_threads(previous)


@pytest.mark.fast
def test_combined_unqualified_thread_profile_keeps_drift_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Integration cannot silently admit the unqualified two-thread strict variant."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setenv("CHIPIRON_CANDIDATE_WORKERS", "1")
    monkeypatch.setenv("CHIPIRON_CANDIDATE_CPU_BUDGET", "4")
    monkeypatch.setenv("CHIPIRON_CANDIDATE_RESERVED_CPUS", "2")
    monkeypatch.setattr(candidate_resources, "cpu_capacity", lambda: 8.0)
    try:
        rows = rows_file(tmp_path)
        with torch_execution_profile("strict_math"):
            with pytest.raises(TorchExecutionProfileError, match="settings changed"):
                run_candidate(_job(rows, tmp_path / "unqualified", 2))
            assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(previous)

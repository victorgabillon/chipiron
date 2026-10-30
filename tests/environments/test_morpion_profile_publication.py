"""Accepted retries retain execution profile identity without altering defaults."""

from typing import TYPE_CHECKING

import pytest

from chipiron.environments.morpion.bootstrap import training_publication as training
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    publication_records,
)
from chipiron.environments.morpion.bootstrap.pipeline.stages import (
    run_pipeline_training_stage,
)
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from chipiron.learning.torch_runtime import (
    TORCH_EXECUTION_PROFILE_KEY,
    TorchExecutionProfileError,
    torch_execution_profile,
    torch_execution_profile_metadata,
)
from tests.environments.morpion_fenced_dataset_helpers import (
    local_leaderboard,
    run_dataset,
)
from tests.environments.test_morpion_continual_publication import grow, setup_continual

if TYPE_CHECKING:
    from pathlib import Path

    from chipiron.learning.torch_runtime import TorchExecutionProfile


@pytest.mark.fast
@pytest.mark.parametrize("profile", ["existing", "strict_math"])
def test_accepted_retry_requires_same_profile_without_training_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile: TorchExecutionProfile
) -> None:
    """The acceptance receipt binds the opt-in and rejects upgrades/downgrades."""
    monkeypatch.setenv("CHIPIRON_CANDIDATE_WORKERS", "1")
    local_leaderboard(monkeypatch, tmp_path)
    with torch_execution_profile(profile):
        args, paths = setup_continual(tmp_path / "workspace", memory_floor_mb=1)
        state = grow(args)
        run_dataset(args, generation=state.generation)
        result = run_pipeline_training_stage(args, generation=state.generation)
        state = load_bootstrap_run_state(paths.run_state_path)
        reference = publication_records(state.metadata)[-1][training.ACCEPTED]
        proposal = training._proposal(paths, reference)
        configuration = proposal["admission"]["configuration"]
        expected = torch_execution_profile_metadata()
        if profile == "existing":
            assert TORCH_EXECUTION_PROFILE_KEY not in configuration
        else:
            assert (
                configuration[TORCH_EXECUTION_PROFILE_KEY]
                == expected[TORCH_EXECUTION_PROFILE_KEY]
            )
        authority = paths.run_state_path.read_bytes()
        monkeypatch.setattr(
            training,
            "_compare_incumbent",
            lambda *a, **k: pytest.fail("accepted retry trained again"),
        )
        assert run_pipeline_training_stage(args, generation=state.generation) == result
        assert paths.run_state_path.read_bytes() == authority
    other: TorchExecutionProfile = (
        "strict_math" if profile == "existing" else "existing"
    )
    with (
        torch_execution_profile(other),
        pytest.raises(
            (TorchExecutionProfileError, ValueError), match=r"configuration|profile"
        ),
    ):
        run_pipeline_training_stage(args, generation=state.generation)
    assert paths.run_state_path.read_bytes() == authority

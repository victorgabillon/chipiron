"""Final acceptance fences for real private work crossing an authoritative stop."""

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest

from chipiron.environments.morpion.bootstrap import training_publication as training
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    committed_evaluator,
    growth_receipt_ownership,
    publication_records,
)
from chipiron.environments.morpion.bootstrap.learning_schedule import (
    BoundedGrowthWindow,
    window,
)
from chipiron.environments.morpion.bootstrap.pipeline import stages
from chipiron.environments.morpion.bootstrap.run_state import (
    load_bootstrap_run_state,
    save_bootstrap_run_state,
)
from tests.environments.morpion_fenced_dataset_helpers import run_dataset
from tests.environments.test_morpion_continual_publication import grow, setup_continual
from tests.environments.test_morpion_learning_schedule import monitored as monitored

if TYPE_CHECKING:
    from pathlib import Path

    from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
        MorpionBootstrapPaths,
    )


@pytest.mark.fast
@pytest.mark.parametrize("stage", ["dataset", "training"])
@pytest.mark.usefixtures("monitored")
def test_private_work_cannot_be_accepted_after_termination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    """Real published output cannot advance authority after its owner has stopped."""
    args, paths = setup_continual(
        tmp_path / "run",
        automatic=True,
        memory_floor_mb=1,
        learning_schedule=BoundedGrowthWindow(2),
    )
    source = grow(args)
    incumbent = committed_evaluator(paths.work_dir, source)[0]
    if stage == "training":
        run_dataset(args, generation=source.generation)
    module = stages if stage == "dataset" else training
    name = f"accept_{stage}"
    original = getattr(module, name)
    stopped: list[bytes] = []

    def terminate_before_accept(
        actual_paths: MorpionBootstrapPaths, reference: dict[str, Any]
    ) -> Any:
        assert actual_paths.work_dir == paths.work_dir
        assert reference["attempt_id"]
        # Preserve the existing admission lifetime: only the short authoritative
        # stop commit takes Growth ownership, after real private publication.
        with growth_receipt_ownership(paths):
            state = load_bootstrap_run_state(paths.run_state_path)
            state = replace(
                state,
                metadata={
                    **state.metadata,
                    "experiment_termination": {"reason": "declared_budget_reached"},
                },
            )
            save_bootstrap_run_state(state, paths.run_state_path, durable=True)
            stopped.append(paths.run_state_path.read_bytes())
        return original(actual_paths, reference)

    monkeypatch.setattr(module, name, terminate_before_accept)
    execute = run_dataset if stage == "dataset" else stages.run_pipeline_training_stage
    with pytest.raises(ValueError, match="terminated"):
        execute(args, generation=source.generation)
    assert len(stopped) == 1
    assert paths.run_state_path.read_bytes() == stopped[0]
    state = load_bootstrap_run_state(paths.run_state_path)
    record = next(
        r
        for r in publication_records(state.metadata)
        if r["generation"] == source.generation
    )
    assert f"accepted_{stage}" not in record
    assert committed_evaluator(paths.work_dir, state)[0] == incumbent
    assert window(state)["active"] == window(source)["active"]
    # A fresh attempt after the stop must refuse before reaching publication.
    with pytest.raises(ValueError, match="terminated"):
        execute(args, generation=source.generation)
    assert len(stopped) == 1
    assert paths.run_state_path.read_bytes() == stopped[0]

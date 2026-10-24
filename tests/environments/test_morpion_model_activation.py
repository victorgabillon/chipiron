"""Scientific model activation uses the existing single durable commit."""

from pathlib import Path

import pytest
import torch

from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    committed_evaluator,
    publication_records,
    read_object,
)
from chipiron.environments.morpion.bootstrap.model_activation import (
    activate_model,
    prepare_model_activation_request,
)
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from tests.environments.morpion_activation_helpers import setup_activation
from tests.environments.morpion_fenced_dataset_helpers import local_leaderboard


@pytest.mark.fast
@pytest.mark.parametrize("layout", ["sharded", "json-zst"])
def test_activate_accepted_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layout: str
) -> None:
    """Training does not activate; explicit commit binds M2 and exact retry is inert."""
    local_leaderboard(monkeypatch, tmp_path)
    _, paths, _ = setup_activation(tmp_path / "workspace", layout=layout)
    before = load_bootstrap_run_state(paths.run_state_path)
    m1, _ = committed_evaluator(paths.work_dir, before)
    request = prepare_model_activation_request(paths, training_generation=1)
    rng = torch.get_rng_state().clone()
    after = activate_model(paths, request)
    assert torch.equal(rng, torch.get_rng_state())
    m2, physical = committed_evaluator(paths.work_dir, after)
    assert m2 == request["candidate"] and m2 != m1
    assert physical.is_dir()
    assert after.generation == before.generation + 1
    assert (
        after.metadata["atomic_reevaluation"] == before.metadata["atomic_reevaluation"]
    )
    assert after.metadata["checkpoint_bound_reevaluation_v1"]["receipts"] == []
    head = publication_records(after.metadata)[-1]
    seal = read_object(paths.work_dir / head["descriptor_path"])
    assert seal["active_evaluator"] == m2
    assert seal["model_activation"]["request"] == request
    assert len(seal["model_pins"]) == 2
    authority = paths.run_state_path.read_bytes()
    assert activate_model(paths, request) == after
    assert paths.run_state_path.read_bytes() == authority

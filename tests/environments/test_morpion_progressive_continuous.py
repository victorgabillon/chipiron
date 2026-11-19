"""Existing six-slot coordinator runs model-only promotion with real Adam lineage."""

import json
import time
from typing import TYPE_CHECKING, Any

import pytest
import torch

from chipiron.environments.morpion.bootstrap import (
    continuous_publication as publication,
)
from chipiron.environments.morpion.bootstrap import promotion
from chipiron.environments.morpion.bootstrap import training_publication as training
from chipiron.environments.morpion.bootstrap.config import load_bootstrap_config
from chipiron.environments.morpion.bootstrap.continuous_contract import (
    Policy,
    Reservation,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    committed_evaluator,
    publication_records,
)
from chipiron.environments.morpion.bootstrap.managed_sessions import begin_owned
from chipiron.environments.morpion.bootstrap.model_activation import (
    activation_completion,
)
from chipiron.environments.morpion.bootstrap.pipeline.stages import (
    run_pipeline_training_stage,
)
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from tests.environments.morpion_fenced_dataset_helpers import run_dataset
from tests.environments.test_morpion_continual_publication import grow, setup_continual
from tests.environments.test_morpion_continuous_runtime import candidate_family
from tests.environments.test_morpion_continuous_runtime import continuous as continuous
from tests.environments.test_morpion_learning_schedule import monitored as monitored
from tests.environments.test_morpion_prediction_origin_writes import tracked
from tests.environments.test_morpion_rolling_reevaluation import accepted, propose

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.usefixtures("continuous")
def test_real_comparison_abc_and_candidate_adam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use genuine accepted comparisons and each candidate's own optimizer parent."""
    from atomheart.games.morpion import MorpionDynamics, initial_state

    from chipiron.environments.morpion.bootstrap.runtime import runner as runtime_module

    dynamics = MorpionDynamics()
    position = initial_state()
    for _ in range(30):
        position = dynamics.step(
            position, dynamics.all_legal_actions(position)[0]
        ).next_state
    assert len(dynamics.all_legal_actions(position)) > 1
    with monkeypatch.context() as start:
        start.setattr(runtime_module, "initial_state", lambda: position)
        args, paths = setup_continual(
            tmp_path / "run",
            automatic=True,
            memory_floor_mb=1,
            evaluators=candidate_family(),
            learning_schedule=Policy(activation="owned_cold_model_only_v1"),
            strict_execution=True,
            managed=True,
            target_source="effective",
            proof=True,
            extra_steps=64,
        )
    begin_owned(
        paths,
        load_bootstrap_config(paths.bootstrap_config_path),
        "42",
        time.monotonic(),
        180,
    )
    first = committed_evaluator(
        paths.work_dir, load_bootstrap_run_state(paths.run_state_path)
    )[0]
    assert first is not None
    models = [first["sha256"]]
    previous = None
    previous_steps: dict[str, int] = {}
    previous_weights: dict[str, Any] = {}
    observations = []
    # The proof fixture compares improving models on its actual rollout targets.
    for number in (1, 2):
        started = time.perf_counter()
        state = grow(args)
        assert propose(args, 1).patch_written
        state = grow(args)
        cursor = accepted(paths)
        generation = state.generation
        run_dataset(args, generation=generation)
        result = run_pipeline_training_stage(args, generation=generation)
        assert result.training_status == "done"
        state = load_bootstrap_run_state(paths.run_state_path)
        ref = next(
            r["accepted_training"]
            for r in publication_records(state.metadata)
            if r["generation"] == generation
        )
        proposal = training._proposal(paths, ref)
        for name, binding in proposal["models"].items():
            parent = proposal["admission"]["parents"][name]
            if previous is not None:
                assert parent["accepted_training"] == previous
            else:
                assert parent is None
            bundle = paths.work_dir / binding["path"]
            adam = torch.load(bundle / "optimizer.pt", weights_only=True)
            steps = max(int(v["step"]) for v in adam["state"].values())
            assert steps > previous_steps.get(name, 0)
            weights = torch.load(bundle / "param.pt", weights_only=True)
            if name in previous_weights:
                assert any(
                    not torch.equal(value, previous_weights[name][key])
                    for key, value in weights.items()
                )
            previous_steps[name] = steps
            previous_weights[name] = weights
        assert promotion.service(paths, args, ttl_seconds=120, owner=None) == generation
        reservation = promotion.reservation(
            paths, load_bootstrap_run_state(paths.run_state_path)
        )
        assert reservation is not None and isinstance(reservation.payload, Reservation)
        assert reservation.payload.phase == "VERIFIED"
        promotion.activate_at_boundary(paths)
        state = load_bootstrap_run_state(paths.run_state_path)
        assert publication.load_frame(paths, state).state.reserved_promotion is None
        model = committed_evaluator(paths.work_dir, state)[0]
        assert model is not None and model["sha256"] != models[-1]
        models.append(model["sha256"])
        completion = activation_completion(
            paths, state, publication_records(state.metadata)[-1]
        )
        assert completion["request"]["schema_version"] == 4
        assert (
            completion["mode"] == "owned_cold_model_only_v1"
            and "refresh" not in completion
        )
        assert accepted(paths) == cursor
        grow(args)
        origins = tracked(paths)
        assert {models[0], models[-1]} <= {
            o.evaluator_sha256 for o in origins.current_origins.values()
        }
        observations.append({
            "round": number,
            "wall_s": time.perf_counter() - started,
            "steps": dict(previous_steps),
            "models": dict(proposal["models"]),
        })
        previous = ref
    assert len(set(models)) == 3
    (tmp_path / "continuous-observations.json").write_text(
        json.dumps({"models": models, "rounds": observations}, indent=2)
    )

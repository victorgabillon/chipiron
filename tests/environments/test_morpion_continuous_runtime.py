"""Real context4 sources, native input handoff, own Adam and serial promotion."""

import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest
import torch

from chipiron.environments.morpion.bootstrap import (
    continuous_publication as publication,
)
from chipiron.environments.morpion.bootstrap import promotion
from chipiron.environments.morpion.bootstrap import training_publication as training
from chipiron.environments.morpion.bootstrap.config import load_bootstrap_config
from chipiron.environments.morpion.bootstrap.continuous_contract import Policy
from chipiron.environments.morpion.bootstrap.evaluator_config import (
    MorpionEvaluatorsConfig,
    MorpionEvaluatorSpec,
)
from chipiron.environments.morpion.bootstrap.evaluator_family import (
    transformer_v1_morpion_evaluator_spec,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    publication_records,
)
from chipiron.environments.morpion.bootstrap.managed_sessions import begin_owned
from chipiron.environments.morpion.bootstrap.pipeline.stages import (
    run_pipeline_training_stage,
)
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from chipiron.learning.torch_runtime import (
    configure_torch_process_threads,
    torch_execution_profile,
)
from tests.environments.morpion_fenced_dataset_helpers import run_dataset
from tests.environments.test_morpion_continual_publication import grow, setup_continual
from tests.environments.test_morpion_learning_schedule import monitored as monitored

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from chipiron.environments.morpion.learning.target_policy import TargetSource


@pytest.fixture
def continuous(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> Iterator[None]:
    """Use real monitored identities and explicit shared preparation, without GPU changes."""
    request.getfixturevalue("monitored")
    monkeypatch.setenv("CHIPIRON_SHARED_PREPARATION", "1")
    monkeypatch.delenv("CHIPIRON_SHARED_PREPARATION_REUSE", raising=False)
    monkeypatch.delenv("CHIPIRON_STATE_SPLIT_CACHE_DIR", raising=False)
    monkeypatch.setenv("MORPION_RESOURCE_MONITOR_SESSION", "42")
    configure_torch_process_threads(intra_op=1, inter_op=1)
    with torch_execution_profile("strict_math"):
        yield


def candidate_family() -> MorpionEvaluatorsConfig:
    """Use the supported flat plus relational family in every continuous fixture."""
    return MorpionEvaluatorsConfig(
        evaluators={
            "linear": MorpionEvaluatorSpec("linear", "linear", None, 1, 2, 0.001),
            "transformer_v1": replace(
                transformer_v1_morpion_evaluator_spec(),
                num_epochs=1,
                batch_size=2,
                entity_d_model=8,
                entity_n_head=2,
                entity_n_layer=1,
                entity_dim_feedforward=8,
            ),
        }
    )


def fixture(
    root: Path,
    *,
    proof: bool = False,
    extra_steps: int = 2,
    target_source: TargetSource | None = None,
) -> tuple[Any, Any]:
    """Create fresh managed context4 with two supported tiny independent models."""
    args, paths = setup_continual(
        root,
        automatic=True,
        memory_floor_mb=1,
        evaluators=candidate_family(),
        learning_schedule=Policy(),
        strict_execution=True,
        managed=True,
        target_source=target_source,
        proof=proof,
        extra_steps=extra_steps,
    )
    begin_owned(
        paths,
        load_bootstrap_config(paths.bootstrap_config_path),
        "42",
        time.monotonic(),
        60,
    )
    return args, paths


@pytest.mark.fast
@pytest.mark.usefixtures("continuous")
@pytest.mark.parametrize("target_source", [None, "effective"])
def test_real_two_round_own_adam_and_reserved_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target_source: TargetSource | None
) -> None:
    """Accept every candidate once, retain losing lineage, then compare separately."""
    args, paths = fixture(tmp_path / "run", target_source=target_source)
    previous = None
    previous_steps: dict[str, int] = {}
    for number in (1, 2):
        state = grow(args)
        generation = state.generation
        source = publication.source(paths, state, generation)
        run_dataset(args, generation=generation)
        ready_state = load_bootstrap_run_state(paths.run_state_path)
        before = publication.load_frame(paths, ready_state)
        assert before.state.ready is not None and before.state.dataset_job is None
        epoch = before.state.incumbent_epoch
        result = run_pipeline_training_stage(args, generation=generation)
        assert result.training_status == "done"
        state = load_bootstrap_run_state(paths.run_state_path)
        frame = publication.load_frame(paths, state)
        assert frame.state.incumbent_epoch == epoch
        assert frame.state.last_accepted_round == number
        assert frame.state.cohort is None and frame.state.pending_promotion is not None
        reference = next(
            r["accepted_training"]
            for r in publication_records(state.metadata)
            if r["generation"] == generation
        )
        proposal = training._proposal(paths, reference)
        assert "learned_cohort_v1" in proposal["status"]["metadata"]
        assert "continual_selection" not in proposal["status"]["metadata"]
        assert len(proposal["models"]) == 2
        for name in proposal["models"]:
            parent = proposal["admission"]["parents"][name]
            if previous is not None:
                assert parent["accepted_training"] == previous
            else:
                assert parent is None
            bundle = paths.work_dir / proposal["models"][name]["path"]
            from chipiron.environments.morpion.players.evaluators.neural_networks.bundle import (
                load_morpion_model_bundle,
            )

            _, _, manifest = load_morpion_model_bundle(bundle)
            assert manifest.target_kind == (
                "effective_value" if target_source else "backup_value"
            )
            if target_source:
                assert (
                    manifest.metadata["dataset_target_policy"]["source"]
                    == target_source
                )
                assert (
                    manifest.metadata["continuation"]["dataset_target_policy"]
                    == manifest.metadata["dataset_target_policy"]
                )
            adam = torch.load(
                bundle / "optimizer.pt", map_location="cpu", weights_only=True
            )
            steps = max(int(v["step"]) for v in adam["state"].values())
            assert steps > previous_steps.get(name, 0)
            previous_steps[name] = steps
        assert publication.source(paths, state, generation) == source
        authority = paths.run_state_path.read_bytes()
        with monkeypatch.context() as guard:
            guard.setattr(
                training,
                "_compare_incumbent",
                lambda *a, **k: pytest.fail("legacy comparison called"),
            )
            assert run_pipeline_training_stage(args, generation=generation) == result
            assert paths.run_state_path.read_bytes() == authority
        assert promotion.service(paths, args, ttl_seconds=60, owner=None) == generation
        reserved = promotion.reservation(
            paths, load_bootstrap_run_state(paths.run_state_path)
        )
        assert reserved is not None and reserved.payload.phase == "VERIFIED"
        promotion.activate_at_boundary(paths)
        state = load_bootstrap_run_state(paths.run_state_path)
        assert publication.load_frame(paths, state).state.reserved_promotion is None
        previous = reference
    assert not torch.cuda.is_initialized()


@pytest.mark.usefixtures("continuous")
def test_two_round_same_parent_numeric_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every candidate matches a private-input reference from the identical own parent."""
    import json
    import shutil

    from chipiron.environments.morpion.bootstrap import cycle_training
    from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
        MorpionBootstrapPaths,
    )
    from chipiron.environments.morpion.learning import (
        morpion_supervised_rows_source_from_path,
    )
    from tests.environments.test_morpion_candidate_execution import assert_state_equal

    train = cycle_training.train_and_select_evaluators_streaming
    rounds = 0

    def compare(**kwargs: Any) -> Any:
        nonlocal rounds
        result = train(**kwargs)
        rounds += 1
        reference = MorpionBootstrapPaths.from_work_dir(
            tmp_path / f"reference-{rounds}"
        )
        rows = reference.work_dir / "rows.jsonl"
        rows.parent.mkdir(parents=True)
        shutil.copyfile(kwargs["rows_path"], rows)
        with torch.random.fork_rng(devices=[]):
            expected = train(**{
                **kwargs,
                "paths": reference,
                "rows_path": rows,
                "rows_source": morpion_supervised_rows_source_from_path(rows),
                "prepared_inputs": None,
            })
        assert result.selected_evaluator_name == expected.selected_evaluator_name
        for name in result.model_bundle_paths:
            actual_path = kwargs["paths"].resolve_work_dir_path(
                result.model_bundle_paths[name]
            )
            expected_path = reference.resolve_work_dir_path(
                expected.model_bundle_paths[name]
            )
            for artifact in ("param.pt", "optimizer.pt"):
                assert_state_equal(
                    torch.load(actual_path / artifact, weights_only=True),
                    torch.load(expected_path / artifact, weights_only=True),
                )
            actual = json.loads((actual_path / "morpion_manifest.json").read_text())[
                "metadata"
            ]
            control = json.loads((expected_path / "morpion_manifest.json").read_text())[
                "metadata"
            ]
            for field in ("train_loss", "validation_loss", "state_split"):
                assert actual[field] == control[field]
            for field in (
                "rng_before_initialization",
                "rng_before_training",
                "rng_after_training",
                "optimizer_steps",
                "cumulative_optimizer_steps",
                "parent_optimizer_steps",
            ):
                assert actual["continuation"][field] == control["continuation"][field]
        return result

    monkeypatch.setattr(
        cycle_training, "train_and_select_evaluators_streaming", compare
    )
    test_real_two_round_own_adam_and_reserved_promotion(tmp_path, monkeypatch, None)
    assert rounds == 2

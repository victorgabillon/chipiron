"""Live runtime contracts fail before mutation and preserve real accepted retries."""

from typing import TYPE_CHECKING, Any

import pytest
import torch

from chipiron.environments.morpion.bootstrap import cycle_training
from chipiron.environments.morpion.bootstrap import model_activation as activation
from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
    MorpionBootstrapPaths,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    committed_evaluator,
    publication_records,
)
from chipiron.environments.morpion.bootstrap.pipeline.stages import (
    run_pipeline_growth_stage,
    run_pipeline_training_stage,
)
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from chipiron.environments.morpion.bootstrap.runtime import runner as runner_module
from chipiron.environments.morpion.bootstrap.runtime.runner import (
    AnemoneMorpionSearchRunner,
    AnemoneMorpionSearchRunnerArgs,
    default_search_args,
)
from tests.environments.morpion_activation_helpers import setup_activation
from tests.environments.morpion_fenced_dataset_helpers import (
    local_leaderboard,
    run_dataset,
)
from tests.environments.test_morpion_continual_publication import setup_continual
from tests.environments.test_morpion_value_certainty import endgame

if TYPE_CHECKING:
    from pathlib import Path


class DelegatedRuntime:
    """An external adapter: delegate real execution without Anemone inheritance."""

    def __init__(
        self,
        delegate: AnemoneMorpionSearchRunner,
        *,
        damaged: str | None = None,
        damage: str = "absent",
        unavailable: tuple[str, ...] = (),
    ) -> None:
        """Keep real hooks observable and optionally expose a broken external adapter."""
        self.delegate = delegate
        self.damaged = damaged
        self.damage = damage
        self.unavailable = unavailable
        self.calls: list[str] = []

    def __getattr__(self, name: str) -> Any:
        """Forward supported hooks while preserving external optional attributes."""
        if name in self.unavailable:
            raise AttributeError(name)
        if name == self.damaged:
            if self.damage == "absent":
                raise AttributeError(name)
            return None if self.damage == "none" else 42
        value = getattr(self.delegate, name)
        if not callable(value):
            return value

        def invoke(*args: Any, **kwargs: Any) -> Any:
            self.calls.append(name)
            return value(*args, **kwargs)

        return invoke


def _inventory(root: Path) -> dict[str, bytes]:
    """Observe every persisted authority, configuration and scientific artifact."""
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _grow(args: Any, runtime: DelegatedRuntime) -> Any:
    paths = MorpionBootstrapPaths.from_work_dir(args.work_dir)
    state = load_bootstrap_run_state(paths.run_state_path)
    producer = publication_records(state.metadata)[-1]["scientific_identity"][
        "producer"
    ]["identity"]
    return run_pipeline_growth_stage(
        args,
        runtime,
        checkpoint_bound_receipts=True,
        durable_generation_publication=True,
        scientific_v2_producer=producer,
        closed_scientific_snapshot=True,
    )


@pytest.mark.parametrize("damage", ["absent", "none", "noncallable"])
@pytest.mark.parametrize(
    "method",
    [
        "load_or_create",
        "grow",
        "current_tree_size",
        "set_unique_node_budget",
        "save_checkpoint",
        "discard_uncommitted_runtime",
        "apply_reevaluation_patch",
        "export_closed_scientific_snapshot",
    ],
)
def test_incomplete_continual_runtime_preserves_scientific_workspace(
    tmp_path: Path, method: str, damage: str
) -> None:
    """An absent or malformed operation cannot reach a budget setter or commit."""
    args, paths = setup_continual(tmp_path / "workspace", automatic=True)
    runtime = DelegatedRuntime(
        AnemoneMorpionSearchRunner(AnemoneMorpionSearchRunnerArgs()),
        damaged=method,
        damage=damage,
    )
    before = _inventory(paths.work_dir)
    with pytest.raises(ValueError, match=r"[Cc]apabilit|[Rr]untime"):
        _grow(args, runtime)
    assert runtime.calls == []
    assert _inventory(paths.work_dir) == before


@pytest.mark.parametrize("damage", ["absent", "none", "noncallable"])
@pytest.mark.parametrize(
    "method",
    [
        "load_or_create",
        "current_tree_size",
        "save_checkpoint",
        "discard_uncommitted_runtime",
        "apply_reevaluation_patch",
        "export_closed_scientific_snapshot",
        "validate_model_refresh_certificates",
        "refresh_model_estimates",
    ],
)
def test_incomplete_activation_runtime_preserves_scientific_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str, damage: str
) -> None:
    """Refuse before freeze, restore, patch drain, refresh or scientific publication."""
    local_leaderboard(monkeypatch, tmp_path)
    _, paths, _ = setup_activation(tmp_path / "workspace")
    state = load_bootstrap_run_state(paths.run_state_path)
    delegate, effective = activation._runner(paths, state)
    runtime = DelegatedRuntime(delegate, damaged=method, damage=damage)
    request = activation.prepare_model_activation_request(paths, training_generation=1)
    monkeypatch.setattr(activation, "_runner", lambda *_: (runtime, effective))
    before = _inventory(paths.work_dir)
    with pytest.raises(ValueError, match=r"[Cc]apabilit|[Rr]untime"):
        activation.activate_model(paths, request)
    assert runtime.calls == []
    assert _inventory(paths.work_dir) == before


@pytest.mark.fast
def test_delegated_runtime_runs_continual_cycle_and_activation_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real training remains separate from activation through an external adapter."""
    local_leaderboard(monkeypatch, tmp_path)
    position, _ = endgame.__wrapped__()
    with monkeypatch.context() as genesis:
        genesis.setattr(runner_module, "initial_state", lambda: position)
        args, paths = setup_continual(tmp_path / "workspace", proof=True)
    runtime = DelegatedRuntime(
        AnemoneMorpionSearchRunner(
            AnemoneMorpionSearchRunnerArgs(
                runtime_checkpoint_format="sharded",
                random_seed=4,
                search_args=default_search_args(rollout=args.search.rollout),
            )
        ),
        unavailable=("export_training_tree_snapshot",),
    )
    state = _grow(args, runtime)
    assert "grow" in runtime.calls
    assert "save_checkpoint" in runtime.calls
    assert "export_closed_scientific_snapshot" in runtime.calls
    before_training = committed_evaluator(paths.work_dir, state)[0]
    run_dataset(args, generation=state.generation)
    result = run_pipeline_training_stage(args, generation=state.generation)
    accepted = load_bootstrap_run_state(paths.run_state_path)
    assert committed_evaluator(paths.work_dir, accepted)[0] == before_training
    authority = paths.run_state_path.read_bytes()
    with monkeypatch.context() as retry:
        retry.setattr(
            cycle_training,
            "train_and_select_evaluators_streaming",
            lambda *a, **k: pytest.fail("Accepted retry retrained"),
        )
        assert run_pipeline_training_stage(args, generation=state.generation) == result
    assert paths.run_state_path.read_bytes() == authority
    delegate, effective = activation._runner(paths, accepted)
    activation_runtime = DelegatedRuntime(
        delegate, unavailable=("grow", "export_training_tree_snapshot")
    )
    monkeypatch.setattr(
        activation, "_runner", lambda *_: (activation_runtime, effective)
    )
    request = activation.prepare_model_activation_request(
        paths, training_generation=state.generation
    )
    rng = torch.get_rng_state().clone()
    activated = activation.activate_model(paths, request)
    assert torch.equal(rng, torch.get_rng_state())
    assert activation_runtime.calls.count("refresh_model_estimates") == 1
    assert "validate_model_refresh_certificates" in activation_runtime.calls
    assert committed_evaluator(paths.work_dir, activated)[0] == request["candidate"]
    authority = paths.run_state_path.read_bytes()
    paths.pipeline_active_model_path.unlink()
    paths.pipeline_manifest_path_for_generation(activated.generation).unlink()
    monkeypatch.setattr(
        activation,
        "_runner",
        lambda *_: pytest.fail("Accepted retry created a live runtime"),
    )
    assert activation.activate_model(paths, request) == activated
    assert paths.run_state_path.read_bytes() == authority
    assert paths.pipeline_active_model_path.is_file()
    assert paths.pipeline_manifest_path_for_generation(activated.generation).is_file()
    assert activation_runtime.calls.count("refresh_model_estimates") == 1

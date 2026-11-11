"""Fresh Morpion epochs and saved recipe authority across launcher restarts."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, cast

import pytest

from chipiron.environments.morpion.bootstrap.bootstrap_args import MorpionBootstrapArgs
from chipiron.environments.morpion.bootstrap.bootstrap_loop import (
    run_morpion_bootstrap_loop,
)
from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
    MorpionBootstrapPaths,
)
from chipiron.environments.morpion.bootstrap.config import (
    bootstrap_config_from_args,
    bootstrap_config_from_dict,
    bootstrap_config_sha256,
    bootstrap_config_to_dict,
    load_bootstrap_config,
    save_bootstrap_config,
)
from chipiron.environments.morpion.bootstrap.evaluator_config import (
    MorpionEvaluatorsConfig,
)
from chipiron.environments.morpion.bootstrap.evaluator_family import (
    CANONICAL_8_LINEAR_MLP_TRANSFORMER_V1_PRESET,
)
from chipiron.environments.morpion.bootstrap.launcher import (
    _collect_launcher_startup_status,
    launcher_args_from_cli,
)
from chipiron.environments.morpion.bootstrap.run_state import (
    initialize_bootstrap_run_state,
    save_bootstrap_run_state,
)

if TYPE_CHECKING:
    from pathlib import Path

    from chipiron.environments.morpion.bootstrap.search_runner_protocol import (
        MorpionSearchRunner,
    )


@pytest.mark.fast
def test_fresh_single_and_nine_candidate_family_resolve_one_epoch(
    tmp_path: Path,
) -> None:
    """The public fresh launcher persists one epoch for every generated member."""
    single = bootstrap_config_from_args(MorpionBootstrapArgs(work_dir=tmp_path))
    assert [spec.num_epochs for spec in single.evaluators.evaluators.values()] == [1]

    launch = launcher_args_from_cli([
        "--work-dir",
        str(tmp_path / "family"),
        "--evaluator-family",
        CANONICAL_8_LINEAR_MLP_TRANSFORMER_V1_PRESET,
    ])
    startup = _collect_launcher_startup_status(launch)
    specs = startup.bootstrap_config.evaluators.evaluators
    assert len(specs) == 9
    assert specs["transformer_v1"].num_epochs == 1
    assert {spec.num_epochs for spec in specs.values()} == {1}
    assert (
        load_bootstrap_config(startup.paths.bootstrap_config_path)
        == startup.bootstrap_config
    )
    assert (
        bootstrap_config_from_dict(bootstrap_config_to_dict(startup.bootstrap_config))
        == startup.bootstrap_config
    )


@pytest.mark.fast
def test_explicit_mixed_epochs_and_saved_five_survive_default_change(
    tmp_path: Path,
) -> None:
    """Explicit recipe values and an existing saved five-epoch family stay intact."""
    family_args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        evaluator_family_preset=CANONICAL_8_LINEAR_MLP_TRANSFORMER_V1_PRESET,
    )
    fresh = family_args.resolved_evaluators_config().evaluators
    mixed = dict(fresh)
    mixed["transformer_v1"] = replace(fresh["transformer_v1"], num_epochs=5)
    mixed_config = bootstrap_config_from_args(
        replace(
            family_args,
            evaluator_family_preset=None,
            evaluators_config=MorpionEvaluatorsConfig(evaluators=mixed),
        )
    )
    assert mixed_config.evaluators.evaluators["transformer_v1"].num_epochs == 5
    assert {
        spec.num_epochs for spec in mixed_config.evaluators.evaluators.values()
    } == {1, 5}

    old = bootstrap_config_from_args(
        replace(
            family_args,
            evaluator_family_preset=None,
            evaluators_config=MorpionEvaluatorsConfig(
                evaluators={
                    name: replace(spec, num_epochs=5) for name, spec in fresh.items()
                }
            ),
        )
    )
    config_path = tmp_path / "bootstrap_config.json"
    save_bootstrap_config(old, config_path)
    original_hash = bootstrap_config_sha256(old)
    launch = launcher_args_from_cli([
        "--work-dir",
        str(tmp_path),
        "--evaluator-family",
        CANONICAL_8_LINEAR_MLP_TRANSFORMER_V1_PRESET,
    ])
    startup = _collect_launcher_startup_status(launch)
    assert startup.bootstrap_config == old
    assert {
        spec.num_epochs
        for spec in startup.resolved_bootstrap_args.resolved_evaluators_config().evaluators.values()
    } == {5}
    assert bootstrap_config_sha256(load_bootstrap_config(config_path)) == original_hash

    run_morpion_bootstrap_loop(
        MorpionBootstrapArgs(
            work_dir=tmp_path,
            evaluator_family_preset=CANONICAL_8_LINEAR_MLP_TRANSFORMER_V1_PRESET,
        ),
        cast("MorpionSearchRunner", object()),
        max_cycles=0,
    )
    assert bootstrap_config_sha256(load_bootstrap_config(config_path)) == original_hash


@pytest.mark.fast
def test_legacy_preset_only_run_keeps_historical_five(tmp_path: Path) -> None:
    """An existing pre-config-file run is not reinterpreted as a fresh recipe."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    save_bootstrap_run_state(initialize_bootstrap_run_state(), paths.run_state_path)
    launch = launcher_args_from_cli([
        "--work-dir",
        str(tmp_path),
        "--evaluator-family",
        CANONICAL_8_LINEAR_MLP_TRANSFORMER_V1_PRESET,
    ])
    startup = _collect_launcher_startup_status(launch)
    assert {
        spec.num_epochs
        for spec in startup.bootstrap_config.evaluators.evaluators.values()
    } == {5}
    assert {
        spec.num_epochs
        for spec in startup.resolved_bootstrap_args.resolved_evaluators_config().evaluators.values()
    } == {5}

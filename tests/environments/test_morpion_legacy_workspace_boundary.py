"""Legacy APIs must reject scientific authority before any workspace mutation."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from chipiron.environments.morpion.bootstrap import (
    MorpionBootstrapArgs,
    MorpionBootstrapPaths,
    initialize_bootstrap_run_state,
    load_bootstrap_run_state,
    run_morpion_bootstrap_loop,
    run_one_bootstrap_cycle,
    save_bootstrap_run_state,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    PUBLICATION_KEY,
    GenerationPublicationError,
    publication_namespace,
)
from chipiron.environments.morpion.bootstrap.search_runner_protocol import (
    MorpionSearchRunner,
)
from tests.environments.test_morpion_continual_publication import setup_continual

if TYPE_CHECKING:
    from pathlib import Path


def _inventory(root: Path) -> dict[str, tuple[bytes | None, int]]:
    """Include all authority, artifacts, directories and modification times."""
    return {
        path.relative_to(root).as_posix(): (
            path.read_bytes() if path.is_file() else None,
            path.stat().st_mtime_ns,
        )
        for path in [root, *root.rglob("*")]
    }


@pytest.mark.fast
@pytest.mark.parametrize("entrypoint", ["loop_zero", "loop_one", "cycle"])
def test_legacy_entrypoints_preserve_scientific_workspace(
    tmp_path: Path, entrypoint: str
) -> None:
    """Default single-process arguments cannot overwrite real scientific genesis."""
    _, paths = setup_continual(tmp_path / "scientific")
    before = _inventory(paths.work_dir)
    args = MorpionBootstrapArgs(work_dir=paths.work_dir)
    assert args.pipeline_mode == "single_process"
    runner = Mock(spec=MorpionSearchRunner)

    with pytest.raises(GenerationPublicationError, match=r"legacy.*scientific"):
        if entrypoint == "cycle":
            run_one_bootstrap_cycle(
                args=args,
                paths=paths,
                runner=runner,
                # A stale caller snapshot must not bypass persisted authority.
                run_state=initialize_bootstrap_run_state(),
            )
        else:
            run_morpion_bootstrap_loop(
                args, runner, max_cycles=0 if entrypoint == "loop_zero" else 1
            )

    assert runner.mock_calls == []
    assert _inventory(paths.work_dir) == before


@pytest.mark.fast
@pytest.mark.parametrize("version", [1, 2, 3])
@pytest.mark.parametrize("entrypoint", ["loop_zero", "cycle"])
def test_legacy_entrypoints_preserve_interrupted_publication(
    tmp_path: Path, version: int, entrypoint: str
) -> None:
    """Incomplete scientific genesis without run state is not a legacy workspace."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    attempt = tmp_path / publication_namespace(version) / "staging" / "attempt"
    attempt.mkdir(parents=True)
    (attempt / "checkpoint").write_bytes(b"interrupted scientific payload")
    paths.bootstrap_config_path.write_bytes(b"preserve even an incomplete config")
    before = _inventory(tmp_path)
    runner = Mock(spec=MorpionSearchRunner)
    args = MorpionBootstrapArgs(work_dir=tmp_path)

    with pytest.raises(GenerationPublicationError, match=r"legacy.*scientific"):
        if entrypoint == "cycle":
            run_one_bootstrap_cycle(
                args=args,
                paths=paths,
                runner=runner,
                run_state=initialize_bootstrap_run_state(),
            )
        else:
            run_morpion_bootstrap_loop(args, runner, max_cycles=0)

    assert runner.mock_calls == []
    assert _inventory(tmp_path) == before


@pytest.mark.fast
@pytest.mark.parametrize("persisted", [False, True])
def test_legacy_cycle_rejects_publication_marker_without_payload(
    tmp_path: Path, persisted: bool
) -> None:
    """A damaged or caller-provided publication marker cannot become legacy state."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    state = replace(initialize_bootstrap_run_state(), metadata={PUBLICATION_KEY: None})
    if persisted:
        save_bootstrap_run_state(state, paths.run_state_path)
        state = initialize_bootstrap_run_state()
    before = _inventory(tmp_path)
    runner = Mock(spec=MorpionSearchRunner)

    with pytest.raises(GenerationPublicationError, match=r"legacy.*scientific"):
        run_one_bootstrap_cycle(
            args=MorpionBootstrapArgs(work_dir=tmp_path),
            paths=paths,
            runner=runner,
            run_state=state,
        )

    assert runner.mock_calls == []
    assert _inventory(tmp_path) == before


@pytest.mark.fast
def test_zero_cycle_legacy_loop_remains_supported(tmp_path: Path) -> None:
    """Existing genuine legacy workspaces keep their ordinary zero-cycle behavior."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    state = initialize_bootstrap_run_state()
    save_bootstrap_run_state(state, paths.run_state_path)
    before = paths.run_state_path.read_bytes()
    runner = Mock(spec=MorpionSearchRunner)

    result = run_morpion_bootstrap_loop(
        MorpionBootstrapArgs(work_dir=tmp_path), runner, max_cycles=0
    )

    assert result.generation == state.generation
    assert paths.bootstrap_config_path.is_file()
    assert paths.run_state_path.read_bytes() == before
    assert load_bootstrap_run_state(paths.run_state_path) == state
    assert runner.mock_calls == []

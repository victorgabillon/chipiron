"""Morpion runner checkpoint parity through the published Anemone boundary."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest
from anemone.progress_monitor.progress_monitor import TreeBranchLimitArgs

from chipiron.environments.morpion.bootstrap.runtime.runner import (
    AnemoneMorpionSearchRunner,
    AnemoneMorpionSearchRunnerArgs,
    default_search_args,
)

if TYPE_CHECKING:
    from pathlib import Path


def _selection_signature(runner: AnemoneMorpionSearchRunner) -> tuple[Any, ...]:
    """Read the actual selector report while excluding timings and cache details."""
    report = runner._require_runtime().node_selector.base.latest_selection_report
    assert report is not None
    return (
        report.selected_node_id,
        report.selected_depth,
        report.depth_selection_policy,
        report.depth_selection_subpolicy,
        report.depth_selection_step,
        report.depth_selection_step_parity,
        report.selected_depth_selection_index,
        report.selected_depth_selection_weight,
        report.selected_depth_selection_probability,
        report.depth_rows,
    )


@pytest.mark.parametrize("save_after", [1, 2])
@pytest.mark.parametrize("suffix", ["json", "json.zst", "json.gz", "sharded"])
def test_runner_checkpoint_preserves_selection_sequence(
    tmp_path: Path, save_after: int, suffix: str
) -> None:
    """Real file and streaming restores preserve the next policies, nodes and RNG."""
    search_args = replace(
        default_search_args(),
        stopping_criterion=TreeBranchLimitArgs(
            type="tree_branch_limit", tree_branch_limit=4096
        ),
    )
    args = AnemoneMorpionSearchRunnerArgs(
        search_args=search_args,
        random_seed=4,
        runtime_checkpoint_format="sharded" if suffix == "sharded" else "json-zst",
    )
    continuous = AnemoneMorpionSearchRunner(args)
    continuous.load_or_create(None, None)
    continuous.grow(save_after)
    assert _selection_signature(continuous)[4] == save_after
    path = tmp_path / f"generation_000001.{suffix}"
    continuous.save_checkpoint(path)

    # An unrelated constructor seed must be replaced by the saved RNG state.
    restored = AnemoneMorpionSearchRunner(replace(args, random_seed=999))
    restored.load_or_create(path, None)
    assert (
        restored._random_generator.getstate() == continuous._random_generator.getstate()
    )
    assert restored.current_tree_size() == continuous.current_tree_size()
    for expected_step in range(save_after + 1, save_after + 3):
        continuous.grow(1)
        restored.grow(1)
        actual = _selection_signature(restored)
        assert actual == _selection_signature(continuous)
        assert actual[2] == "alternating_by_step"
        assert actual[3] == (
            "inverse_depth" if expected_step % 2 else "opened_count_depth_index"
        )
        assert actual[4] == expected_step
        assert actual[5] == ("odd" if expected_step % 2 else "even")
        assert restored.current_tree_size() == continuous.current_tree_size()
        assert (
            restored._random_generator.getstate()
            == continuous._random_generator.getstate()
        )

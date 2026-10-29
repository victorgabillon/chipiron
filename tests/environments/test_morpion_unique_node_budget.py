"""Whole-expansion node limits agree with an unrestricted continuation oracle."""

import pytest
from anemone.checkpoints import (
    build_search_checkpoint_payload,
    checkpoint_payload_to_jsonable,
)

from chipiron.environments.morpion.bootstrap.config import MorpionBootstrapRolloutConfig
from chipiron.environments.morpion.bootstrap.runtime.runner import (
    AnemoneMorpionSearchRunner,
    AnemoneMorpionSearchRunnerArgs,
    default_search_args,
)


def payload(runner: AnemoneMorpionSearchRunner) -> object:
    """Capture every checkpoint field for the independent unrestricted oracle."""
    return checkpoint_payload_to_jsonable(
        build_search_checkpoint_payload(
            runner._require_runtime(), state_codec=runner._state_codec
        )
    )


def runner(extra: int | None, maximum: int | None) -> AnemoneMorpionSearchRunner:
    """Construct an ordinary real runtime with an optional hard node cap."""
    value = AnemoneMorpionSearchRunner(
        AnemoneMorpionSearchRunnerArgs(
            runtime_checkpoint_format="sharded",
            max_tree_nodes=maximum,
            search_args=default_search_args(
                rollout=None
                if extra is None
                else MorpionBootstrapRolloutConfig(
                    enabled=True, max_extra_steps=extra, random_seed=23
                )
            ),
        )
    )
    value.load_or_create(None, None)
    return value


@pytest.mark.fast
@pytest.mark.parametrize("maximum", [1, 4, 5, 20, 100])
@pytest.mark.parametrize("extra", [None, 2])
def test_budget_commits_last_complete_expansion(
    maximum: int, extra: int | None
) -> None:
    """The bounded graph and all continuation state equal the last safe prefix."""
    reference = runner(extra, None)
    expected = payload(reference)
    for _ in range(32):
        reference.grow(1)
        if reference.current_tree_size() > maximum:
            break
        expected = payload(reference)
        if reference.current_tree_size() == maximum:
            break
    bounded = runner(extra, maximum)
    try:
        bounded.grow(32)
        assert bounded.current_tree_size() <= maximum
        actual = payload(bounded)
        assert actual == expected
        assert bounded.last_node_budget_stop_reason is not None
    finally:
        bounded.discard_uncommitted_runtime()
        reference.discard_uncommitted_runtime()

"""Runtime-control helpers for Morpion bootstrap search runtime."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Protocol, cast

from anemone.progress_monitor.progress_monitor import (
    TreeBranchLimit,
    TreeBranchLimitArgs,
)

from chipiron.environments.morpion.bootstrap.control import (
    MorpionBootstrapEffectiveRuntimeConfig,
)

__all__ = [
    "apply_runtime_config_to_runtime",
    "apply_runtime_control_to_runner_args",
    "runtime_config_from_search_args",
    "search_args_with_tree_branch_limit",
]

if TYPE_CHECKING:
    from typing import Any

    from anemone.factory import SearchArgs

_TREE_BRANCH_LIMIT_ARGS_REQUIRED_MESSAGE = (
    "Morpion bootstrap runtime reconfiguration currently supports only "
    "TreeBranchLimitArgs stopping criteria."
)
_LIVE_TREE_BRANCH_LIMIT_REQUIRED_MESSAGE = (
    "Morpion bootstrap runtime reconfiguration currently supports only "
    "tree-branch-limit stopping criteria on the live runtime."
)


class _RunnerArgsWithSearchArgs(Protocol):
    @property
    def search_args(self) -> SearchArgs:
        """Return the Anemone search args held by this runner-args object."""
        ...


def apply_runtime_control_to_runner_args[RunnerArgsT: _RunnerArgsWithSearchArgs](
    runner_args: RunnerArgsT,
    runtime_config: MorpionBootstrapEffectiveRuntimeConfig,
) -> RunnerArgsT:
    """Return runner args rebound to one effective runtime config.

    This helper is kept as the pure arg-transformation counterpart of the live
    runtime patching path used during checkpoint restore.
    """
    updated_runner_args = replace(
        cast("Any", runner_args),
        search_args=search_args_with_tree_branch_limit(
            runner_args.search_args,
            tree_branch_limit=runtime_config.tree_branch_limit,
        ),
    )
    return cast("RunnerArgsT", updated_runner_args)


def runtime_config_from_search_args(
    search_args: SearchArgs,
) -> MorpionBootstrapEffectiveRuntimeConfig:
    """Extract the supported effective runtime config from one SearchArgs object."""
    stopping_criterion = search_args.stopping_criterion
    if not isinstance(stopping_criterion, TreeBranchLimitArgs):
        raise TypeError(_missing_tree_branch_limit_args_error())
    return MorpionBootstrapEffectiveRuntimeConfig(
        tree_branch_limit=stopping_criterion.tree_branch_limit,
    )


def search_args_with_tree_branch_limit(
    search_args: SearchArgs,
    *,
    tree_branch_limit: int,
) -> SearchArgs:
    """Return SearchArgs rebound to one explicit tree-branch limit."""
    stopping_criterion = search_args.stopping_criterion
    if not isinstance(stopping_criterion, TreeBranchLimitArgs):
        raise TypeError(_missing_tree_branch_limit_args_error())
    return replace(
        search_args,
        stopping_criterion=replace(
            stopping_criterion,
            tree_branch_limit=tree_branch_limit,
        ),
    )


def apply_runtime_config_to_runtime(
    runtime: object,
    runtime_config: MorpionBootstrapEffectiveRuntimeConfig,
) -> None:
    """Apply the supported runtime config to one live runtime after create/restore."""
    stopping_criterion = getattr(runtime, "stopping_criterion", None)
    if not isinstance(stopping_criterion, TreeBranchLimit):
        raise TypeError(_unsupported_runtime_stopping_criterion_error())
    stopping_criterion.tree_branch_limit = runtime_config.tree_branch_limit


def _missing_tree_branch_limit_args_error() -> str:
    return _TREE_BRANCH_LIMIT_ARGS_REQUIRED_MESSAGE


def _unsupported_runtime_stopping_criterion_error() -> str:
    return _LIVE_TREE_BRANCH_LIMIT_REQUIRED_MESSAGE

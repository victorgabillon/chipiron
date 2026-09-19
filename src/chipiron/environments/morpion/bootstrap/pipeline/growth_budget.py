"""Growth-budget helpers for Morpion artifact-pipeline stages."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chipiron.environments.morpion.bootstrap.bootstrap_args import (
        MorpionBootstrapArgs,
    )
    from chipiron.environments.morpion.bootstrap.control import (
        MorpionBootstrapEffectiveRuntimeConfig,
    )

LOGGER = logging.getLogger(__name__)


def apply_effective_runtime_config_if_supported(
    runner: object,
    runtime_config: MorpionBootstrapEffectiveRuntimeConfig,
) -> None:
    """Patch the live runner runtime when it exposes the optional hook."""
    apply_config = getattr(runner, "apply_effective_runtime_config", None)
    if callable(apply_config):
        apply_config(runtime_config)


def missing_branch_count_for_additional_budget_error() -> RuntimeError:
    """Return the stable error for unresolved additional branch budgets."""
    return RuntimeError(
        "Cannot apply --growth-additional-branch-budget because the runner "
        "does not expose a current branch count after restore."
    )


def growth_budget_runtime_config(
    *,
    args: MorpionBootstrapArgs,
    runner: object,
    current_branch_count: int | None,
    effective_runtime_config: MorpionBootstrapEffectiveRuntimeConfig,
) -> tuple[MorpionBootstrapEffectiveRuntimeConfig, dict[str, object]]:
    """Resolve the effective branch-limit budget for one growth cycle."""
    if args.growth_additional_branch_budget is None:
        LOGGER.info(
            "[growth-budget] mode=absolute effective_branch_limit=%s",
            effective_runtime_config.tree_branch_limit,
        )
        return (
            effective_runtime_config,
            {
                "growth_budget_mode": "absolute",
                "branch_count_before_growth": current_branch_count,
                "growth_additional_branch_budget": None,
                "effective_branch_limit": effective_runtime_config.tree_branch_limit,
            },
        )
    if current_branch_count is None:
        raise missing_branch_count_for_additional_budget_error()
    effective_branch_limit = current_branch_count + args.growth_additional_branch_budget
    resolved_runtime_config = replace(
        effective_runtime_config,
        tree_branch_limit=effective_branch_limit,
    )
    apply_effective_runtime_config_if_supported(runner, resolved_runtime_config)
    LOGGER.info(
        "[growth-budget] mode=additional current_branches=%s additional=%s effective_branch_limit=%s",
        current_branch_count,
        args.growth_additional_branch_budget,
        effective_branch_limit,
    )
    return (
        resolved_runtime_config,
        {
            "growth_budget_mode": "additional",
            "branch_count_before_growth": current_branch_count,
            "growth_additional_branch_budget": args.growth_additional_branch_budget,
            "effective_branch_limit": effective_branch_limit,
        },
    )

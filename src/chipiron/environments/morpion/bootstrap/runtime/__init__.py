"""Stable Morpion bootstrap runtime APIs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .runner import (
        AnemoneMorpionSearchRunner,
        AnemoneMorpionSearchRunnerArgs,
        MorpionRegressorMasterEvaluator,
        run_morpion_growth_search_once,
    )

__all__ = [
    "AnemoneMorpionSearchRunner",
    "AnemoneMorpionSearchRunnerArgs",
    "MorpionRegressorMasterEvaluator",
    "run_morpion_growth_search_once",
]


def __getattr__(name: str) -> object:
    """Lazily expose runner APIs without loading runner for helper submodules."""
    if name in __all__:
        from . import runner  # pylint: disable=import-outside-toplevel

        return getattr(runner, name)
    raise AttributeError(name)

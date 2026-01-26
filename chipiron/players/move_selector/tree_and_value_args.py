"""Tree-and-value move selector args with game-specific evaluator inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from anemone import TreeAndValuePlayerArgs as AnemoneTreeArgs
from valanga import TurnState

StateT = TypeVar("StateT", bound=TurnState)
EvalArgsT = TypeVar("EvalArgsT")


@dataclass(frozen=True)
class TreeAndValueAppArgs(Generic[StateT, EvalArgsT]):
    """Generic wrapper for tree-and-value settings plus evaluator args."""

    anemone_args: AnemoneTreeArgs
    evaluator_args: EvalArgsT

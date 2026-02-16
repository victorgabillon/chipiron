"""Composable recommendation modifiers for branch selectors."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeVar, cast

from valanga import BranchKey, Color, Dynamics, TurnState
from valanga.evaluations import FloatyStateEvaluation, ForcedOutcome, StateEvaluation
from valanga.game import BranchName, Seed
from valanga.over_event import HowOver
from valanga.policy import BranchSelector, NotifyProgressCallable, Recommendation

TurnStateT = TypeVar("TurnStateT", bound=TurnState)


class RecommendationModifier[StateT: TurnState](Protocol):
    """Protocol for recommendation post-processors."""

    @property
    def name(self) -> str:
        """Return the name of the recommendation modifier."""
        ...

    def modify(
        self,
        state: StateT,
        rec: Recommendation,
        seed: Seed,
        dynamics: Dynamics[StateT],
    ) -> Recommendation | None:
        """Return an overridden recommendation or ``None`` to keep the original."""


class HasZeroing(Protocol):
    """Capability protocol for states exposing chess zeroing detection."""

    def is_zeroing(self, move: BranchKey) -> bool:
        """Return whether a move is a zeroing move."""
        ...


@dataclass(frozen=True, slots=True)
class ComposedBranchSelector[StateT: TurnState](BranchSelector[StateT]):
    """Branch selector that applies recommendation modifiers in sequence."""

    base: BranchSelector[StateT]
    dynamics: Dynamics[StateT]
    modifiers: tuple[RecommendationModifier[StateT], ...] = ()

    def recommend(
        self,
        state: StateT,
        seed: Seed,
        notify_progress: NotifyProgressCallable | None = None,
    ) -> Recommendation:
        """Recommend from base selector and then apply modifiers in order."""
        recommendation = self.base.recommend(
            state=state,
            seed=seed,
            notify_progress=notify_progress,
        )
        for modifier in self.modifiers:
            overridden = modifier.modify(
                state=state,
                rec=recommendation,
                seed=seed,
                dynamics=self.dynamics,
            )
            if overridden is not None:
                recommendation = overridden
        return recommendation


def is_winning_eval(
    evaluation: StateEvaluation,
    player: Color,
    threshold: float = 0.98,
) -> bool:
    """Return whether ``evaluation`` is winning for ``player``."""
    match evaluation:
        case FloatyStateEvaluation(value_white=value_white):
            if value_white is None:
                return False
            if player is Color.WHITE:
                return value_white > threshold
            return value_white < -threshold
        case ForcedOutcome(outcome=outcome):
            return outcome.how_over is HowOver.WIN and outcome.is_winner(player)


ProgressGainFn = Callable[[TurnState, BranchKey], float]


@dataclass(frozen=True, slots=True)
class AccelerateWhenWinning[StateT: TurnState]:
    """Prefer winning branches that make more game progress."""

    progress_gain_fn: ProgressGainFn
    name: str = "accelerate_when_winning"
    threshold: float = 0.98

    def modify(
        self,
        state: StateT,
        rec: Recommendation,
        seed: Seed,
        dynamics: Dynamics[StateT],
    ) -> Recommendation | None:
        """Return an overridden recommendation that prefers winning branches with more progress."""
        _ = seed  # This modifier does not use randomness.

        if rec.evaluation is None or rec.branch_evals is None:
            return None
        if not is_winning_eval(rec.evaluation, state.turn, threshold=self.threshold):
            return None

        best_branch: BranchName | None = None
        best_gain: float = 0.0

        for branch_name, child_eval in rec.branch_evals.items():
            if not is_winning_eval(child_eval, state.turn, threshold=self.threshold):
                continue

            branch_key = dynamics.action_from_name(state, branch_name)
            _ = dynamics.step(state, branch_key).next_state

            gain: float = self.progress_gain_fn(state, branch_key)
            if gain > best_gain:
                best_gain = gain
                best_branch = branch_name

        if best_branch is None:
            return None

        return Recommendation(
            recommended_name=best_branch,
            evaluation=rec.evaluation,
            policy=rec.policy,
            branch_evals=rec.branch_evals,
        )


def chess_progress_gain_zeroing(state: TurnState, branch_key: BranchKey) -> float:
    """Progress gain function that returns 1.0 for zeroing moves and 0.0 for non-zeroing moves."""
    s = cast("HasZeroing", state)
    return 1.0 if s.is_zeroing(branch_key) else 0.0

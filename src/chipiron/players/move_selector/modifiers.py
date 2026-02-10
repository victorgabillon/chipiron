"""Composable recommendation modifiers for branch selectors."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeVar

from valanga import BranchKey, BranchName, Color, StateModifications, TurnState
from valanga.evaluations import FloatyStateEvaluation, ForcedOutcome, StateEvaluation
from valanga.over_event import HowOver
from valanga.policy import BranchSelector, NotifyProgressCallable, Recommendation

if TYPE_CHECKING:
    from valanga.game import Seed

TurnStateT = TypeVar("TurnStateT", bound=TurnState)


class RecommendationModifier[StateT: TurnState](Protocol):
    """Protocol for recommendation post-processors."""

    name: str

    def modify(
        self,
        state: StateT,
        rec: Recommendation,
        seed: Seed,
    ) -> Recommendation | None:
        """Return an overridden recommendation or ``None`` to keep the original."""


class HasZeroing(Protocol):
    """Capability protocol for states exposing chess zeroing detection."""

    def is_zeroing(self, move: BranchKey) -> bool:
        """Return whether a move is a zeroing move."""


@dataclass(frozen=True, slots=True)
class ComposedBranchSelector[StateT: TurnState](BranchSelector[StateT]):
    """Branch selector that applies recommendation modifiers in sequence."""

    base: BranchSelector[StateT]
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
            overridden = modifier.modify(state=state, rec=recommendation, seed=seed)
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
        case _:
            return False


type ProgressGainFn[StateT: TurnState] = Callable[
    [StateT, BranchName, BranchKey, StateT, StateModifications],
    float,
]


@dataclass(frozen=True, slots=True)
class AccelerateWhenWinning[StateT: TurnState]:
    """Prefer winning branches that make more game progress."""

    progress_gain_fn: ProgressGainFn[StateT]
    name: str = "accelerate_when_winning"
    threshold: float = 0.98
    copy_stack: bool = False
    deep_copy_legal_moves: bool = True

    def modify(
        self,
        state: StateT,
        rec: Recommendation,
        seed: Seed,
    ) -> Recommendation | None:
        """Override recommendation when winning and a progress-improving move exists."""
        if state.is_game_over():
            return None
        if rec.evaluation is None or rec.branch_evals is None:
            return None

        if not is_winning_eval(rec.evaluation, state.turn, threshold=self.threshold):
            return None

        best_branch: BranchName | None = None
        best_gain: float = 0.0

        for branch_name, child_eval in rec.branch_evals.items():
            if not is_winning_eval(child_eval, state.turn, threshold=self.threshold):
                continue
            branch_key = state.branch_key_from_name(name=branch_name)
            child_state = state.copy(
                stack=self.copy_stack,
                deep_copy_legal_moves=self.deep_copy_legal_moves,
            )
            mods = child_state.step(branch_key)
            if mods is None:
                continue
            gain = self.progress_gain_fn(
                state,
                branch_name,
                branch_key,
                child_state,
                mods,
            )
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


def chess_progress_gain_zeroing(
    state: HasZeroing,
    branch_name: BranchName,
    branch_key: BranchKey,
    child_state: TurnState,
    mods: StateModifications,
) -> float:
    """Chess progress proxy: prefer zeroing moves (captures/pawn moves)."""
    return 1.0 if state.is_zeroing(branch_key) else 0.0

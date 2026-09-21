"""Cheap current-state latent-window geometry for Morpion move tokens."""
# ruff: noqa: TC001, TC003, TRY003
# pyright: reportMissingImports=false

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from atomheart.games.morpion.dynamics import DIRECTIONS
from atomheart.games.morpion.state import Variant

from chipiron.environments.morpion.action_geometry import (
    morpion_action_new_point,
    morpion_action_points,
    morpion_action_segments,
)
from chipiron.environments.morpion.types import MorpionAction, MorpionState

MORPION_LATENT_WINDOW_PROXY_VERSION: Final[str] = (
    "morpion_current_state_latent_window_proxy_v1"
)
MORPION_CORRECTED_BLOCKING_PROXY_VERSION: Final[str] = "morpion_latent_window_proxy_v2"
MORPION_LATENT_WINDOW_IMPLEMENTATION: Final[str] = "batched_state_index_v1"


@dataclass(slots=True)
class MorpionLatentWindowInstrumentation:
    """Optional per-conversion operation counts without global mutable state."""

    state_context_build_count: int = 0
    batched_feature_call_count: int = 0
    scalar_reference_call_count: int = 0
    canonical_windows_enumerated: int = 0
    window_compatibility_checks: int = 0
    played_action_segment_builds: int = 0


@dataclass(frozen=True, slots=True)
class _PreparedLatentWindow:
    """One occupancy-qualified window with current compatibility precomputed."""

    identity: LatentWindowIdentity
    candidate_segments: frozenset[tuple[tuple[int, int], tuple[int, int]]]


@dataclass(frozen=True, slots=True)
class MorpionLatentWindowFeatureContext:
    """Immutable state-derived index used for one ephemeral token conversion."""

    state_fingerprint: str
    occupied_points: frozenset[tuple[int, int]]
    used_unit_segments: frozenset[tuple[tuple[int, int], tuple[int, int]]]
    variant: Variant
    direction_usage: Mapping[tuple[tuple[int, int], int], int]
    legal_actions: tuple[MorpionAction, ...]
    windows_by_new_dot: Mapping[tuple[int, int], tuple[_PreparedLatentWindow, ...]]


@dataclass(frozen=True, order=True, slots=True)
class LatentWindowIdentity:
    """Canonical identity of one five-point window and its remaining gap."""

    direction: int
    origin_x: int
    origin_y: int
    missing_slot_after_play: int

    @property
    def action(self) -> MorpionAction:
        """Return the action geometry the window could promote."""
        return (
            self.direction,
            self.origin_x,
            self.origin_y,
            self.missing_slot_after_play,
        )


@dataclass(frozen=True, slots=True)
class LatentWindowMoveFeatureCounts:
    """Primitive counts derived without transitions or successor enumeration."""

    three_of_five_window_count: int
    promoted_latent_window_count: int
    blocked_by_consumed_segment_count: int
    blocked_by_same_direction_touching_count: int
    potential_windows: tuple[LatentWindowIdentity, ...]
    promoted_windows: tuple[LatentWindowIdentity, ...]
    segment_blocked_windows: tuple[LatentWindowIdentity, ...]
    touching_blocked_windows: tuple[LatentWindowIdentity, ...]


@dataclass(frozen=True, slots=True)
class PotentialLatentWindowMechanism:
    """Independent static and played-action outcomes for one potential window."""

    identity: LatentWindowIdentity
    current_segments_available: bool
    current_same_direction_compatible: bool
    blocked_by_consumed_segment: bool
    blocked_by_new_same_direction_touching: bool
    promoted: bool

    @property
    def category(self) -> str:
        """Return the played-action blocking overlap category."""
        if (
            self.blocked_by_consumed_segment
            and self.blocked_by_new_same_direction_touching
        ):
            return "both"
        if self.blocked_by_consumed_segment:
            return "segment_only"
        if self.blocked_by_new_same_direction_touching:
            return "touching_only"
        return "neither"


@dataclass(frozen=True, slots=True)
class CorrectedLatentWindowMoveFeatureCounts:
    """Occupancy-first potential windows and independently classified outcomes."""

    potential_window_count: int
    promoted_latent_window_count: int
    blocked_by_consumed_segment_count: int
    blocked_by_same_direction_touching_count: int
    segment_only_count: int
    touching_only_count: int
    both_count: int
    neither_count: int
    mechanisms: tuple[PotentialLatentWindowMechanism, ...]


def latent_window_move_feature_counts(
    state: MorpionState,
    action: MorpionAction,
) -> LatentWindowMoveFeatureCounts:
    """Return cheap latent-window counts from current geometry only.

    A potential window has three occupied points and two gaps, one of which is
    the played action's new dot. Windows already incompatible with the current
    state's used segments or same-direction usage are excluded. A promoted
    window is potential and remains compatible after accounting geometrically
    for the played line's four segments and direction usage.

    The two blocked counts are independent: one window may contribute once to
    each when both mechanisms apply.
    """
    new_dot = morpion_action_new_point(action)
    played_segments = frozenset(morpion_action_segments(action))
    potential: set[LatentWindowIdentity] = set()
    promoted: set[LatentWindowIdentity] = set()
    segment_blocked: set[LatentWindowIdentity] = set()
    touching_blocked: set[LatentWindowIdentity] = set()
    for direction, (dx, dy) in enumerate(DIRECTIONS):
        for new_dot_slot in range(5):
            origin_x = new_dot[0] - new_dot_slot * dx
            origin_y = new_dot[1] - new_dot_slot * dy
            probe: MorpionAction = (direction, origin_x, origin_y, 0)
            points = morpion_action_points(probe)
            missing_before = tuple(
                slot for slot, point in enumerate(points) if point not in state.points
            )
            if len(missing_before) != 2 or new_dot_slot not in missing_before:
                continue
            remaining_slot = next(
                slot for slot in missing_before if slot != new_dot_slot
            )
            identity = LatentWindowIdentity(
                direction=direction,
                origin_x=origin_x,
                origin_y=origin_y,
                missing_slot_after_play=remaining_slot,
            )
            candidate = identity.action
            candidate_segments = frozenset(morpion_action_segments(candidate))
            if candidate_segments & state.used_unit_segments:
                continue
            if not _parallel_compatible_before_play(state, candidate):
                continue
            potential.add(identity)
            blocked_segment = bool(candidate_segments & played_segments)
            blocked_touching = not _parallel_compatible_after_play_geometry(
                state=state,
                candidate=candidate,
                played_action=action,
            )
            if blocked_segment:
                segment_blocked.add(identity)
            if blocked_touching:
                touching_blocked.add(identity)
            if not blocked_segment and not blocked_touching:
                promoted.add(identity)
    return LatentWindowMoveFeatureCounts(
        three_of_five_window_count=len(potential),
        promoted_latent_window_count=len(promoted),
        blocked_by_consumed_segment_count=len(segment_blocked),
        blocked_by_same_direction_touching_count=len(touching_blocked),
        potential_windows=tuple(sorted(potential)),
        promoted_windows=tuple(sorted(promoted)),
        segment_blocked_windows=tuple(sorted(segment_blocked)),
        touching_blocked_windows=tuple(sorted(touching_blocked)),
    )


def compute_promoted_latent_window_count_reference(
    state: MorpionState,
    action: MorpionAction,
    *,
    instrumentation: MorpionLatentWindowInstrumentation | None = None,
) -> LatentWindowMoveFeatureCounts:
    """Run the retained scalar oracle; production conversion must not call this."""
    if instrumentation is not None:
        instrumentation.scalar_reference_call_count += 1
    return latent_window_move_feature_counts(state, action)


def build_morpion_latent_window_feature_context(
    state: MorpionState,
    legal_actions: Sequence[MorpionAction],
    *,
    instrumentation: MorpionLatentWindowInstrumentation | None = None,
) -> MorpionLatentWindowFeatureContext:
    """Build one immutable occupancy/compatibility index for a converted state."""
    ordered_actions = tuple(sorted(legal_actions))
    if len(ordered_actions) != len(set(ordered_actions)):
        raise ValueError("Latent-window context received duplicate action identities.")
    if instrumentation is not None:
        instrumentation.state_context_build_count += 1
    occupied = frozenset(state.points)
    used_segments = frozenset(state.used_unit_segments)
    direction_usage = MappingProxyType(dict(state.dir_usage))
    state_fingerprint = _latent_window_state_fingerprint(state, ordered_actions)
    context_shell = MorpionLatentWindowFeatureContext(
        state_fingerprint=state_fingerprint,
        occupied_points=occupied,
        used_unit_segments=used_segments,
        variant=state.variant,
        direction_usage=direction_usage,
        legal_actions=ordered_actions,
        windows_by_new_dot=MappingProxyType({}),
    )
    new_dots = frozenset(morpion_action_new_point(action) for action in ordered_actions)
    prepared_by_new_dot: dict[tuple[int, int], list[_PreparedLatentWindow]] = {
        new_dot: [] for new_dot in new_dots
    }
    candidate_windows = {
        (direction, new_dot[0] - slot * dx, new_dot[1] - slot * dy)
        for new_dot in new_dots
        for direction, (dx, dy) in enumerate(DIRECTIONS)
        for slot in range(5)
    }
    for direction, origin_x, origin_y in sorted(candidate_windows):
        if instrumentation is not None:
            instrumentation.canonical_windows_enumerated += 1
        probe: MorpionAction = (direction, origin_x, origin_y, 0)
        points = morpion_action_points(probe)
        missing_before = tuple(
            slot for slot, point in enumerate(points) if point not in occupied
        )
        if len(missing_before) != 2:
            continue
        for new_dot_slot in missing_before:
            new_dot = points[new_dot_slot]
            if new_dot not in new_dots:
                continue
            remaining_slot = next(
                slot for slot in missing_before if slot != new_dot_slot
            )
            identity = LatentWindowIdentity(
                direction=direction,
                origin_x=origin_x,
                origin_y=origin_y,
                missing_slot_after_play=remaining_slot,
            )
            candidate_segments = frozenset(morpion_action_segments(identity.action))
            if candidate_segments & used_segments:
                continue
            if instrumentation is not None:
                instrumentation.window_compatibility_checks += 1
            if not _parallel_compatible_with_context(
                context_shell,
                identity.action,
                extra_direction=direction,
                extra_points=(),
            ):
                continue
            prepared_by_new_dot[new_dot].append(
                _PreparedLatentWindow(
                    identity=identity,
                    candidate_segments=candidate_segments,
                )
            )
    windows_by_new_dot = {
        new_dot: tuple(sorted(prepared, key=lambda item: item.identity))
        for new_dot, prepared in sorted(prepared_by_new_dot.items())
    }
    return MorpionLatentWindowFeatureContext(
        state_fingerprint=state_fingerprint,
        occupied_points=occupied,
        used_unit_segments=used_segments,
        variant=state.variant,
        direction_usage=direction_usage,
        legal_actions=ordered_actions,
        windows_by_new_dot=MappingProxyType(windows_by_new_dot),
    )


def compute_promoted_latent_window_counts(
    *,
    context: MorpionLatentWindowFeatureContext,
    legal_actions: Sequence[MorpionAction],
    state: MorpionState | None = None,
    instrumentation: MorpionLatentWindowInstrumentation | None = None,
) -> Mapping[MorpionAction, LatentWindowMoveFeatureCounts]:
    """Compute exactly one proxy-v1 feature record per canonical legal action."""
    ordered_actions = tuple(sorted(legal_actions))
    if len(ordered_actions) != len(set(ordered_actions)):
        raise ValueError("Batched latent-window input contains duplicate actions.")
    if ordered_actions != context.legal_actions:
        raise ValueError(
            "Latent-window context is incompatible with the supplied state actions."
        )
    if (
        state is not None
        and _latent_window_state_fingerprint(state, ordered_actions)
        != context.state_fingerprint
    ):
        raise ValueError("Latent-window context has an incompatible state fingerprint.")
    if instrumentation is not None:
        instrumentation.batched_feature_call_count += 1
    result: dict[MorpionAction, LatentWindowMoveFeatureCounts] = {}
    for action in ordered_actions:
        new_dot = morpion_action_new_point(action)
        prepared_windows = context.windows_by_new_dot.get(new_dot)
        if prepared_windows is None:
            raise ValueError(f"Missing latent-window index for action {action!r}.")
        played_segments = frozenset(morpion_action_segments(action))
        if instrumentation is not None:
            instrumentation.played_action_segment_builds += 1
        potential: list[LatentWindowIdentity] = []
        promoted: list[LatentWindowIdentity] = []
        segment_blocked: list[LatentWindowIdentity] = []
        touching_blocked: list[LatentWindowIdentity] = []
        played_direction = action[0]
        played_points = morpion_action_points(action)
        for prepared in prepared_windows:
            potential.append(prepared.identity)
            blocked_segment = bool(prepared.candidate_segments & played_segments)
            if instrumentation is not None:
                instrumentation.window_compatibility_checks += 1
            blocked_touching = not _parallel_compatible_with_context(
                context,
                prepared.identity.action,
                extra_direction=played_direction,
                extra_points=played_points,
            )
            if blocked_segment:
                segment_blocked.append(prepared.identity)
            if blocked_touching:
                touching_blocked.append(prepared.identity)
            if not blocked_segment and not blocked_touching:
                promoted.append(prepared.identity)
        result[action] = LatentWindowMoveFeatureCounts(
            three_of_five_window_count=len(potential),
            promoted_latent_window_count=len(promoted),
            blocked_by_consumed_segment_count=len(segment_blocked),
            blocked_by_same_direction_touching_count=len(touching_blocked),
            potential_windows=tuple(potential),
            promoted_windows=tuple(promoted),
            segment_blocked_windows=tuple(segment_blocked),
            touching_blocked_windows=tuple(touching_blocked),
        )
    if result.keys() != set(ordered_actions):
        raise ValueError("Batched latent-window output is missing action results.")
    return MappingProxyType(result)


def corrected_latent_window_move_feature_counts(
    state: MorpionState,
    action: MorpionAction,
) -> CorrectedLatentWindowMoveFeatureCounts:
    """Classify occupancy-defined potential windows before blocking filters.

    Unlike proxy v1, the common population is determined only by occupancy.
    Segment blocking is canonical segment intersection. Touching blocking is a
    transition from same-direction-compatible before the played line to
    incompatible after its geometric direction usage is added. No successor
    state or successor legal-action enumeration is used.
    """
    mechanisms = tuple(
        classify_potential_latent_window(
            state=state,
            played_action=action,
            identity=identity,
        )
        for identity in enumerate_potential_latent_windows(state, action)
    )
    categories = {
        name: sum(mechanism.category == name for mechanism in mechanisms)
        for name in ("segment_only", "touching_only", "both", "neither")
    }
    return CorrectedLatentWindowMoveFeatureCounts(
        potential_window_count=len(mechanisms),
        promoted_latent_window_count=sum(
            mechanism.promoted for mechanism in mechanisms
        ),
        blocked_by_consumed_segment_count=sum(
            mechanism.blocked_by_consumed_segment for mechanism in mechanisms
        ),
        blocked_by_same_direction_touching_count=sum(
            mechanism.blocked_by_new_same_direction_touching for mechanism in mechanisms
        ),
        segment_only_count=categories["segment_only"],
        touching_only_count=categories["touching_only"],
        both_count=categories["both"],
        neither_count=categories["neither"],
        mechanisms=mechanisms,
    )


def enumerate_potential_latent_windows(
    state: MorpionState,
    action: MorpionAction,
) -> tuple[LatentWindowIdentity, ...]:
    """Enumerate unique three-occupied/two-missing windows before rule filters."""
    new_dot = morpion_action_new_point(action)
    identities: set[LatentWindowIdentity] = set()
    for direction, (dx, dy) in enumerate(DIRECTIONS):
        for new_dot_slot in range(5):
            origin_x = new_dot[0] - new_dot_slot * dx
            origin_y = new_dot[1] - new_dot_slot * dy
            probe: MorpionAction = (direction, origin_x, origin_y, 0)
            missing_before = tuple(
                slot
                for slot, point in enumerate(morpion_action_points(probe))
                if point not in state.points
            )
            if len(missing_before) != 2 or new_dot_slot not in missing_before:
                continue
            identities.add(
                LatentWindowIdentity(
                    direction=direction,
                    origin_x=origin_x,
                    origin_y=origin_y,
                    missing_slot_after_play=next(
                        slot for slot in missing_before if slot != new_dot_slot
                    ),
                )
            )
    return tuple(sorted(identities))


def classify_potential_latent_window(
    *,
    state: MorpionState,
    played_action: MorpionAction,
    identity: LatentWindowIdentity,
) -> PotentialLatentWindowMechanism:
    """Classify one potential window with independent canonical mechanisms."""
    candidate = identity.action
    candidate_segments = frozenset(morpion_action_segments(candidate))
    current_segments_available = not bool(candidate_segments & state.used_unit_segments)
    current_same_direction_compatible = _parallel_compatible_before_play(
        state, candidate
    )
    blocked_segment = bool(
        candidate_segments & frozenset(morpion_action_segments(played_action))
    )
    compatible_after = _parallel_compatible_after_play_geometry(
        state=state,
        candidate=candidate,
        played_action=played_action,
    )
    blocked_touching = current_same_direction_compatible and not compatible_after
    return PotentialLatentWindowMechanism(
        identity=identity,
        current_segments_available=current_segments_available,
        current_same_direction_compatible=current_same_direction_compatible,
        blocked_by_consumed_segment=blocked_segment,
        blocked_by_new_same_direction_touching=blocked_touching,
        promoted=(
            current_segments_available
            and current_same_direction_compatible
            and not blocked_segment
            and compatible_after
        ),
    )


def _parallel_compatible_before_play(
    state: MorpionState,
    candidate: MorpionAction,
) -> bool:
    direction, _x, _y, _missing = candidate
    return _parallel_compatible_with_extra_usage(
        state=state,
        candidate=candidate,
        extra_direction=direction,
        extra_points=(),
    )


def _parallel_compatible_after_play_geometry(
    *,
    state: MorpionState,
    candidate: MorpionAction,
    played_action: MorpionAction,
) -> bool:
    played_direction, _x, _y, _missing = played_action
    return _parallel_compatible_with_extra_usage(
        state=state,
        candidate=candidate,
        extra_direction=played_direction,
        extra_points=morpion_action_points(played_action),
    )


def _parallel_compatible_with_extra_usage(
    *,
    state: MorpionState,
    candidate: MorpionAction,
    extra_direction: int,
    extra_points: tuple[tuple[int, int], ...],
) -> bool:
    direction, _x, _y, _missing = candidate
    points = morpion_action_points(candidate)
    extra_usage = {
        point: (1 if slot in (0, 4) else 2) for slot, point in enumerate(extra_points)
    }
    if state.variant == Variant.DISJOINT_5D:
        return all(
            (point, direction) not in state.dir_usage
            and not (direction == extra_direction and point in extra_usage)
            for point in points
        )
    for slot, point in enumerate(points):
        wanted = 1 if slot in (0, 4) else 2
        existing = state.dir_usage.get((point, direction), 0)
        if direction == extra_direction:
            existing |= extra_usage.get(point, 0)
        if existing & 2 or (existing & 1 and wanted != 1):
            return False
    return True


def _parallel_compatible_with_context(
    context: MorpionLatentWindowFeatureContext,
    candidate: MorpionAction,
    *,
    extra_direction: int,
    extra_points: tuple[tuple[int, int], ...],
) -> bool:
    direction = candidate[0]
    points = morpion_action_points(candidate)
    extra_usage = {
        point: (1 if slot in (0, 4) else 2) for slot, point in enumerate(extra_points)
    }
    if context.variant == Variant.DISJOINT_5D:
        return all(
            (point, direction) not in context.direction_usage
            and not (direction == extra_direction and point in extra_usage)
            for point in points
        )
    for slot, point in enumerate(points):
        wanted = 1 if slot in (0, 4) else 2
        existing = context.direction_usage.get((point, direction), 0)
        if direction == extra_direction:
            existing |= extra_usage.get(point, 0)
        if existing & 2 or (existing & 1 and wanted != 1):
            return False
    return True


def _latent_window_state_fingerprint(
    state: MorpionState, legal_actions: tuple[MorpionAction, ...]
) -> str:
    payload = repr((
        tuple(sorted(state.points)),
        tuple(sorted(state.used_unit_segments)),
        state.dir_usage_entries,
        state.moves,
        state.variant.value,
        legal_actions,
    ))
    return hashlib.sha256(payload.encode()).hexdigest()


__all__ = [
    "MORPION_CORRECTED_BLOCKING_PROXY_VERSION",
    "MORPION_LATENT_WINDOW_IMPLEMENTATION",
    "MORPION_LATENT_WINDOW_PROXY_VERSION",
    "CorrectedLatentWindowMoveFeatureCounts",
    "LatentWindowIdentity",
    "LatentWindowMoveFeatureCounts",
    "MorpionLatentWindowFeatureContext",
    "MorpionLatentWindowInstrumentation",
    "PotentialLatentWindowMechanism",
    "build_morpion_latent_window_feature_context",
    "classify_potential_latent_window",
    "compute_promoted_latent_window_count_reference",
    "compute_promoted_latent_window_counts",
    "corrected_latent_window_move_feature_counts",
    "enumerate_potential_latent_windows",
    "latent_window_move_feature_counts",
]

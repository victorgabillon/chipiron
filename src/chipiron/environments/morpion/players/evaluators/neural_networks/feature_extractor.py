"""Handcrafted Morpion feature extraction for neural-network inputs."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Final

from atomheart.games.morpion.state import (
    Point,
    Segment,
    norm_seg,
)
from atomheart.games.morpion.state import (
    Variant as MorpionVariant,
)

from chipiron.environments.morpion.types import (
    MorpionDynamics,
    MorpionState,
)

DIRECTIONS: Final[tuple[Point, ...]] = (
    (1, 0),
    (0, 1),
    (1, 1),
    (1, -1),
)

FEATURE_NAMES: Final[tuple[str, ...]] = (
    "moves",
    "num_points",
    "num_used_unit_segments",
    "bbox_width",
    "bbox_height",
    "bbox_area",
    "point_density_in_bbox",
    "legal_action_count",
    "legal_actions_dir_0",
    "legal_actions_dir_1",
    "legal_actions_dir_2",
    "legal_actions_dir_3",
    "num_distinct_playable_cells",
    "mean_legal_actions_per_playable_cell",
    "max_legal_actions_per_playable_cell",
    "playable_cells_with_1_action",
    "playable_cells_with_2_actions",
    "playable_cells_with_ge_3_actions",
    "dir_usage_value_0_count",
    "dir_usage_value_1_count",
    "dir_usage_value_2_count",
    "dir_usage_value_3_count",
    "points_with_any_dir_usage_3",
    "points_with_ge_2_nonzero_dir_usages",
    "segments_4_present_1_missing_geometric",
    "segments_4_present_1_missing_overlap_ok",
    "segments_4_present_1_missing_parallel_ok",
    "segments_4_present_1_missing_legal",
    "segments_3_present_2_missing_geometric",
    "segments_3_present_2_missing_overlap_ok",
    "segments_3_present_2_missing_parallel_ok",
    "segments_3_present_2_missing_alive",
    "segments_4p1m_dir_0_legal",
    "segments_4p1m_dir_1_legal",
    "segments_4p1m_dir_2_legal",
    "segments_4p1m_dir_3_legal",
    "frontier_cell_count",
    "frontier_cells_in_any_ge3_candidate_segment",
    "frontier_cells_in_any_legal_4p1m_segment",
    "occupied_connected_components",
    "largest_occupied_component_size",
)


@dataclass(frozen=True, slots=True)
class CandidateSegment:
    """Analysis record for one scanned five-point Morpion segment."""

    dir_index: int
    start_x: int
    start_y: int
    points5: tuple[Point, Point, Point, Point, Point]
    unit_segments4: tuple[Segment, Segment, Segment, Segment]
    num_present: int
    num_missing: int
    missing_points: tuple[Point, ...]
    overlap_blocked: bool
    parallel_compatible: bool
    legal: bool


@dataclass(frozen=True, slots=True)
class _BoundingBox:
    """Axis-aligned bounding box over occupied lattice points."""

    min_x: int
    max_x: int
    min_y: int
    max_y: int

    @property
    def width(self) -> int:
        """Return inclusive lattice width."""
        return self.max_x - self.min_x + 1

    @property
    def height(self) -> int:
        """Return inclusive lattice height."""
        return self.max_y - self.min_y + 1

    @property
    def area(self) -> int:
        """Return inclusive lattice area."""
        return self.width * self.height


def morpion_feature_names() -> tuple[str, ...]:
    """Return deterministic Morpion handcrafted feature names."""
    return FEATURE_NAMES


def extract_morpion_features(
    state: MorpionState,
    dynamics: MorpionDynamics | None = None,
) -> OrderedDict[str, float]:
    """Extract deterministic scalar Morpion features from ``state``."""
    dyn = dynamics if dynamics is not None else MorpionDynamics()
    candidate_segments = _candidate_segments(state)
    values: dict[str, float] = {name: 0.0 for name in FEATURE_NAMES}

    values.update(_geometry_features(state))
    values.update(_legal_action_features(state=state, dynamics=dyn))
    values.update(_dir_usage_features(state))
    values.update(_candidate_segment_features(candidate_segments))
    values.update(_frontier_features(state=state, candidate_segments=candidate_segments))
    values.update(_connectivity_features(state.points))

    return OrderedDict((name, float(values[name])) for name in FEATURE_NAMES)


def _geometry_features(state: MorpionState) -> dict[str, float]:
    """Return size and bounding-box features."""
    bbox = _bbox(state.points)
    if bbox is None:
        return {
            "moves": float(state.moves),
            "num_points": 0.0,
            "num_used_unit_segments": float(len(state.used_unit_segments)),
            "bbox_width": 0.0,
            "bbox_height": 0.0,
            "bbox_area": 0.0,
            "point_density_in_bbox": 0.0,
        }

    return {
        "moves": float(state.moves),
        "num_points": float(len(state.points)),
        "num_used_unit_segments": float(len(state.used_unit_segments)),
        "bbox_width": float(bbox.width),
        "bbox_height": float(bbox.height),
        "bbox_area": float(bbox.area),
        "point_density_in_bbox": float(len(state.points)) / float(bbox.area),
    }


def _legal_action_features(
    *,
    state: MorpionState,
    dynamics: MorpionDynamics,
) -> dict[str, float]:
    """Return immediate-mobility features based on current legal actions."""
    raw_actions = dynamics.all_legal_actions(state)
    legal_actions_by_dir = [0, 0, 0, 0]
    playable_cell_action_counts: defaultdict[Point, int] = defaultdict(int)

    for action in raw_actions:
        dir_index, x0, y0, missing_i = action
        legal_actions_by_dir[dir_index] += 1
        playable_cell_action_counts[
            _missing_point_from_action(
                dir_index=dir_index,
                x0=x0,
                y0=y0,
                missing_i=missing_i,
            )
        ] += 1

    num_playable_cells = len(playable_cell_action_counts)
    total_cell_actions = sum(playable_cell_action_counts.values())
    mean_per_cell = (
        float(total_cell_actions) / float(num_playable_cells)
        if num_playable_cells > 0
        else 0.0
    )
    max_per_cell = max(playable_cell_action_counts.values(), default=0)

    return {
        "legal_action_count": float(len(raw_actions)),
        "legal_actions_dir_0": float(legal_actions_by_dir[0]),
        "legal_actions_dir_1": float(legal_actions_by_dir[1]),
        "legal_actions_dir_2": float(legal_actions_by_dir[2]),
        "legal_actions_dir_3": float(legal_actions_by_dir[3]),
        "num_distinct_playable_cells": float(num_playable_cells),
        "mean_legal_actions_per_playable_cell": mean_per_cell,
        "max_legal_actions_per_playable_cell": float(max_per_cell),
        "playable_cells_with_1_action": float(
            _count_cells_with_action_count(playable_cell_action_counts, 1)
        ),
        "playable_cells_with_2_actions": float(
            _count_cells_with_action_count(playable_cell_action_counts, 2)
        ),
        "playable_cells_with_ge_3_actions": float(
            sum(1 for count in playable_cell_action_counts.values() if count >= 3)
        ),
    }


def _dir_usage_features(state: MorpionState) -> dict[str, float]:
    """Return same-direction usage-value features over occupied points."""
    dir_usage = state.dir_usage
    usage_value_counts = [0, 0, 0, 0]
    points_with_any_dir_usage_3 = 0
    points_with_ge_2_nonzero_dir_usages = 0

    for point in state.points:
        nonzero_usage_count = 0
        has_usage_3 = False
        for dir_index in range(4):
            usage_value = dir_usage.get((point, dir_index), 0)
            if 0 <= usage_value <= 3:
                usage_value_counts[usage_value] += 1
            if usage_value != 0:
                nonzero_usage_count += 1
            if usage_value == 3:
                has_usage_3 = True
        if has_usage_3:
            points_with_any_dir_usage_3 += 1
        if nonzero_usage_count >= 2:
            points_with_ge_2_nonzero_dir_usages += 1

    return {
        "dir_usage_value_0_count": float(usage_value_counts[0]),
        "dir_usage_value_1_count": float(usage_value_counts[1]),
        "dir_usage_value_2_count": float(usage_value_counts[2]),
        "dir_usage_value_3_count": float(usage_value_counts[3]),
        "points_with_any_dir_usage_3": float(points_with_any_dir_usage_3),
        "points_with_ge_2_nonzero_dir_usages": float(
            points_with_ge_2_nonzero_dir_usages
        ),
    }


def _candidate_segment_features(
    candidate_segments: tuple[CandidateSegment, ...],
) -> dict[str, float]:
    """Return features aggregated over scanned five-point candidate segments."""
    features: dict[str, float] = {
        "segments_4_present_1_missing_geometric": 0.0,
        "segments_4_present_1_missing_overlap_ok": 0.0,
        "segments_4_present_1_missing_parallel_ok": 0.0,
        "segments_4_present_1_missing_legal": 0.0,
        "segments_3_present_2_missing_geometric": 0.0,
        "segments_3_present_2_missing_overlap_ok": 0.0,
        "segments_3_present_2_missing_parallel_ok": 0.0,
        "segments_3_present_2_missing_alive": 0.0,
        "segments_4p1m_dir_0_legal": 0.0,
        "segments_4p1m_dir_1_legal": 0.0,
        "segments_4p1m_dir_2_legal": 0.0,
        "segments_4p1m_dir_3_legal": 0.0,
    }

    for candidate_segment in candidate_segments:
        if candidate_segment.num_present == 4 and candidate_segment.num_missing == 1:
            features["segments_4_present_1_missing_geometric"] += 1.0
            if not candidate_segment.overlap_blocked:
                features["segments_4_present_1_missing_overlap_ok"] += 1.0
            if candidate_segment.parallel_compatible:
                features["segments_4_present_1_missing_parallel_ok"] += 1.0
            if candidate_segment.legal:
                features["segments_4_present_1_missing_legal"] += 1.0
                features[
                    f"segments_4p1m_dir_{candidate_segment.dir_index}_legal"
                ] += 1.0

        if candidate_segment.num_present == 3 and candidate_segment.num_missing == 2:
            features["segments_3_present_2_missing_geometric"] += 1.0
            if not candidate_segment.overlap_blocked:
                features["segments_3_present_2_missing_overlap_ok"] += 1.0
            if candidate_segment.parallel_compatible:
                features["segments_3_present_2_missing_parallel_ok"] += 1.0
            if (
                not candidate_segment.overlap_blocked
                and candidate_segment.parallel_compatible
            ):
                features["segments_3_present_2_missing_alive"] += 1.0

    return features


def _frontier_features(
    *,
    state: MorpionState,
    candidate_segments: tuple[CandidateSegment, ...],
) -> dict[str, float]:
    """Return empty-neighbor features tied to promising candidate segments."""
    frontier_cells = _frontier_cells(state.points)
    missing_candidate_cells = {
        point
        for candidate_segment in candidate_segments
        if candidate_segment.num_present >= 3
        for point in candidate_segment.missing_points
    }
    legal_4p1m_cells = {
        candidate_segment.missing_points[0]
        for candidate_segment in candidate_segments
        if candidate_segment.legal
    }

    return {
        "frontier_cell_count": float(len(frontier_cells)),
        "frontier_cells_in_any_ge3_candidate_segment": float(
            len(frontier_cells & missing_candidate_cells)
        ),
        "frontier_cells_in_any_legal_4p1m_segment": float(
            len(frontier_cells & legal_4p1m_cells)
        ),
    }


def _connectivity_features(points: frozenset[Point]) -> dict[str, float]:
    """Return 8-neighborhood connectivity features over occupied points."""
    component_sizes = _connected_component_sizes_8(points)
    return {
        "occupied_connected_components": float(len(component_sizes)),
        "largest_occupied_component_size": float(max(component_sizes, default=0)),
    }


def _candidate_segments(state: MorpionState) -> tuple[CandidateSegment, ...]:
    """Scan the same bounded lattice box used by atomheart raw action generation."""
    bbox = _bbox(state.points)
    if bbox is None:
        return ()

    margin = 4
    min_x = bbox.min_x - margin
    max_x = bbox.max_x + margin
    min_y = bbox.min_y - margin
    max_y = bbox.max_y + margin
    candidate_segments: list[CandidateSegment] = []

    for dir_index, direction in enumerate(DIRECTIONS):
        for x0 in range(min_x, max_x + 1):
            for y0 in range(min_y, max_y + 1):
                points5 = _line_points(x0=x0, y0=y0, direction=direction)
                missing_points = tuple(
                    point for point in points5 if point not in state.points
                )
                unit_segments4 = _unit_segments_on_line(points5)
                overlap_blocked = any(
                    segment in state.used_unit_segments for segment in unit_segments4
                )
                parallel_compatible = _is_parallel_compatible(
                    state=state,
                    dir_index=dir_index,
                    points5=points5,
                )
                candidate_segments.append(
                    CandidateSegment(
                        dir_index=dir_index,
                        start_x=x0,
                        start_y=y0,
                        points5=points5,
                        unit_segments4=unit_segments4,
                        num_present=5 - len(missing_points),
                        num_missing=len(missing_points),
                        missing_points=missing_points,
                        overlap_blocked=overlap_blocked,
                        parallel_compatible=parallel_compatible,
                        legal=(
                            len(missing_points) == 1
                            and not overlap_blocked
                            and parallel_compatible
                        ),
                    )
                )

    return tuple(candidate_segments)


def _bbox(points: frozenset[Point]) -> _BoundingBox | None:
    """Return an inclusive bounding box or ``None`` for an empty state."""
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return _BoundingBox(
        min_x=min(xs),
        max_x=max(xs),
        min_y=min(ys),
        max_y=max(ys),
    )


# These helpers intentionally mirror atomheart's Morpion dynamics geometry and
# same-direction compatibility logic.
def _line_points(
    *,
    x0: int,
    y0: int,
    direction: Point,
) -> tuple[Point, Point, Point, Point, Point]:
    """Return the five lattice points in one candidate Morpion line."""
    dx, dy = direction
    return (
        (x0 + 0 * dx, y0 + 0 * dy),
        (x0 + 1 * dx, y0 + 1 * dy),
        (x0 + 2 * dx, y0 + 2 * dy),
        (x0 + 3 * dx, y0 + 3 * dy),
        (x0 + 4 * dx, y0 + 4 * dy),
    )


def _unit_segments_on_line(
    points5: tuple[Point, Point, Point, Point, Point],
) -> tuple[Segment, Segment, Segment, Segment]:
    """Return the four normalized unit segments composing a five-point line."""
    return (
        norm_seg(points5[0], points5[1]),
        norm_seg(points5[1], points5[2]),
        norm_seg(points5[2], points5[3]),
        norm_seg(points5[3], points5[4]),
    )


def _is_parallel_compatible(
    *,
    state: MorpionState,
    dir_index: int,
    points5: tuple[Point, Point, Point, Point, Point],
) -> bool:
    """Mirror atomheart's 5T/5D same-direction compatibility check."""
    dir_usage = state.dir_usage
    if state.variant == MorpionVariant.DISJOINT_5D:
        return all((point, dir_index) not in dir_usage for point in points5)

    for index, point in enumerate(points5):
        want = _point_usage_kind(index)
        have = dir_usage.get((point, dir_index), 0)
        if have == 0:
            continue
        if have & 2:
            return False
        if have & 1 and want != 1:
            return False
    return True


def _point_usage_kind(index: int) -> int:
    """Return atomheart's endpoint/middle bitmask for one line-point index."""
    return 1 if index in (0, 4) else 2


def _missing_point_from_action(
    *,
    dir_index: int,
    x0: int,
    y0: int,
    missing_i: int,
) -> Point:
    """Return the absent point represented by a raw Morpion action."""
    direction = DIRECTIONS[dir_index]
    dx, dy = direction
    return (x0 + missing_i * dx, y0 + missing_i * dy)


def _count_cells_with_action_count(
    playable_cell_action_counts: defaultdict[Point, int],
    expected_count: int,
) -> int:
    """Count playable cells having exactly ``expected_count`` legal actions."""
    return sum(
        1 for count in playable_cell_action_counts.values() if count == expected_count
    )


def _frontier_cells(points: frozenset[Point]) -> frozenset[Point]:
    """Return empty cells in Chebyshev distance one from occupied points."""
    frontier_cells: set[Point] = set()
    for point in points:
        x, y = point
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                candidate = (x + dx, y + dy)
                if candidate not in points:
                    frontier_cells.add(candidate)
    return frozenset(frontier_cells)


def _connected_component_sizes_8(points: frozenset[Point]) -> tuple[int, ...]:
    """Return occupied-point component sizes under 8-neighborhood adjacency."""
    unvisited = set(points)
    component_sizes: list[int] = []

    while unvisited:
        stack = [unvisited.pop()]
        component_size = 0
        while stack:
            x, y = stack.pop()
            component_size += 1
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    neighbor = (x + dx, y + dy)
                    if neighbor in unvisited:
                        unvisited.remove(neighbor)
                        stack.append(neighbor)
        component_sizes.append(component_size)

    return tuple(component_sizes)


__all__ = [
    "DIRECTIONS",
    "FEATURE_NAMES",
    "CandidateSegment",
    "extract_morpion_features",
    "morpion_feature_names",
]
